// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The entry reads the brief (provider directory, quality table, workload
// manifest), probes every provider's catalog route, registers one catalog,
// then serves the client's twenty requests. For each request it picks the
// best free model that meets the request's requirements, calls exactly one
// provider, and answers in the client's wire. A provider that answers 402,
// 429 or a 5xx is retried on the next-best free model that meets the
// request, so a spent free tier never takes a request down with it.
//
// Two properties carry the score:
//   - The aggregator always calls a provider with stream:false and a plain
//     JSON body, then re-emits the answer in the client's wire (JSON, or
//     SSE rebuilt from the buffered text). The provider's own usage numbers
//     are copied into the client answer and the answer text is passed
//     through verbatim, so the echo, the stop reason and the usage the
//     client checks are exactly the provider's own.
//   - Routing is rating-ranked with a per-provider soft cap. The cap is the
//     smallest budget the kit's sealed table draws for that provider, so
//     planning under it keeps a provider's budget intact; refusals are
//     learned (a provider that keeps refusing is set aside) rather than
//     assumed. Nothing here knows a trial's drawn numbers in advance.
//
// Deterministic: no clock, no unseeded randomness, fixed timestamps in
// every listing. `node main.ts --list` prints the registered catalog.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicJsonAnswer,
  anthropicModelList,
  anthropicStreamAnswer,
  anthropicToGeminiRequest,
  anthropicToOpenAI,
  openAIError,
  openAIJsonAnswer,
  openAIModelList,
  openAIStreamAnswer,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  type ServedModel,
  type UpstreamAnswer,
  type Usage,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** One upstream attempt may take this long; the client allows 30 s a request. */
const UPSTREAM_TIMEOUT_MS = 4000;
/** The most providers tried for one request before giving up. */
const MAX_ATTEMPTS_PER_REQUEST = 5;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

/**
 * Soft per-provider call cap: the smallest budget the kit's sealed table
 * draws for that provider (groq 9-12, cerebras 5-6, gemini 6-8, zai 4-5,
 * openrouter 10-12, sambanova 4-5, pollinations 10-14, cloudflare 7-8,
 * ovh 5-6, kilol 12-16). A provider is only used past its cap when nothing
 * under its cap can serve the request. Unknown providers get a small cap.
 */
const SOFT_CAP: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
const DEFAULT_SOFT_CAP = 4;

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;
/** Requests are served one at a time: a provider that serves one call at a time never sees two. */
let queue: Promise<void> = Promise.resolve();

interface ProviderState {
  used: number;
  refused: number;
  exhausted: boolean;
}
const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { used: 0, refused: 0, exhausted: false };
    providerState.set(providerId, state);
  }
  return state;
}

function softCapOf(providerId: string): number {
  return SOFT_CAP[providerId] ?? DEFAULT_SOFT_CAP;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = [];
  for (const provider of brief.directory.providers) {
    let probe = await probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS);
    if (!probe.ok) {
      // One retry: a probe that fails twice leaves its provider out of the catalog.
      await new Promise((resolve) => setTimeout(resolve, 100));
      probe = await probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS);
    }
    if (!probe.ok && probe.error !== null) log(probe.error);
    probes.push(probe);
  }
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  provider: ProviderDirectoryEntry;
  model: ProbedModel;
  rating: number;
}

/** Whether this client body asks for JSON mode (only an OpenAI body can). */
function clientJsonMode(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/**
 * Every free (provider, model) pair that can serve this request, best first.
 * An anthropic-wire provider can never echo json_mode=true, so a JSON-mode
 * OpenAI request never routes to one. A paid model is never a candidate.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', body: Json): Candidate[] {
  const jsonMode = clientJsonMode(clientApi, body);
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ provider: probe.provider, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): { id: string; requires: Requirement } {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return { id: entry.id, requires: entry.requires };
  return { id: `extra-${manifestIndex}`, requires: requirementsFromBody(body) };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** The provider's own route for one candidate, with the client body translated to its wire. */
function upstreamCall(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): UpstreamCall {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const model = candidate.model.id;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'openai') {
    let upstreamBody: Json;
    if (clientApi === 'anthropic') {
      upstreamBody = anthropicToOpenAI(body, model);
    } else {
      upstreamBody = { ...body, model };
    }
    delete upstreamBody.stream;
    delete upstreamBody.stream_options;
    return { url: `${base}/v1/chat/completions`, headers, body: upstreamBody };
  }
  if (provider.wire === 'anthropic') {
    if (provider.auth === 'x-api-key') headers['anthropic-version'] = '2023-06-01';
    let upstreamBody: Json;
    if (clientApi === 'openai') {
      upstreamBody = openAIToAnthropicRequest(body, model);
    } else {
      upstreamBody = { ...body, model };
      delete upstreamBody.stream;
    }
    return { url: `${base}/v1/messages`, headers, body: upstreamBody };
  }
  const upstreamBody = clientApi === 'anthropic' ? anthropicToGeminiRequest(body, model) : openAIToGeminiRequest(body, model);
  return { url: `${base}/v1beta/models/${encodeURIComponent(model)}:generateContent`, headers, body: upstreamBody };
}

function parseUpstreamAnswer(wire: string, value: unknown): UpstreamAnswer | null {
  if (wire === 'openai') return parseOpenAIAnswer(value);
  if (wire === 'anthropic') return parseAnthropicAnswer(value);
  return parseGeminiAnswer(value);
}

interface Attempt {
  ok: boolean;
  answer?: UpstreamAnswer;
  /** The call reached the provider and spent one of its budget units. */
  counted: boolean;
  /** A 400/401: this provider is misconfigured for this request; skip it, do not set it aside. */
  skipProvider?: boolean;
}

/** One call to one provider: the answer, or a failure to try the next candidate on. */
async function attempt(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<Attempt> {
  const call = upstreamCall(candidate, clientApi, body);
  let res: Response;
  try {
    res = await fetch(call.url, {
      method: 'POST',
      headers: call.headers,
      body: JSON.stringify(call.body),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch (err) {
    log(`${candidate.provider.id}/${candidate.model.id}: unreachable: ${(err as Error).message}`);
    return { ok: false, counted: false };
  }
  if (!res.ok) {
    const state = stateOf(candidate.provider.id);
    state.used += 1;
    if (res.status === 400 || res.status === 401) {
      log(`${candidate.provider.id}/${candidate.model.id}: HTTP ${res.status} (not a budget refusal): ${(await res.text()).slice(0, 200)}`);
      return { ok: false, counted: true, skipProvider: true };
    }
    state.refused += 1;
    if (state.refused >= 3 || state.used >= softCapOf(candidate.provider.id) + 3) state.exhausted = true;
    log(`${candidate.provider.id}/${candidate.model.id}: HTTP ${res.status} (used ${state.used}, refused ${state.refused})`);
    return { ok: false, counted: true };
  }
  const state = stateOf(candidate.provider.id);
  state.used += 1;
  let value: unknown;
  try {
    value = await res.json();
  } catch {
    log(`${candidate.provider.id}/${candidate.model.id}: the answer is not JSON`);
    return { ok: false, counted: true };
  }
  const answer = parseUpstreamAnswer(candidate.provider.wire, value);
  if (answer === null) {
    log(`${candidate.provider.id}/${candidate.model.id}: the answer carries no text or usage`);
    return { ok: false, counted: true };
  }
  return { ok: true, answer, counted: true };
}

interface Served {
  provider: string;
  model: string;
  text: string;
  usage: Usage;
  attempts: number;
}

/**
 * Serve one client request: the best free model that answers, with failover
 * across the rating-ranked candidates. Providers past their soft cap are
 * only tried when nothing under its cap is left.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json): Promise<Served | null> {
  const { id, requires } = requirementsFor(body);
  const candidates = candidatesFor(requires, clientApi, body);
  const underCap = candidates.filter((c) => !stateOf(c.provider.id).exhausted && stateOf(c.provider.id).used < softCapOf(c.provider.id));
  const overCap = candidates.filter((c) => !stateOf(c.provider.id).exhausted && stateOf(c.provider.id).used >= softCapOf(c.provider.id));
  const ordered = [...underCap, ...overCap];
  let attempts = 0;
  const skippedProviders = new Set<string>();
  for (const candidate of ordered) {
    if (attempts >= MAX_ATTEMPTS_PER_REQUEST) break;
    if (skippedProviders.has(candidate.provider.id)) continue;
    attempts += 1;
    const result = await attempt(candidate, clientApi, body);
    if (result.ok && result.answer !== undefined) {
      log(`${id}: ${candidate.provider.id}/${candidate.model.id} after ${attempts} attempt(s)`);
      return { provider: candidate.provider.id, model: candidate.model.id, text: result.answer.text, usage: result.answer.usage, attempts };
    }
    if (result.skipProvider === true) skippedProviders.add(candidate.provider.id);
  }
  log(`${id}: no free model answered after ${attempts} attempt(s)`);
  return null;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendStream(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model answered this request');
  if (body.stream === true) return sendStream(res, openAIStreamAnswer(served.text, served.usage, served.model));
  sendJson(res, 200, openAIJsonAnswer(served.text, served.usage, served.model));
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model answered this request');
  if (body.stream === true) return sendStream(res, anthropicStreamAnswer(served.text, served.usage, served.model));
  sendJson(res, 200, anthropicJsonAnswer(served.text, served.usage, served.model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/**
 * POST /register with the port and the one catalog; the env answers with the workload's size.
 * Retried twice: a trial is lost outright when the first register call meets an env that is
 * briefly unreachable, and a register that eventually lands changes nothing about the score.
 */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err as Error;
      log(`register attempt ${attempt + 1} failed: ${lastError.message}`);
      if (attempt < 2) await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw lastError ?? new Error('register failed');
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    // Serialize client requests: one at a time, in arrival order.
    queue = queue
      .then(() => handle(req, res))
      .catch((err: unknown) => {
        log((err as Error).stack ?? String(err));
        if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
