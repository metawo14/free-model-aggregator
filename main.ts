// main.ts — free-model-aggregator-01 entry.
//
// One long-running program: it reads the brief ($MAKEFUN_PROMPT_FILE), starts
// an HTTP server on 0.0.0.0, registers with the env ($MAKEFUN_ENV_URL), and
// answers the 20 requests the env's client sends, one at a time, then exits 0.
// Every answer comes from one of the ten providers, through the env, buffered
// upstream and re-emitted downstream in the client's wire. No clock and no
// randomness reach any routing decision, so a replay leaves the same export.
//
// `node main.ts --list` prints the catalog document and exits 0 (with no brief
// to read it prints {"providers":[]}).

import { readFileSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import {
  buildCatalog,
  buildProviders,
  freshState,
  rankCandidates,
  type Candidate,
  type DirectoryRow,
  type ProviderInfo,
  type ProviderState,
  type RatingRow,
} from "./lib/providers.ts";
import {
  anthropicJsonBody,
  anthropicStreamBody,
  buildAnthropic,
  buildGemini,
  buildOpenAI,
  FIXED_CREATED,
  mapStop,
  openAIJsonBody,
  openAIStreamBody,
  parseClient,
  parseUpstream,
  type Canonical,
  type ClientApi,
} from "./lib/wire.ts";

const PROTOCOL_VERSION = 1;
const UPSTREAM_TIMEOUT_MS = 10000;
const WORK_POLL_MS = 100;
const MAX_BODY_BYTES = 2 * 1024 * 1024;

function log(message: string): void {
  process.stderr.write(`free-model-aggregator-01: ${message}\n`);
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

interface Brief {
  directory: DirectoryRow[];
  ratings: RatingRow[];
}

// The brief holds three JSON documents, each on its own line after a label:
// the provider directory, the ratings table, the workload manifest. Pull each
// one out by its schema marker.
function readBrief(path: string): Brief {
  const text = readFileSync(path, "utf8");
  let directory: DirectoryRow[] | null = null;
  let ratings: RatingRow[] | null = null;
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('{"schema":"free-model-aggregator-01/')) continue;
    let value: unknown;
    try {
      value = JSON.parse(trimmed);
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    if (value.schema === "free-model-aggregator-01/providers/v1" && Array.isArray(value.providers)) {
      directory = value.providers as DirectoryRow[];
    } else if (value.schema === "free-model-aggregator-01/ratings/v1" && Array.isArray(value.ratings)) {
      ratings = value.ratings as RatingRow[];
    }
  }
  if (directory === null) throw new Error("the brief holds no provider directory");
  if (ratings === null) throw new Error("the brief holds no ratings table");
  return { directory, ratings };
}

async function postJson(url: string, body: unknown, timeoutMs: number): Promise<{ status: number; text: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    return { status: res.status, text: await res.text() };
  } finally {
    clearTimeout(timer);
  }
}

async function getJson(url: string, timeoutMs: number): Promise<{ status: number; text: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { signal: controller.signal });
    return { status: res.status, text: await res.text() };
  } finally {
    clearTimeout(timer);
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// --- Upstream dialling ------------------------------------------------------

function upstreamHeaders(p: ProviderInfo): Record<string, string> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (p.auth === "bearer") headers.authorization = `Bearer ${p.key}`;
  else if (p.auth === "x-api-key") {
    headers["x-api-key"] = p.key;
    headers["anthropic-version"] = "2023-06-01";
  } else if (p.auth === "x-goog-api-key") headers["x-goog-api-key"] = p.key;
  return headers;
}

async function callUpstream(
  envUrl: string,
  provider: ProviderInfo,
  modelId: string,
  canonical: Canonical,
): Promise<
  | { ok: true; text: string; usage: { input: number; output: number }; stop: string }
  | { ok: false; status: number; code: string; message: string }
> {
  let path: string;
  let body: Record<string, unknown>;
  if (provider.wire === "openai") {
    path = "/v1/chat/completions";
    body = buildOpenAI(canonical, modelId);
  } else if (provider.wire === "anthropic") {
    path = "/v1/messages";
    body = buildAnthropic(canonical, modelId);
  } else {
    const built = buildGemini(canonical, modelId);
    path = built.path;
    body = built.body;
  }
  const url = `${envUrl}${provider.path}${path}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: upstreamHeaders(provider),
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await res.text();
    const parsed = parseUpstream(provider.wire, res.status, text);
    if (!parsed.ok) {
      return {
        ok: false,
        status: parsed.failure.status,
        code: parsed.failure.code,
        message: parsed.failure.message,
      };
    }
    return {
      ok: true,
      text: parsed.answer.text,
      usage: parsed.answer.usage,
      stop: parsed.answer.stop,
    };
  } catch {
    return { ok: false, status: 0, code: "fetch_failed", message: "the provider could not be reached" };
  } finally {
    clearTimeout(timer);
  }
}

// --- Request serving --------------------------------------------------------

interface ManifestRow {
  id: string;
  api: ClientApi;
  stream: boolean;
  weight: number;
  requires: { tools?: boolean; vision?: boolean; json?: boolean; minContext?: number };
}

interface ServerContext {
  envUrl: string;
  providers: ProviderInfo[];
  states: Map<string, ProviderState>;
  manifest: ManifestRow[] | null;
  servedRows: Set<string>;
}

interface ServedAnswer {
  status: number;
  contentType: string;
  body: string;
}

// Try the candidates in order. A refusal is retried once on the same model —
// a provider's failure schedule hits single call numbers, so a retry often
// lands on a good one and keeps the top rating (worth it outright for a
// weight-2 request, neutral for weight 1). But the env's refusal message says
// which refusals are permanent for the trial — a spent budget ("budget is
// spent"), a begun outage ("is out"), a wrong key — and those are never
// retried: the provider is set aside at once, which also saves the wasted
// call. Two consecutive refusals of any kind set the provider aside.
// Returns the served answer, or null when no candidate answered.
async function tryCandidates(
  ctx: ServerContext,
  candidates: Candidate[],
  canonical: Canonical,
  api: ClientApi,
  skipProviders: Set<string>,
): Promise<{ answer: ServedAnswer | null; attempted: Set<string> }> {
  const attempted = new Set<string>();
  for (const candidate of candidates) {
    if (skipProviders.has(candidate.provider.id)) continue;
    const state = ctx.states.get(candidate.provider.id) ?? freshState();
    ctx.states.set(candidate.provider.id, state);
    attempted.add(candidate.provider.id);
    for (let attempt = 0; attempt < 2; attempt += 1) {
      state.attempts += 1;
      const result = await callUpstream(
        ctx.envUrl,
        candidate.provider,
        candidate.model.id,
        canonical,
      );
      if (result.ok) {
        state.consecutiveFailures = 0;
        const stop = mapStop(result.stop, api);
        if (api === "openai") {
          if (canonical.stream) {
            return {
              answer: {
                status: 200,
                contentType: "text/event-stream; charset=utf-8",
                body: openAIStreamBody(candidate.model.id, result.text, result.usage, stop),
              },
              attempted,
            };
          }
          return {
            answer: {
              status: 200,
              contentType: "application/json",
              body: openAIJsonBody(candidate.model.id, result.text, result.usage, stop),
            },
            attempted,
          };
        }
        if (canonical.stream) {
          return {
            answer: {
              status: 200,
              contentType: "text/event-stream; charset=utf-8",
              body: anthropicStreamBody(candidate.model.id, result.text, result.usage, stop),
            },
            attempted,
          };
        }
        return {
          answer: {
            status: 200,
            contentType: "application/json",
            body: anthropicJsonBody(candidate.model.id, result.text, result.usage, stop),
          },
          attempted,
        };
      }
      // A refusal is not an answer: count it, retry once when it may be a
      // one-off, then fail over. A permanent refusal sets the provider aside
      // at once with no retry.
      state.consecutiveFailures += 1;
      log(
        `${candidate.provider.id}/${candidate.model.id} refused (${result.status} ${result.code}); failing over`,
      );
      if (
        state.consecutiveFailures >= 2 ||
        /budget is spent| is out$|key is missing or wrong/i.test(result.message)
      ) {
        state.setAside = true;
        break;
      }
    }
  }
  return { answer: null, attempted };
}

async function serveRequest(
  ctx: ServerContext,
  api: ClientApi,
  body: Record<string, unknown>,
): Promise<ServedAnswer> {
  const canonical = parseClient(api, body);
  const jsonModeOpenAI = api === "openai" && canonical.jsonMode;
  const requires = manifestRequiresFor(ctx, canonical);
  // First pass: respect soft caps and set-asides. Second pass (only when the
  // first served nothing): ignore them as a last resort — a provider set aside
  // on two adjacent one-off failures, or a budget draw above the soft cap,
  // can still answer, and the attempt costs nothing but a call.
  const first = await tryCandidates(
    ctx,
    rankCandidates(ctx.providers, ctx.states, requires, {
      jsonModeOpenAI,
      respectBudgets: true,
    }),
    canonical,
    api,
    new Set<string>(),
  );
  if (first.answer !== null) return first.answer;
  const second = await tryCandidates(
    ctx,
    rankCandidates(ctx.providers, ctx.states, requires, {
      jsonModeOpenAI,
      respectBudgets: false,
    }),
    canonical,
    api,
    first.attempted,
  );
  if (second.answer !== null) return second.answer;
  return {
    status: 502,
    contentType: "application/json",
    body: JSON.stringify({ error: "no provider answered this request" }),
  };
}

// The manifest's per-request requirements, keyed by the request's shape. The
// brief's workload manifest lists them, but the client bodies do not carry the
// request id, so the entry matches a request to its manifest row by its wire
// and its requirement flags — which is exactly how the generator built the
// bodies. min_context is the one flag a body cannot show; it is read from the
// manifest row whose shape matches (openai + long-log bodies are the
// min_context rows). When no row matches, fall back to the body-derived flags.
function manifestRequiresFor(
  ctx: ServerContext,
  canonical: Canonical,
): { tools?: boolean; vision?: boolean; json?: boolean; minContext?: number } {
  const derived: { tools?: boolean; vision?: boolean; json?: boolean; minContext?: number } = {};
  if (canonical.tools.length > 0) derived.tools = true;
  if (canonical.image !== null) derived.vision = true;
  if (canonical.jsonMode) derived.json = true;
  const rows = ctx.manifest;
  if (rows === null) return derived;
  const matches = rows.filter((row) => {
    if (row.api !== canonical.api) return false;
    if ((row.requires.tools === true) !== (derived.tools === true)) return false;
    if ((row.requires.vision === true) !== (derived.vision === true)) return false;
    if ((row.requires.json === true) !== (derived.json === true)) return false;
    return true;
  });
  // Among the rows that match the body's flags, prefer one the entry has not
  // served yet, in manifest order — the client sends the manifest in order.
  for (const row of matches) {
    if (!ctx.servedRows.has(row.id)) {
      ctx.servedRows.add(row.id);
      return row.requires;
    }
  }
  return derived;
}

// --- HTTP plumbing ----------------------------------------------------------

function readBody(req: IncomingMessage): Promise<{ text: string; tooLarge: boolean }> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let tooLarge = false;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) tooLarge = true;
      else chunks.push(chunk);
    });
    req.on("end", () => resolve({ text: Buffer.concat(chunks).toString("utf8"), tooLarge }));
    req.on("error", () => resolve({ text: "", tooLarge: true }));
  });
}

function sendJson(res: ServerResponse, status: number, value: unknown): void {
  const body = `${JSON.stringify(value)}\n`;
  res.writeHead(status, { "content-type": "application/json", "content-length": String(Buffer.byteLength(body)) });
  res.end(body);
}

function sendText(res: ServerResponse, status: number, contentType: string, body: string): void {
  res.writeHead(status, { "content-type": contentType, "content-length": String(Buffer.byteLength(body)) });
  res.end(body);
}

// --- main -------------------------------------------------------------------

async function main(): Promise<number> {
  const args = process.argv.slice(2);
  const listMode = args.includes("--list");

  const promptFile = process.env.MAKEFUN_PROMPT_FILE ?? "";
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/$/, "");

  let brief: Brief | null = null;
  if (promptFile !== "") {
    try {
      brief = readBrief(promptFile);
    } catch (err) {
      log(`cannot read the brief: ${(err as Error).message}`);
      if (listMode) {
        process.stdout.write('{"providers":[]}\n');
        return 0;
      }
      return 1;
    }
  }

  // --list prints the catalog and exits, without starting the server.
  if (listMode) {
    if (brief === null) {
      process.stdout.write('{"providers":[]}\n');
      return 0;
    }
    const catalog = buildCatalog(buildProviders(brief.directory, brief.ratings));
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return 0;
  }

  if (brief === null || envUrl === "") {
    log("a trial needs MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL");
    return 1;
  }

  const providers = buildProviders(brief.directory, brief.ratings);
  const catalog = buildCatalog(providers);
  const catalogJson = JSON.stringify(catalog);
  const states = new Map<string, ProviderState>();
  for (const p of providers) states.set(p.id, freshState());

  // The manifest, for requirement lookup (see manifestRequiresFor).
  const manifest = readManifest(promptFile);

  const ctx: ServerContext = {
    envUrl,
    providers,
    states,
    manifest,
    servedRows: new Set<string>(),
  };

  // The client sends one request at a time; serialise handling anyway so two
  // upstream calls can never be in flight to the same provider at once.
  let queue: Promise<void> = Promise.resolve();

  const server = createServer((req, res) => {
    queue = queue.then(async () => {
      try {
        await handle(req, res, ctx);
      } catch (err) {
        log(`handler error: ${(err as Error).message}`);
        if (!res.writableEnded) sendJson(res, 500, { error: "internal" });
      }
    });
  });

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen({ port: 0, host: "0.0.0.0" }, () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === "string") {
    log("the server has no TCP address");
    return 1;
  }
  const port = address.port;
  log(`listening on 0.0.0.0:${port}`);

  // Register with the env; retry briefly — the env may still be starting.
  let registered = false;
  for (let attempt = 1; attempt <= 30; attempt += 1) {
    try {
      const answer = await postJson(`${envUrl}/register`, {
        protocol_version: PROTOCOL_VERSION,
        port,
        catalog: JSON.parse(catalogJson) as unknown,
      }, 5000);
      if (answer.status === 200) {
        registered = true;
        log(`registered (${answer.text.trim()})`);
        break;
      }
      log(`register attempt ${attempt} answered HTTP ${answer.status}`);
    } catch (err) {
      log(`register attempt ${attempt} failed: ${(err as Error).message}`);
    }
    await sleep(500);
  }
  if (!registered) {
    log("could not register with the env");
    server.close();
    return 1;
  }

  // Poll /work until the client is done, then exit 0.
  for (;;) {
    await sleep(WORK_POLL_MS);
    try {
      const answer = await getJson(`${envUrl}/work`, 5000);
      if (answer.status !== 200) continue;
      const value = JSON.parse(answer.text) as { done?: boolean };
      if (value.done === true) break;
    } catch {
      // A missed poll is not fatal; keep polling.
    }
  }
  // Let any in-flight handler finish before closing.
  await queue;
  server.close();
  log("work done, exiting 0");
  return 0;
}

async function handle(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: ServerContext,
): Promise<void> {
  const url = new URL(req.url ?? "/", "http://entry");
  const method = req.method ?? "GET";
  if (method === "GET" && url.pathname === "/v1/models") {
    const rows = ctx.providers.flatMap((p) =>
      p.models.filter((m) => m.free).map((m) => ({ model: m, provider: p })),
    );
    if (req.headers["x-api-key"] !== undefined) {
      // Anthropic shape.
      sendJson(res, 200, {
        data: rows.map(({ model }) => ({
          type: "model",
          id: model.id,
          display_name: model.id,
          created_at: "2026-01-01T00:00:00Z",
          free: true,
          context_length: model.contextLength,
          tools: model.tools,
          vision: model.vision,
          json: model.json,
        })),
        has_more: false,
      });
    } else {
      // OpenAI shape (also the default when no auth header is sent).
      sendJson(res, 200, {
        object: "list",
        data: rows.map(({ model, provider }) => ({
          id: model.id,
          object: "model",
          created: FIXED_CREATED,
          owned_by: provider.id,
          free: true,
          context_length: model.contextLength,
          tools: model.tools,
          vision: model.vision,
          json: model.json,
        })),
      });
    }
    return;
  }
  if (
    method === "POST" &&
    (url.pathname === "/v1/chat/completions" || url.pathname === "/v1/messages")
  ) {
    const { text, tooLarge } = await readBody(req);
    if (tooLarge) {
      sendJson(res, 413, { error: "too_large" });
      return;
    }
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      sendJson(res, 400, { error: "bad_json" });
      return;
    }
    if (!isRecord(value)) {
      sendJson(res, 400, { error: "bad_body" });
      return;
    }
    const api: ClientApi = url.pathname === "/v1/chat/completions" ? "openai" : "anthropic";
    const answer = await serveRequest(ctx, api, value);
    sendText(res, answer.status, answer.contentType, answer.body);
    return;
  }
  sendJson(res, 404, { error: "not_found" });
}

// Read the workload manifest out of the brief (same line-scan as readBrief).
function readManifest(promptFile: string): ManifestRow[] | null {
  try {
    const text = readFileSync(promptFile, "utf8");
    for (const line of text.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('{"schema":"free-model-aggregator-01/workload/v1"')) continue;
      const value = JSON.parse(trimmed) as {
        manifest?: {
          id: string;
          api: string;
          stream: boolean;
          weight: number;
          requires?: Record<string, unknown>;
        }[];
      };
      if (!Array.isArray(value.manifest)) return null;
      return value.manifest.map((row) => ({
        id: row.id,
        api: (row.api === "anthropic" ? "anthropic" : "openai") as ClientApi,
        stream: row.stream,
        weight: row.weight,
        requires: {
          tools: row.requires?.tools === true ? true : undefined,
          vision: row.requires?.vision === true ? true : undefined,
          json: row.requires?.json === true ? true : undefined,
          minContext: typeof row.requires?.min_context === "number"
            ? row.requires.min_context
            : undefined,
        },
      }));
    }
  } catch {
    return null;
  }
  return null;
}

process.exitCode = await main();
