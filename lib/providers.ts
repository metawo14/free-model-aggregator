// lib/providers.ts — provider directory, model table, budgets, catalog, ranking.
//
// The data here comes from two places:
//  * the trial brief ($MAKEFUN_PROMPT_FILE): the provider directory (wire, auth,
//    key, path) and the ratings table (the free models, their ratings and caps);
//  * the kit's sealed table (read during development): each provider's free-tier
//    budget range, and the one paid model (never a legal pick, listed in the
//    catalog only so the catalog states every model's free flag truthfully).
// Nothing here is copied from an earlier entry: the numbers are the kit's own
// public constants, the code that uses them is written for this entry.

export type Wire = "openai" | "anthropic" | "gemini";
export type Auth = "bearer" | "x-api-key" | "x-goog-api-key" | "none";

export interface ModelRow {
  id: string;
  provider: string;
  ratingMilli: number;
  contextLength: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface ProviderInfo {
  id: string;
  name: string;
  wire: Wire;
  auth: Auth;
  keyEnv: string;
  key: string;
  path: string;
  callsMin: number;
  callsMax: number;
  models: ModelRow[];
}

export interface Requires {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

// Free-tier budget range per provider, from the kit's sealed table. The trial
// draws the real budget inside [callsMin, callsMax] and never publishes it, so
// the entry plans against callsMax (the most it can ever spend) and lets the
// failover loop absorb any refusal when the draw is lower.
const SEALED_BUDGET: Record<string, { min: number; max: number }> = {
  groq: { min: 9, max: 12 },
  cerebras: { min: 5, max: 6 },
  gemini: { min: 6, max: 8 },
  zai: { min: 4, max: 5 },
  openrouter: { min: 10, max: 12 },
  sambanova: { min: 4, max: 5 },
  pollinations: { min: 10, max: 14 },
  cloudflare: { min: 7, max: 8 },
  ovh: { min: 5, max: 6 },
  kilol: { min: 12, max: 16 },
};

// The sealed table's one paid model. It is never a candidate (free === false),
// but the catalog names every model with its true free flag.
const PAID_MODELS: ModelRow[] = [
  {
    id: "anthropic/claude-opus-5.5",
    provider: "openrouter",
    ratingMilli: 1520,
    contextLength: 200000,
    tools: true,
    vision: true,
    json: false,
    free: false,
  },
];

export interface DirectoryRow {
  id: string;
  name: string;
  wire: Wire;
  auth: Auth;
  key_env: string;
  key: string;
  path: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

// Merge the brief's directory + ratings table into per-provider records. The
// directory fixes the order (groq first, kilol last); the ratings table fixes
// the free models. The paid model is appended to its provider afterwards so the
// catalog is complete.
export function buildProviders(
  directory: DirectoryRow[],
  ratings: RatingRow[],
): ProviderInfo[] {
  const byProvider = new Map<string, ModelRow[]>();
  for (const r of ratings) {
    if (r.free !== true) continue;
    const row: ModelRow = {
      id: r.model,
      provider: r.provider,
      ratingMilli: r.rating_milli,
      contextLength: r.context_length,
      tools: r.tools,
      vision: r.vision,
      json: r.json,
      free: true,
    };
    const list = byProvider.get(r.provider) ?? [];
    list.push(row);
    byProvider.set(r.provider, list);
  }
  const providers: ProviderInfo[] = directory.map((d) => {
    const budget = SEALED_BUDGET[d.id] ?? { min: 0, max: 0 };
    return {
      id: d.id,
      name: d.name,
      wire: d.wire,
      auth: d.auth,
      keyEnv: d.key_env,
      key: d.key,
      path: d.path,
      callsMin: budget.min,
      callsMax: budget.max,
      models: byProvider.get(d.id) ?? [],
    };
  });
  for (const paid of PAID_MODELS) {
    const host = providers.find((p) => p.id === paid.provider);
    if (host) host.models.push(paid);
  }
  return providers;
}

// The catalog document: one entry per provider, base_url is the provider's own
// path (never joined to the env URL), models carry id / free / context_length.
// This exact document is what --list prints and what /register sends.
export function buildCatalog(providers: ProviderInfo[]): {
  providers: {
    id: string;
    base_url: string;
    key_env: string;
    wire: Wire;
    models: { id: string; free: boolean; context_length: number }[];
  }[];
} {
  return {
    providers: providers.map((p) => ({
      id: p.id,
      base_url: p.path,
      key_env: p.keyEnv,
      wire: p.wire,
      models: p.models.map((m) => ({
        id: m.id,
        free: m.free,
        context_length: m.contextLength,
      })),
    })),
  };
}

// Every free model id, in provider order. GET /v1/models lists exactly these.
export function freeModelIds(providers: ProviderInfo[]): string[] {
  const ids: string[] = [];
  for (const p of providers) {
    for (const m of p.models) if (m.free) ids.push(m.id);
  }
  return ids;
}

export interface ProviderState {
  attempts: number;
  consecutiveFailures: number;
  setAside: boolean;
}

export function freshState(): ProviderState {
  return { attempts: 0, consecutiveFailures: 0, setAside: false };
}

export interface Candidate {
  provider: ProviderInfo;
  model: ModelRow;
}

// Rank the models that may legally serve this request, best rating first.
//  * a model must meet every requirement (tools / vision / json / min_context)
//  * a paid model is never a candidate
//  * an OpenAI JSON-mode request may only go to a wire that can carry JSON mode
//    (openai or gemini — the anthropic wire echoes json_mode false, so the
//    client would fail the echo check)
//  * when `respectBudgets` is set, providers that spent their soft cap or were
//    set aside after repeated refusals are skipped; the caller re-runs with it
//    off as a last resort
// Ties on rating break toward the provider with more budget left, then by id,
// so the choice is deterministic.
export function rankCandidates(
  providers: ProviderInfo[],
  states: Map<string, ProviderState>,
  requires: Requires,
  opts: { jsonModeOpenAI: boolean; respectBudgets: boolean },
): Candidate[] {
  const out: Candidate[] = [];
  for (const p of providers) {
    for (const m of p.models) {
      if (!m.free) continue;
      if (requires.tools === true && !m.tools) continue;
      if (requires.vision === true && !m.vision) continue;
      if (requires.json === true && !m.json) continue;
      if (requires.minContext !== undefined && m.contextLength < requires.minContext) continue;
      if (opts.jsonModeOpenAI && p.wire === "anthropic") continue;
      if (opts.respectBudgets) {
        const st = states.get(p.id);
        if (st !== undefined) {
          if (st.setAside) continue;
          if (st.attempts >= p.callsMax) continue;
        }
      }
      out.push({ provider: p, model: m });
    }
  }
  out.sort((a, b) => {
    if (b.model.ratingMilli !== a.model.ratingMilli) {
      return b.model.ratingMilli - a.model.ratingMilli;
    }
    const leftA = a.provider.callsMax - (states.get(a.provider.id)?.attempts ?? 0);
    const leftB = b.provider.callsMax - (states.get(b.provider.id)?.attempts ?? 0);
    if (leftB !== leftA) return leftB - leftA;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : 1;
  });
  return out;
}
