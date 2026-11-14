# free-model-aggregator-01

One server, two APIs (OpenAI-style and Anthropic-style), ten simulated
free-tier providers, twenty requests per trial. For every request the server
picks the strongest free model that meets the request's requirements and
still has budget, speaks that provider's wire, and returns the provider's
answer in the wire the client used.

## Layout

- `main.ts` — the entrypoint: reads the brief, serves `POST
  /v1/chat/completions`, `POST /v1/messages` and `GET /v1/models`,
  registers with the env, polls `/work`, exits 0 when the client is done.
  `node main.ts --list` prints the catalog and exits 0.
- `lib/providers.ts` — the provider directory, the model table and budget
  ranges, the catalog document, and candidate ranking (best rating first,
  never a paid model).
- `lib/wire.ts` — the three wires: canonical parsing of a client request
  (system text, last user text, tool names, max_tokens, stops, image, JSON
  mode), rebuilding it in a provider's wire, parsing the provider's buffered
  answer, and re-emitting it to the client (JSON or SSE) with the provider's
  text whole, its usage counts verbatim, and its stop reason mapped.

## How a request is served

1. The client body is parsed into canonical form (the fields the provider
   echo checks).
2. Candidates are ranked by rating, filtered by the manifest's requirements
   (tools / vision / json / min_context) and by wire: an OpenAI JSON-mode
   request never goes to an anthropic-wire provider, whose wire cannot carry
   JSON mode.
3. Each provider has a soft cap from the kit's sealed budget range
   (calls_max). Providers that spent their cap, or refused twice in a row,
   are set aside; a second pass ignores caps and set-asides as a last resort.
4. The provider is always called buffered (`stream: false`). A refusal
   (402 / 429 / 5xx) is retried once on the same model — a provider's failure
   schedule hits single call numbers — then the request fails over to the
   next-best candidate.
5. The answer is re-emitted in the client's wire, streamed or not as asked,
   with the provider's usage counts and stop reason (mapped into the
   client's wire family). No wall clock and no randomness reach any routing
   decision, so a replay leaves the same export.

## Catalog

`--list` prints the same document the entry registers: one entry per
provider (`id`, `base_url` = the provider's own path such as `/p/groq`,
`key_env`, `wire`, `models` with `id` / `free` / `context_length`), built
from the brief's directory and ratings table plus the sealed table's one
paid model (listed `free: false`, never a candidate). `GET /v1/models`
lists exactly the free model ids in the catalog, in OpenAI shape for a
Bearer header and Anthropic shape for `x-api-key`.

## Grade locally

```sh
node <kit>/cli/makefun.mjs grade-local --kit <kit> --bot . --driver process --instances 3
```
