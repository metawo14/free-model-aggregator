# free-model-aggregator-01

One server, two client wires, ten free providers. The entry for
make.fun's "The best free AI model aggregator" ($500, top 20).

## What it does

The client sends twenty requests (OpenAI Chat Completions and Anthropic
Messages, JSON and streaming) through this server. Each request needs a
free model that meets its requirements (tools, vision, JSON mode, long
context). The server reads the brief (provider directory, quality table,
workload manifest), probes every provider's catalog route, registers one
catalog, then for each request picks the best free model that can serve
it, calls exactly one provider, and answers in the client's wire.

## Design

- **Rating-ranked routing.** Candidates are all free (provider, model)
  pairs meeting the request's requirements, sorted by the quality
  table's `rating_milli` (best first). A paid model is never a pick.
  A JSON-mode OpenAI request never routes to an anthropic-wire provider
  (one can never echo `json_mode=true`).
- **Failover.** A provider that answers 402, 429 or a 5xx is retried on
  the next-best free model that meets the request. A provider that keeps
  refusing is set aside. Providers past their soft cap (the smallest
  budget the kit's sealed table draws for them) are only tried when
  nothing under its cap is left, so one provider's budget is never
  burned on the first requests.
- **Buffered upstream, re-emitted downstream.** The server always calls
  a provider with `stream:false` and a plain JSON body, then re-emits
  the answer in the client's wire (JSON, or SSE rebuilt from the
  buffered text). The provider's own usage numbers are copied into the
  client answer and the answer text passes through verbatim, so the echo,
  the stop reason and the usage the client checks are exactly the
  provider's own. No cross-wire streaming translation can garble them.
- **All three upstream wires.** OpenAI (`POST /v1/chat/completions`),
  Anthropic (`POST /v1/messages`, `x-api-key` +
  `anthropic-version: 2023-06-01`) and Gemini
  (`POST /v1beta/models/<model>:generateContent`, `x-goog-api-key`).
  Translations preserve the echo fields exactly: system text, last user
  text, tool names in order, `max_tokens`, stop sequences, image bytes,
  JSON mode.
- **Deterministic.** No clock, no unseeded randomness, fixed timestamps
  in every listing. Requests are served one at a time, in arrival order.
- **Real services.** The same code, given a real key and base URL in the
  directory, runs against the real service: the directory's `path` is
  joined to the env URL, or an absolute `base_url` is used as it stands.

## Layout

- `main.ts` — the server: brief, catalog probe, register, routes,
  rating-ranked routing with failover.
- `lib/providers.ts` — the brief's parser, the catalog probe, the
  catalog builder (from the starter kit).
- `lib/wire.ts` — client-wire translations and answer builders (starter
  kit plus the anthropic/gemini upstream requests and the re-emitted
  SSE shapes).

## Local results (practice instances, `grade-local --driver process`)

| trial | score | quality | served | wire | failover | catalog | upstream calls |
| ----- | ----- | ------- | ------ | ---- | -------- | ------- | -------------- |
| 0     | 99    | 99      | 100    | 100  | 100      | 100     | 25             |
| 1     | 99    | 99      | 100    | 100  | 100      | 100     | 27             |
| 2     | 99    | 99      | 100    | 100  | 100      | 100     | 29             |

Quality 99 is the practical ceiling: only one model rates 1475 and its
provider's budget (at most 12 calls) is smaller than the twenty requests
that want it, so quality 100 (every request at its best rating) is
unreachable on any draw. Every trial replayed byte for byte and the L1
regrade reproduced byte for byte.

## Run

```sh
node main.ts --list        # print the registered catalog, exit 0
node main.ts               # serve: needs MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL
```

Node 22, standard library only.
