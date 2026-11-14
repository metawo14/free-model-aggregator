// lib/wire.ts — the three wires, and the translation between them.
//
// A client request (OpenAI or Anthropic wire) is parsed into one canonical
// form: the fields the env's provider echo checks — system text, the last user
// text, tool names, max_tokens, stop sequences, the image, JSON mode. That
// canonical form is then rebuilt in the chosen provider's wire, so the echo the
// provider returns matches the echo the client sent, field for field. The
// provider is always called buffered (stream:false) and its answer is re-emitted
// to the client in the client's wire, streamed or not as the client asked, with
// the provider's text whole, its usage counts verbatim, and its stop reason
// mapped into the client's wire family.
//
// The parsing below mirrors the env's own parsers (program/env/server.mjs):
// system text is every system message joined with "\n" (OpenAI) or the system
// field's text blocks (Anthropic); the user text is the LAST user message's
// text; the image is the last image block across the user messages; tools are
// the function/tool names in order.

export type ClientApi = "openai" | "anthropic";

export type ImageRef =
  | { kind: "base64"; mediaType: string; data: string }
  | { kind: "url"; url: string };

export interface ToolRef {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
}

export interface Canonical {
  api: ClientApi;
  stream: boolean;
  systemText: string;
  userText: string;
  image: ImageRef | null;
  tools: ToolRef[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function textOfBlocks(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") {
      parts.push(block.text);
    }
  }
  return parts.join("\n");
}

// Split a data: URL into media type + base64 payload, or return null.
function parseDataUrl(url: string): { mediaType: string; data: string } | null {
  const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mediaType: m[1], data: m[2] };
}

// Scan one message's content blocks for an image; the last image block wins.
function imageFromBlocks(content: unknown, api: ClientApi): ImageRef | null {
  if (!Array.isArray(content)) return null;
  let found: ImageRef | null = null;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (api === "openai" && block.type === "image_url" && isRecord(block.image_url)) {
      const url = block.image_url.url;
      if (typeof url !== "string") continue;
      const data = parseDataUrl(url);
      found = data !== null
        ? { kind: "base64", mediaType: data.mediaType, data: data.data }
        : { kind: "url", url };
    }
    if (api === "anthropic" && block.type === "image" && isRecord(block.source)) {
      const src = block.source;
      if (src.type === "base64" && typeof src.data === "string") {
        found = {
          kind: "base64",
          mediaType: typeof src.media_type === "string" ? src.media_type : "image/png",
          data: src.data,
        };
      } else if (src.type === "url" && typeof src.url === "string") {
        found = { kind: "url", url: src.url };
      }
    }
  }
  return found;
}

function imageToDataUrl(image: ImageRef): string {
  return image.kind === "base64"
    ? `data:${image.mediaType};base64,${image.data}`
    : image.url;
}

function numberOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === "string") return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === "string");
}

export function parseOpenAI(body: Record<string, unknown>): Canonical {
  const messages = Array.isArray(body.messages)
    ? body.messages.filter(isRecord)
    : [];
  const systemParts: string[] = [];
  let userText = "";
  let image: ImageRef | null = null;
  for (const message of messages) {
    if (message.role === "system") {
      const text = textOfBlocks(message.content);
      if (text !== "") systemParts.push(text);
    } else if (message.role === "user") {
      userText = textOfBlocks(message.content);
      const img = imageFromBlocks(message.content, "openai");
      if (img !== null) image = img;
    }
  }
  const tools: ToolRef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === "string") {
        tools.push({
          name: tool.function.name,
          description: typeof tool.function.description === "string"
            ? tool.function.description
            : undefined,
          inputSchema: isRecord(tool.function.parameters)
            ? tool.function.parameters
            : undefined,
        });
      }
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    api: "openai",
    stream: body.stream === true,
    systemText: systemParts.join("\n"),
    userText,
    image,
    tools,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: format === "json_object" || format === "json_schema",
  };
}

export function parseAnthropic(body: Record<string, unknown>): Canonical {
  const systemText = textOfBlocks(body.system);
  const messages = Array.isArray(body.messages)
    ? body.messages.filter(isRecord)
    : [];
  let userText = "";
  let image: ImageRef | null = null;
  for (const message of messages) {
    if (message.role !== "user") continue;
    userText = textOfBlocks(message.content);
    const img = imageFromBlocks(message.content, "anthropic");
    if (img !== null) image = img;
  }
  const tools: ToolRef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && typeof tool.name === "string") {
        tools.push({
          name: tool.name,
          description: typeof tool.description === "string" ? tool.description : undefined,
          inputSchema: isRecord(tool.input_schema) ? tool.input_schema : undefined,
        });
      }
    }
  }
  return {
    api: "anthropic",
    stream: body.stream === true,
    systemText,
    userText,
    image,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
  };
}

export function parseClient(api: ClientApi, body: Record<string, unknown>): Canonical {
  return api === "openai" ? parseOpenAI(body) : parseAnthropic(body);
}

// --- Upstream request builders (always buffered: stream false) -------------

function userContentOpenAI(c: Canonical): unknown {
  if (c.image === null) return c.userText;
  return [
    { type: "text", text: c.userText },
    { type: "image_url", image_url: { url: imageToDataUrl(c.image) } },
  ];
}

function userContentAnthropic(c: Canonical): unknown[] {
  const blocks: Record<string, unknown>[] = [{ type: "text", text: c.userText }];
  if (c.image !== null) {
    if (c.image.kind === "base64") {
      blocks.push({
        type: "image",
        source: { type: "base64", media_type: c.image.mediaType, data: c.image.data },
      });
    } else {
      blocks.push({ type: "image", source: { type: "url", url: c.image.url } });
    }
  }
  return blocks;
}

export function buildOpenAI(c: Canonical, model: string): Record<string, unknown> {
  const messages: Record<string, unknown>[] = [];
  if (c.systemText !== "") messages.push({ role: "system", content: c.systemText });
  messages.push({ role: "user", content: userContentOpenAI(c) });
  const body: Record<string, unknown> = {
    model,
    messages,
    stream: false,
  };
  if (c.tools.length > 0) {
    body.tools = c.tools.map((t) => ({
      type: "function",
      function: {
        name: t.name,
        description: t.description ?? "",
        parameters: t.inputSchema ?? { type: "object" },
      },
    }));
  }
  if (c.maxTokens !== null) body.max_tokens = c.maxTokens;
  if (c.stops.length > 0) body.stop = c.stops;
  if (c.jsonMode) body.response_format = { type: "json_object" };
  return body;
}

export function buildAnthropic(c: Canonical, model: string): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model,
    messages: [{ role: "user", content: userContentAnthropic(c) }],
    stream: false,
  };
  if (c.systemText !== "") body.system = c.systemText;
  if (c.tools.length > 0) {
    body.tools = c.tools.map((t) => ({
      name: t.name,
      description: t.description ?? "",
      input_schema: t.inputSchema ?? { type: "object" },
    }));
  }
  if (c.maxTokens !== null) body.max_tokens = c.maxTokens;
  if (c.stops.length > 0) body.stop_sequences = c.stops;
  return body;
}

export function buildGemini(c: Canonical, model: string): {
  path: string;
  body: Record<string, unknown>;
} {
  const parts: Record<string, unknown>[] = [{ text: c.userText }];
  if (c.image !== null) {
    if (c.image.kind === "base64") {
      parts.push({
        inlineData: { mimeType: c.image.mediaType, data: c.image.data },
      });
    } else {
      parts.push({ fileData: { fileUri: c.image.url } });
    }
  }
  const body: Record<string, unknown> = {
    contents: [{ role: "user", parts }],
  };
  if (c.systemText !== "") {
    body.systemInstruction = { parts: [{ text: c.systemText }] };
  }
  if (c.tools.length > 0) {
    body.tools = [{
      functionDeclarations: c.tools.map((t) => ({
        name: t.name,
        description: t.description ?? "",
        parameters: t.inputSchema ?? { type: "object" },
      })),
    }];
  }
  const config: Record<string, unknown> = {};
  if (c.maxTokens !== null) config.maxOutputTokens = c.maxTokens;
  if (c.stops.length > 0) config.stopSequences = c.stops;
  if (c.jsonMode) config.responseMimeType = "application/json";
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return {
    path: `/v1beta/models/${encodeURIComponent(model)}:generateContent`,
    body,
  };
}

// --- Upstream answer parsing ----------------------------------------------

export interface UpstreamAnswer {
  text: string;
  usage: { input: number; output: number };
  stop: string;
}

export interface UpstreamFailure {
  status: number;
  code: string;
  message: string;
}

// Parse one buffered upstream answer. A non-2xx status is a failure (the error
// envelope differs per wire, but the status alone is enough to fail over).
export function parseUpstream(
  wire: "openai" | "anthropic" | "gemini",
  status: number,
  text: string,
): { ok: true; answer: UpstreamAnswer } | { ok: false; failure: UpstreamFailure } {
  if (status < 200 || status >= 300) {
    let code = `http_${status}`;
    let message = text.slice(0, 200);
    try {
      const v = JSON.parse(text) as Record<string, unknown>;
      const err = isRecord(v.error) ? v.error : null;
      if (err !== null && typeof err.code === "string") code = err.code;
      if (err !== null && typeof err.message === "string") message = err.message;
      if (typeof v.type === "string" && v.type === "error" && err !== null) {
        if (typeof err.type === "string") code = err.type;
      }
    } catch {
      // keep the raw text as the message
    }
    return { ok: false, failure: { status, code, message } };
  }
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return {
      ok: false,
      failure: { status, code: "bad_json", message: "upstream answered non-JSON" },
    };
  }
  if (!isRecord(value)) {
    return {
      ok: false,
      failure: { status, code: "bad_json", message: "upstream answered a non-object" },
    };
  }
  if (wire === "openai") {
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    const message = choice !== null && isRecord(choice.message) ? choice.message : null;
    const usage = isRecord(value.usage) ? value.usage : null;
    if (message === null || usage === null) {
      return {
        ok: false,
        failure: { status, code: "bad_shape", message: "openai answer has no message/usage" },
      };
    }
    const content = message.content;
    const textOut = typeof content === "string" ? content : textOfBlocks(content);
    return {
      ok: true,
      answer: {
        text: textOut,
        usage: {
          input: numberOrNull(usage.prompt_tokens) ?? 0,
          output: numberOrNull(usage.completion_tokens) ?? 0,
        },
        stop: typeof choice.finish_reason === "string" ? choice.finish_reason : "stop",
      },
    };
  }
  if (wire === "anthropic") {
    const usage = isRecord(value.usage) ? value.usage : null;
    if (!Array.isArray(value.content) || usage === null) {
      return {
        ok: false,
        failure: { status, code: "bad_shape", message: "anthropic answer has no content/usage" },
      };
    }
    return {
      ok: true,
      answer: {
        text: textOfBlocks(value.content),
        usage: {
          input: numberOrNull(usage.input_tokens) ?? 0,
          output: numberOrNull(usage.output_tokens) ?? 0,
        },
        stop: typeof value.stop_reason === "string" ? value.stop_reason : "end_turn",
      },
    };
  }
  // gemini
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : null;
  const content = cand !== null && isRecord(cand.content) ? cand.content : null;
  const meta = isRecord(value.usageMetadata) ? value.usageMetadata : null;
  if (content === null || !Array.isArray(content.parts) || meta === null) {
    return {
      ok: false,
      failure: { status, code: "bad_shape", message: "gemini answer has no parts/usage" },
    };
  }
  const textOut = content.parts
    .filter(isRecord)
    .map((p) => (typeof p.text === "string" ? p.text : ""))
    .join("");
  return {
    ok: true,
    answer: {
      text: textOut,
      usage: {
        input: numberOrNull(meta.promptTokenCount) ?? 0,
        output: numberOrNull(meta.candidatesTokenCount) ?? 0,
      },
      stop: typeof cand.finishReason === "string" ? cand.finishReason : "STOP",
    },
  };
}

// --- Stop-reason mapping ----------------------------------------------------
// The provider always answers with a family-0 stop (stop / end_turn / STOP),
// but map generally: end_turn -> stop, max_tokens -> length, tool_use ->
// tool_calls, refusal -> content_filter (and the anthropic mirror of each).
export function mapStop(providerStop: string, clientApi: ClientApi): string {
  const openai: Record<string, string> = {
    stop: "stop",
    end_turn: "stop",
    STOP: "stop",
    length: "length",
    max_tokens: "length",
    MAX_TOKENS: "length",
    tool_calls: "tool_calls",
    function_call: "tool_calls",
    tool_use: "tool_calls",
    content_filter: "content_filter",
    refusal: "content_filter",
    SAFETY: "content_filter",
  };
  const anthropic: Record<string, string> = {
    stop: "end_turn",
    end_turn: "end_turn",
    STOP: "end_turn",
    length: "max_tokens",
    max_tokens: "max_tokens",
    MAX_TOKENS: "max_tokens",
    tool_calls: "tool_use",
    function_call: "tool_use",
    tool_use: "tool_use",
    content_filter: "refusal",
    refusal: "refusal",
    SAFETY: "refusal",
  };
  const table = clientApi === "openai" ? openai : anthropic;
  return table[providerStop] ?? (clientApi === "openai" ? "stop" : "end_turn");
}

// --- Client response emitters ----------------------------------------------
// The created timestamp is fixed (the env uses the same constant) so no wall
// clock ever reaches a response.
export const FIXED_CREATED = 1767225600;

function splitText(text: string, size: number): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += size) pieces.push(text.slice(at, at + size));
  return pieces.length === 0 ? [""] : pieces;
}

export function openAIJsonBody(
  model: string,
  text: string,
  usage: { input: number; output: number },
  stop: string,
): string {
  return JSON.stringify({
    id: "chatcmpl-sim",
    object: "chat.completion",
    created: FIXED_CREATED,
    model,
    choices: [{
      index: 0,
      message: { role: "assistant", content: text },
      finish_reason: stop,
      logprobs: null,
    }],
    usage: {
      prompt_tokens: usage.input,
      completion_tokens: usage.output,
      total_tokens: usage.input + usage.output,
    },
  });
}

export function openAIStreamBody(
  model: string,
  text: string,
  usage: { input: number; output: number },
  stop: string,
): string {
  const chunk = (delta: Record<string, unknown>, finish: string | null): string =>
    `data: ${JSON.stringify({
      id: "chatcmpl-sim",
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    })}\n\n`;
  const frames: string[] = [chunk({ role: "assistant", content: "" }, null)];
  for (const piece of splitText(text, 64)) {
    frames.push(chunk({ content: piece }, null));
  }
  frames.push(chunk({}, stop));
  // The usage chunk carries the provider's counts verbatim; the checker reads
  // usage off any chunk that carries it.
  frames.push(
    `data: ${JSON.stringify({
      id: "chatcmpl-sim",
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: usage.input,
        completion_tokens: usage.output,
        total_tokens: usage.input + usage.output,
      },
    })}\n\n`,
  );
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

export function anthropicJsonBody(
  model: string,
  text: string,
  usage: { input: number; output: number },
  stop: string,
): string {
  return JSON.stringify({
    id: "msg_sim",
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text }],
    stop_reason: stop,
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  });
}

export function anthropicStreamBody(
  model: string,
  text: string,
  usage: { input: number; output: number },
  stop: string,
): string {
  const frame = (event: string, data: unknown): string =>
    `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  const frames: string[] = [
    frame("message_start", {
      type: "message_start",
      message: {
        id: "msg_sim",
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: 0 },
      },
    }),
    frame("content_block_start", {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    }),
  ];
  for (const piece of splitText(text, 64)) {
    frames.push(frame("content_block_delta", {
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text: piece },
    }));
  }
  frames.push(frame("content_block_stop", { type: "content_block_stop", index: 0 }));
  frames.push(frame("message_delta", {
    type: "message_delta",
    delta: { stop_reason: stop, stop_sequence: null },
    usage: { output_tokens: usage.output },
  }));
  frames.push(frame("message_stop", { type: "message_stop" }));
  return frames.join("");
}
