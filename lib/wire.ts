// The client wires, the one upstream wire, and the model field.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), with a JSON body or with server-sent events. The entry dials the grader's
// gateway, an OpenAI-compatible endpoint, so every prompt goes upstream as a Chat Completions request
// and the Anthropic path is translated both ways:
//
//   - the OpenAI path passes the client's body (with the dialled model's id, an output cap always
//     present, and usage requested on streams) to the gateway and relays the gateway's JSON or SSE
//     answer, with every `model` field set to the id this entry dialled;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the gateway's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the
//     usage translated. Tool calls become tool_use blocks and input_json_delta events.
//
// THE MODEL FIELD
//   The house records every gateway call and scores the model that really answered, so a response's
//   `model` must be the model this entry dialled. OpenRouter's own answer names a free model without
//   the gateway's `openrouter/` prefix; these functions put the dialled id back on every answer and
//   every streamed chunk. They never make a model up: the id came from the brief and went to the
//   gateway in the request's `model`.
//
// STREAMING IS BUFFERED
//   Upstream SSE is read to the end before anything is sent downstream. That keeps the entry honest
//   about what it serves: a stream whose text never carries the prompt's token is a failed attempt
//   and the prompt fails over, and a stream the provider cut short is completed (a finish chunk, a
//   usage chunk, [DONE] / message_stop) instead of relayed broken. Answers here are one word or one
//   JSON object, so buffering costs nothing against the client's 30-second cap.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body
// and the same gateway answer give the same bytes on a replay. The two model listings use fixed
// timestamps for the same reason: a `created` that moved between runs would be a nondeterministic
// document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The gateway's per-request output cap: a larger max_tokens is refused before it is forwarded. */
const MAX_OUTPUT_TOKENS = 8192;
/** The output cap sent when a client body names none: the gateway refuses a request with no cap. */
const DEFAULT_MAX_TOKENS = 1024;

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The answer text of a Chat Completions JSON body, as the env reads it: the first choice's message. */
export function textOfOpenAIJson(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return '';
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null) return '';
  return typeof message.content === 'string' ? message.content : textOf(message.content);
}

/** The answer text of a Messages JSON body, as the env reads it: its text blocks joined. */
export function textOfAnthropicJson(body: Json): string {
  if (!Array.isArray(body.content)) return '';
  return body.content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

/** Whether a Chat Completions JSON body holds a first choice with a message: the shape check's floor. */
export function openAIJsonUsable(body: Json): boolean {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  return choice !== null && isRecord(choice.message);
}

/**
 * A deterministic output-token estimate from answer text, used ONLY when the provider's own usage is
 * missing or reports zero output tokens: the wire check needs a whole output count of at least 1,
 * and a provider that omits usage must not sink an otherwise valid answer. Four characters a token,
 * at least one.
 */
export function estimateOutputTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** An output cap the gateway accepts: the body's own when present and in range, else the default. */
function cappedMaxTokens(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value) && value >= 1) return Math.min(Math.floor(value), MAX_OUTPUT_TOKENS);
  return DEFAULT_MAX_TOKENS;
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An OpenAI-shaped client body as the upstream Chat Completions body that names `model`: the model's
 * id replaced, an output cap always present and within the gateway's limit, usage requested on
 * streams, and no provider-side state (a `store: true` is refused before forwarding, so it is
 * dropped). Everything else — messages, system prompt, tools, temperature, response_format — passes
 * through unchanged.
 */
export function openAIUpstreamBody(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  delete out.store;
  out.max_tokens = cappedMaxTokens(body.max_tokens ?? body.max_completion_tokens);
  if (body.stream === true) out.stream_options = { include_usage: true };
  return out;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through. The output cap is always
 * present: the gateway refuses a request with none.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages, max_tokens: cappedMaxTokens(body.max_tokens) };
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled.
 * Usage comes from the provider's own counts when it reports them; when it reports none, or zero
 * output tokens, the output count is estimated from the answer text so the wire check's floor
 * (an output of at least 1) is met by a real answer.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  const inputTokens = numberOrZero(usage.prompt_tokens);
  const reportedOutput = numberOrZero(usage.completion_tokens);
  const outputTokens = reportedOutput >= 1 ? reportedOutput : estimateOutputTokens(text);
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: inputTokens, output_tokens: outputTokens },
  };
}

/**
 * A Chat Completions answer with `model` set to the id the entry dialled, and the shape floor filled
 * in: a finish_reason (default `stop`), the assistant role, and whole usage counts (the provider's
 * own when reported, else an estimate from the answer text for the output count). The one truthful
 * model claim, plus the fields the wire check reads.
 */
export function openAIJsonWithModel(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  const choices = Array.isArray(body.choices) ? body.choices : [];
  if (choices.length > 0 && isRecord(choices[0])) {
    const choice = { ...choices[0] };
    if (typeof choice.finish_reason !== 'string' || choice.finish_reason === '') choice.finish_reason = 'stop';
    if (isRecord(choice.message)) choice.message = { role: 'assistant', ...choice.message, role: 'assistant' };
    out.choices = [choice, ...choices.slice(1)];
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const inputTokens = numberOrZero(usage.prompt_tokens);
  const reportedOutput = numberOrZero(usage.completion_tokens);
  const outputTokens = reportedOutput >= 1 ? reportedOutput : estimateOutputTokens(textOfOpenAIJson(body));
  out.usage = {
    prompt_tokens: inputTokens,
    completion_tokens: outputTokens,
    total_tokens: inputTokens + outputTokens,
  };
  return out;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One parsed SSE frame: its event name (null when unnamed) and its joined data lines. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split buffered SSE text into frames on blank lines, as the SSE grammar does. Comment lines drop out. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const raw of normalized.split('\n\n')) {
    if (raw.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** The result of translating one buffered upstream SSE body for a downstream client. */
export interface StreamTranslation {
  /** The exact bytes to send downstream (SSE text, ending in a blank line). */
  text: string;
  /** The answer text the env will read from the downstream stream. */
  answerText: string;
  /** The stop reason the downstream stream carries (OpenAI finish_reason or Anthropic stop_reason). */
  stopReason: string | null;
}

/**
 * A buffered Chat Completions SSE body as the downstream OpenAI stream: every data payload's `model`
 * set to the id dialled, everything after the first [DONE] dropped, a finish chunk added when the
 * provider sent none, a usage chunk added when the provider sent no usage, and exactly one [DONE]
 * at the end. The env's order check needs one finish chunk and [DONE] last; the usage check needs a
 * whole usage with output >= 1, which the added chunk estimates from the streamed text.
 */
export function openAIStreamFromText(upstreamText: string, model: string): StreamTranslation {
  const frames = parseSseFrames(upstreamText);
  const out: string[] = [];
  let text = '';
  let stopReason: string | null = null;
  let sawFinish = false;
  let sawUsage = false;
  let lastId: string | null = null;
  let lastCreated: number | null = null;
  let sawDone = false;

  for (const frame of frames) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      sawDone = true;
      break;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (typeof chunk.id === 'string') lastId = chunk.id;
    if (typeof chunk.created === 'number') lastCreated = chunk.created;
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    // A chunk whose choices are malformed would fail the shape check downstream: drop it rather
    // than relay it. (A usage-only chunk carries choices: [] and passes through.)
    if (chunk.choices !== undefined && !Array.isArray(chunk.choices)) continue;
    if (choices.length > 0 && !isRecord(choices[0])) continue;
    if (isRecord(chunk.usage)) sawUsage = true;
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice !== null) {
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') text += delta.content;
      if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') {
        sawFinish = true;
        stopReason = choice.finish_reason;
      }
    }
    out.push(`data: ${JSON.stringify({ ...chunk, model })}\n\n`);
  }

  const chunkId = lastId ?? 'chatcmpl-aggregator';
  const created = lastCreated ?? FIXED_OPENAI_CREATED;
  if (!sawFinish) {
    stopReason = 'stop';
    out.push(
      `data: ${JSON.stringify({ id: chunkId, object: 'chat.completion.chunk', created, model, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\n\n`,
    );
  }
  if (!sawUsage) {
    out.push(
      `data: ${JSON.stringify({ id: chunkId, object: 'chat.completion.chunk', created, model, choices: [], usage: { prompt_tokens: 0, completion_tokens: estimateOutputTokens(text), total_tokens: estimateOutputTokens(text) } })}\n\n`,
    );
  }
  void sawDone;
  out.push('data: [DONE]\n\n');
  return { text: out.join(''), answerText: text, stopReason };
}

/**
 * A buffered Chat Completions SSE body as the Anthropic event sequence: message_start on the first
 * chunk, content_block_start/delta/stop per text block or tool call, then message_delta with the
 * stop reason and usage, then message_stop. `model` is the id the entry dialled. When the provider
 * reports no output usage, the output count is estimated from the streamed text; when the provider
 * sends no chunks at all, the sequence is still well-formed (an empty text block, end_turn).
 */
export function anthropicEventsFromText(upstreamText: string, model: string): StreamTranslation {
  const frames = parseSseFrames(upstreamText);
  const out: string[] = [];
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;
  let text = '';

  const startEvents = (): void => {
    if (started) return;
    started = true;
    out.push(
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    );
  };
  const closeEvents = (): void => {
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
  };
  const finishEvents = (stop: string): void => {
    if (finished) return;
    finished = true;
    if (nextIndex === 0) {
      // No content block ever started (the provider sent no chunks): an empty text block, so the
      // message is well-formed even though it carries nothing the prompt asked for.
      out.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
      nextIndex = 1;
    }
    closeEvents();
    const output = outputTokens >= 1 ? outputTokens : estimateOutputTokens(text);
    out.push(
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: output } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    );
  };

  for (const frame of frames) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      startEvents();
      finishEvents(pendingStop ?? 'end_turn');
      break;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    startEvents();
    if (typeof delta.content === 'string' && delta.content !== '') {
      text += delta.content;
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
      }
      out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        out.push(
          sseEvent('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: {
              type: 'tool_use',
              id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
              name: typeof fn.name === 'string' ? fn.name : '',
              input: {},
            },
          }),
        );
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
      }
    }
    // Hold the stop reason until [DONE]: OpenAI sends the usage after the finish_reason, and
    // Anthropic's message_delta carries the output tokens.
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
  }
  if (!finished) {
    startEvents();
    finishEvents(pendingStop ?? 'end_turn');
  }
  return { text: out.join(''), answerText: text, stopReason: pendingStop ?? 'end_turn' };
}
