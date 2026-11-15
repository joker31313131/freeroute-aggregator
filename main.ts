// free-model-aggregator-02: one server, two client wires, and a real free model behind every answer.
//
// The trial gives this server a brief (the free model list with the house's ratings, and the six
// prompts' manifest), an env to register with, and a gateway that dials OpenRouter's free models on
// the entrant's pushed key. The server:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves GET /v1/models (in the shape of the auth header it got: x-api-key means Anthropic,
//      Authorization means OpenAI), POST /v1/chat/completions and POST /v1/messages, JSON or SSE;
//   4. answers each prompt by dialling free models through the gateway's Chat Completions route,
//      best-rated first, failing over on refusals, and never with an answer of its own;
//   5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts.
//
// ROUTING (quality, 60%): candidates come from the brief alone — the free models meeting the
// prompt's requirements, rating_milli descending, ties in the brief's order. The highest-rated
// model that cannot do JSON never serves a JSON prompt, because the requirement filter runs before
// the ranking. No model id is named anywhere in this file: the ranking is a pure function of the
// brief the trial hands the server.
//
// FAILOVER (10%) AND RATE LIMITS: a refused dial (4xx/5xx, or the gateway unreachable) is recorded
// in the response's `x-makefun-upstream-failures` trail and the prompt moves to the next candidate.
// A 5xx earns the model one immediate retry — a transient server error, not a verdict on the model.
// A 429 earns none: the model is cooling down, and spending another call on it helps nobody. A
// refused model cools down for the next prompt (cooldowns count prompts, never wall time, so a
// replay makes the same choices). A 401 ends the prompt at once: the key, not the model, is refused,
// and no other model can answer. A 200 whose text never carries the prompt's token is a soft
// failure — the model answered, but unproven. The model earns one retry at twice the output cap
// before the prompt moves on: a reasoning model can spend a small cap before reaching content, and
// a second dial with room to answer is worth more than abandoning the best-ranked candidate. The
// retry carries no trail row, because no refusal happened, and the model is NOT cooled down for
// it, because a missed echo on one prompt is weak evidence against the next. A 200 that answers
// in a shape the env would reject — an illegal finish reason, an output usage below 1, a streamed
// chunk with no choices, a second finish chunk, or a tool call the request never declared — is a
// soft failure too, but a recorded one: the trail carries a row for it (the dial's real status,
// 200, and the reason), and the prompt moves to the next candidate at once, with no retry and no
// cooldown. The tool-name check runs only on prompts that require tools, because the env reads
// tool names only there.
//
// CAPS: at most 8 dials a prompt and 54 a trial (the gateway refuses past 60), counted, never timed.
// Every dial names max_tokens, never below 1024 and never above 8192. No wall clock and no
// randomness anywhere in a decision: every choice derives from the brief, the requests, and the
// gateway's answers, so a replay leaves the same
// export byte for byte. Completion requests run one at a time through a queue, in arrival order,
// so the manifest row, the cooldowns and the call count can never interleave.
//
// THE MODEL RULE: whatever this server reports in a response's `model` field is the model it
// actually dialled — the house scores the model its own gateway record shows answered, and a claim
// with no matching successful call earns nothing.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  dedupedFreeModels,
  isRecord,
  rankCandidates,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromText,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonUsable,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamFromText,
  openAIToAnthropic,
  openAIUpstreamBody,
  parseSseFrames,
  textOfAnthropicJson,
  textOfOpenAIJson,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most bytes of one gateway answer this server buffers (the env keeps at most 2 MiB of an answer). */
const MAX_UPSTREAM_BYTES = 2 * 1024 * 1024;
/** The most dials one prompt may cost, refusals and soft failures included. */
const MAX_ATTEMPTS_PER_PROMPT = 8;
/** The most dials one trial may cost; the gateway refuses past 60, so this leaves margin. */
const MAX_TRIAL_CALLS = 54;
/** A refused model is skipped while the prompt index is below its refusal's index plus this. */
const COOLDOWN_SPAN = 2;
/** The least output tokens any upstream dial asks for: a smaller cap can strand a reasoning model before its content starts. */
const MIN_UPSTREAM_MAX_TOKENS = 1024;
/** The most output tokens any upstream dial asks for: the gateway refuses a larger ask. */
const MAX_UPSTREAM_MAX_TOKENS = 8192;
/** How many times /register is attempted before the server gives up. */
const REGISTER_ATTEMPTS = 5;
/** The fixed wait between /register attempts, in milliseconds: timing only, never a decision. */
const REGISTER_RETRY_WAIT_MS = 200;
/** The most rows the upstream-failures trail may carry (RULES: at most 8). */
const MAX_TRAIL_ROWS = 8;
/** The most bytes the trail header may carry (the env reads no trail over 8 KiB). */
const MAX_TRAIL_BYTES = 8 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the next completion request is: the brief's prompts are in arrival order. */
let manifestIndex = 0;
/** Gateway dials made so far this trial, against MAX_TRIAL_CALLS. */
let trialCalls = 0;
/** Per model, the first prompt index at which it may be dialled again after a refusal. */
const cooldownUntil = new Map<string, number>();
/** The serial queue completion requests run through, in arrival order. */
let completionQueue: Promise<void> = Promise.resolve();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, each id once, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return dedupedFreeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** One row of the `x-makefun-upstream-failures` trail: a refused dial, in the order it refused. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** The trail header's value for `failures`: at most 8 rows, at most 8 KiB, or null when there were none. */
function trailHeaderValue(failures: readonly FailureRow[]): string | null {
  if (failures.length === 0) return null;
  const rows = failures.slice(0, MAX_TRAIL_ROWS);
  let text = JSON.stringify(rows);
  while (text.length > MAX_TRAIL_BYTES && rows.length > 1) {
    rows.pop();
    text = JSON.stringify(rows);
  }
  return text.length <= MAX_TRAIL_BYTES ? text : null;
}

/** The error code of a refused gateway answer: its error code or type, else `http_<status>`. */
function errorCodeOf(text: string, status: number): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      const code = value.error.code;
      if (typeof code === 'string' && code !== '') return code.slice(0, 64);
      const type = value.error.type;
      if (typeof type === 'string' && type !== '') return type.slice(0, 64);
    }
  } catch {
    // Not JSON: fall through to the status form.
  }
  return `http_${status}`;
}

/**
 * The unique tokens a request carries, from its message and system texts: the WORD-1234 shapes the
 * scorer looks for in the provider's answer body. An answer that carries none of them is unproven,
 * so the prompt fails over. No token is ever invented here — these come only from the live request.
 */
function tokensOf(body: Json): string[] {
  const texts: string[] = [];
  const collectContent = (content: unknown): void => {
    if (typeof content === 'string') {
      texts.push(content);
      return;
    }
    if (!Array.isArray(content)) return;
    for (const block of content) {
      if (isRecord(block) && typeof block.text === 'string') texts.push(block.text);
    }
  };
  collectContent(body.system);
  if (Array.isArray(body.messages)) {
    for (const message of body.messages) {
      if (isRecord(message)) collectContent(message.content);
    }
  }
  const found = new Set<string>();
  for (const text of texts) {
    for (const match of text.matchAll(/[A-Z]{3,12}-\d{4}/g)) found.add(match[0]);
  }
  return [...found];
}

/** Requirements read off a request body: tools declared, image blocks, or a JSON-mode response_format. */
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
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/**
 * The requirements of the prompt now arriving: the brief's row by arrival order, unioned with what
 * the body itself declares. The manifest is the declared source; the body is the backstop, so a
 * JSON-mode body is never served by a model that cannot do JSON even if a manifest row disagrees.
 */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  const fromBody = requirementsFromBody(body);
  const requires: Requirement = {};
  if (entry?.requires.tools === true || fromBody.tools === true) requires.tools = true;
  if (entry?.requires.vision === true || fromBody.vision === true) requires.vision = true;
  if (entry?.requires.json === true || fromBody.json === true) requires.json = true;
  if (entry?.requires.min_context !== undefined) requires.min_context = entry.requires.min_context;
  return requires;
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

/** One gateway answer body, capped: null when the answer runs past MAX_UPSTREAM_BYTES. */
async function readUpstreamText(res: Response): Promise<string | null> {
  if (res.body === null) return '';
  const reader = res.body.getReader();
  const chunks: Buffer[] = [];
  let size = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = Buffer.from(value);
      size += chunk.length;
      if (size > MAX_UPSTREAM_BYTES) return null;
      chunks.push(chunk);
    }
  } finally {
    reader.releaseLock();
  }
  return Buffer.concat(chunks).toString('utf8');
}

/** One dial's outcome: an HTTP answer with its body text, or a refusal the network itself raised. */
type DialOutcome =
  | { kind: 'answered'; status: number; text: string | null }
  | { kind: 'unreachable'; message: string };

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<DialOutcome> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  let res: Response;
  try {
    res = await fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
  } catch (err) {
    return { kind: 'unreachable', message: (err as Error).message };
  }
  return { kind: 'answered', status: res.status, text: await readUpstreamText(res) };
}

/** A translated gateway answer, ready to send downstream in the wire the client used. */
interface Translated {
  /** The downstream JSON body (non-streaming), or the SSE bytes (streaming). */
  jsonBody: Json | null;
  sseText: string | null;
  /** The answer text the env will read from the downstream response. */
  answerText: string;
}

/** The finish_reason values the env accepts on the OpenAI wire (its legal stop reasons). */
const LEGAL_FINISH_REASONS = new Set(['stop', 'length', 'tool_calls', 'function_call', 'content_filter']);

/** The tool names a request declares, from its upstream Chat Completions body. */
function declaredToolNames(upstreamBody: Json): Set<string> {
  const names = new Set<string>();
  if (Array.isArray(upstreamBody.tools)) {
    for (const tool of upstreamBody.tools) {
      if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string' && tool.function.name !== '') {
        names.add(tool.function.name);
      }
    }
  }
  return names;
}

/**
 * A 200 JSON answer's shape defect, or null when the env would accept it: an illegal finish
 * reason, an output usage below 1, or — on prompts that require tools — a tool call the request
 * never declared. A missing finish reason is fine (the translators default it to `stop`), and a
 * missing usage is fine (the translators estimate the output count); only a stated value the env
 * rejects counts. Empty tool names are ignored, as the env ignores them.
 */
function invalidJsonAnswer(parsed: Json, declared: Set<string>, checkTools: boolean): string | null {
  const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice !== null) {
    const finish = choice.finish_reason;
    if (typeof finish === 'string' && finish !== '' && !LEGAL_FINISH_REASONS.has(finish)) return 'invalid_finish_reason';
    const message = isRecord(choice.message) ? choice.message : null;
    if (checkTools && message !== null && Array.isArray(message.tool_calls)) {
      for (const call of message.tool_calls) {
        if (
          isRecord(call) &&
          isRecord(call.function) &&
          typeof call.function.name === 'string' &&
          call.function.name !== '' &&
          !declared.has(call.function.name)
        ) {
          return 'undeclared_tool';
        }
      }
    }
  }
  const usage = isRecord(parsed.usage) ? parsed.usage : null;
  if (usage !== null && typeof usage.completion_tokens === 'number' && usage.completion_tokens < 1) return 'invalid_usage';
  return null;
}

/**
 * A 200 SSE answer's shape defect, or null when the env would accept it. The walk mirrors the
 * relay's drops — unparseable chunks, non-objects, a non-array choices, a non-object first choice,
 * and everything after the first [DONE] never reach the env, so they cannot condemn the answer —
 * and then checks what survives: every chunk carries a choices array, at most one chunk carries a
 * finish reason (none is fine: the relay synthesizes one), every finish reason is legal, the last
 * usage record reports an output of at least 1, and — on prompts that require tools — every named
 * tool call was declared by the request.
 */
function invalidStreamAnswer(text: string, declared: Set<string>, checkTools: boolean): string | null {
  const frames = parseSseFrames(text);
  let finishChunks = 0;
  let lastUsage: Json | null = null;
  for (const frame of frames) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') break;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (chunk.choices !== undefined && !Array.isArray(chunk.choices)) continue;
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length > 0 && !isRecord(choices[0])) continue;
    if (!('choices' in chunk)) return 'invalid_sse_chunk';
    if (isRecord(chunk.usage)) lastUsage = chunk.usage;
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const finish = choice.finish_reason;
    if (typeof finish === 'string') {
      if (!LEGAL_FINISH_REASONS.has(finish)) return 'invalid_finish_reason';
      finishChunks += 1;
      if (finishChunks > 1) return 'multiple_finish_chunks';
    }
    if (checkTools) {
      const delta = isRecord(choice.delta) ? choice.delta : null;
      if (delta !== null && Array.isArray(delta.tool_calls)) {
        for (const call of delta.tool_calls) {
          if (
            isRecord(call) &&
            isRecord(call.function) &&
            typeof call.function.name === 'string' &&
            call.function.name !== '' &&
            !declared.has(call.function.name)
          ) {
            return 'undeclared_tool';
          }
        }
      }
    }
  }
  if (lastUsage !== null) {
    const output = lastUsage.completion_tokens;
    if (typeof output !== 'number' || output < 1) return 'invalid_usage';
  }
  return null;
}

/**
 * A 200 gateway answer's shape defect, or null when the env would accept it. The defect code is
 * the trail row's code when the prompt fails over on it. Unparseable JSON returns null: the
 * translator has already rejected the body by then.
 */
function validateUpstream(stream: boolean, text: string, upstreamBody: Json, checkTools: boolean): string | null {
  const declared = checkTools ? declaredToolNames(upstreamBody) : new Set<string>();
  if (stream) return invalidStreamAnswer(text, declared, checkTools);
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(parsed)) return null;
  return invalidJsonAnswer(parsed, declared, checkTools);
}

/**
 * One 200 gateway answer translated into the client's wire, or null when the answer is unusable
 * (not JSON on a JSON prompt, or no first choice with a message). Streaming answers always
 * translate: the translators complete a cut-short stream instead of relaying it broken.
 */
function translate(
  wire: 'openai' | 'anthropic',
  stream: boolean,
  text: string,
  modelId: string,
): Translated | null {
  if (stream) {
    if (wire === 'openai') {
      const t = openAIStreamFromText(text, modelId);
      return { jsonBody: null, sseText: t.text, answerText: t.answerText };
    }
    const t = anthropicEventsFromText(text, modelId);
    return { jsonBody: null, sseText: t.text, answerText: t.answerText };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(parsed) || !openAIJsonUsable(parsed)) return null;
  if (wire === 'openai') {
    const body = openAIJsonWithModel(parsed, modelId);
    return { jsonBody: body, sseText: null, answerText: textOfOpenAIJson(body) };
  }
  const body = openAIToAnthropic(parsed, modelId);
  return { jsonBody: body, sseText: null, answerText: textOfAnthropicJson(body) };
}

/** The outcome of serving one prompt: served, given up after refusals, or no model eligible. */
type PromptOutcome =
  | { kind: 'served'; model: ModelRow; translated: Translated; failures: FailureRow[] }
  | { kind: 'given-up'; failures: FailureRow[]; lastStatus: number }
  | { kind: 'no-model' };

/**
 * Serve one prompt: walk the ranked candidates, dialling each at most once (plus one immediate
 * retry on a 5xx at the same cap, and one retry at twice the cap on a 200 without the token),
 * skipping models in cooldown and answers whose shape the env would refuse, until one answers
 * with a body carrying the prompt's token — or the attempt and trial caps run out. Deterministic
 * in the brief, the request, the prompt index, and the gateway's answers: no clock, no randomness.
 */
async function servePrompt(wire: 'openai' | 'anthropic', body: Json): Promise<PromptOutcome> {
  const promptIndex = manifestIndex;
  // Read this prompt's manifest row BEFORE advancing the index: requirementsFor reads the row at
  // manifestIndex, and advancing first would hand this prompt the NEXT row's requirements.
  const requires = requirementsFor(body);
  manifestIndex += 1;
  const candidates = rankCandidates(brief, requires);
  if (candidates.length === 0) {
    log(`prompt ${promptIndex}: no free model meets ${JSON.stringify(requires)}`);
    return { kind: 'no-model' };
  }
  const tokens = tokensOf(body);
  const stream = body.stream === true;
  const failures: FailureRow[] = [];
  const tried = new Set<string>();
  const retried5xx = new Set<string>();
  const retriedSoftFail = new Set<string>();
  let nextCandidate = 0;
  /** A same-model retry waiting to run, with the max_tokens its dial must ask for. */
  let pendingRetry: { model: ModelRow; cap: number } | null = null;
  let attempts = 0;
  let lastStatus = 502;

  while (attempts < MAX_ATTEMPTS_PER_PROMPT && trialCalls < MAX_TRIAL_CALLS) {
    let model: ModelRow | null = null;
    let retryCap: number | null = null;
    if (pendingRetry !== null) {
      model = pendingRetry.model;
      retryCap = pendingRetry.cap;
      pendingRetry = null;
    } else {
      while (nextCandidate < candidates.length) {
        const candidate = candidates[nextCandidate];
        nextCandidate += 1;
        if (tried.has(candidate.id)) continue;
        const until = cooldownUntil.get(candidate.id);
        if (until !== undefined && promptIndex < until) continue;
        model = candidate;
        break;
      }
      if (model === null) break;
    }
    tried.add(model.id);
    attempts += 1;
    trialCalls += 1;
    const upstreamBody = wire === 'openai' ? openAIUpstreamBody(body, model.id) : anthropicToOpenAI(body, model.id);
    // Every dial asks for at least the floor: a smaller ask can strand a reasoning model before its
    // content starts. A waiting retry names its own cap instead (the same cap after a 5xx, twice
    // the cap after a 200 without the token), already clamped when it was queued.
    const asked = typeof upstreamBody.max_tokens === 'number' ? upstreamBody.max_tokens : MIN_UPSTREAM_MAX_TOKENS;
    const dialCap = retryCap ?? Math.max(asked, MIN_UPSTREAM_MAX_TOKENS);
    upstreamBody.max_tokens = dialCap;
    const outcome = await dial(model.id, upstreamBody);

    if (outcome.kind === 'unreachable') {
      log(`prompt ${promptIndex}: ${model.id} unreachable (${outcome.message.slice(0, 120)})`);
      failures.push({ status: 502, code: 'upstream_unreachable', model: model.id });
      cooldownUntil.set(model.id, promptIndex + COOLDOWN_SPAN);
      lastStatus = 502;
      if (!retried5xx.has(model.id)) {
        retried5xx.add(model.id);
        pendingRetry = { model, cap: dialCap };
      }
      continue;
    }
    if (outcome.text === null) {
      // An answer past the cap: a soft failure, not a refusal — no trail row, no cooldown.
      log(`prompt ${promptIndex}: ${model.id} answered over the ${MAX_UPSTREAM_BYTES}-byte cap`);
      continue;
    }
    if (outcome.status < 200 || outcome.status >= 300) {
      const code = errorCodeOf(outcome.text, outcome.status);
      log(`prompt ${promptIndex}: ${model.id} refused HTTP ${outcome.status} (${code})`);
      failures.push({ status: outcome.status, code, model: model.id });
      cooldownUntil.set(model.id, promptIndex + COOLDOWN_SPAN);
      lastStatus = outcome.status;
      // A 401 refuses the key, not the model: no other model can answer this prompt.
      if (outcome.status === 401) break;
      if (outcome.status >= 500 && !retried5xx.has(model.id)) {
        retried5xx.add(model.id);
        pendingRetry = { model, cap: dialCap };
      }
      continue;
    }
    const translated = translate(wire, stream, outcome.text, model.id);
    if (translated === null) {
      log(`prompt ${promptIndex}: ${model.id} answered an unusable body`);
      continue;
    }
    // A 200 in a shape the env would reject is a recorded soft failure: the trail carries a row
    // with the dial's real status, and the prompt moves to the next candidate at once — no retry
    // at twice the cap (that retry is only for a missing token), no cooldown.
    const invalid = validateUpstream(stream, outcome.text, upstreamBody, requires.tools === true);
    if (invalid !== null) {
      log(`prompt ${promptIndex}: ${model.id} answered with ${invalid}`);
      failures.push({ status: 200, code: invalid, model: model.id });
      continue;
    }
    if (tokens.length > 0 && !tokens.some((token) => translated.answerText.includes(token))) {
      // The model answered, but nothing proves it: no trail row (no refusal), no cooldown (a missed
      // echo on this prompt is weak evidence against the next). One retry at twice the cap first:
      // a reasoning model may have spent the whole smaller ask before its content started. No retry
      // when doubling changes nothing (the cap is already at the gateway's ceiling).
      log(`prompt ${promptIndex}: ${model.id} answered without the prompt's token`);
      const escalated = Math.min(dialCap * 2, MAX_UPSTREAM_MAX_TOKENS);
      if (!retriedSoftFail.has(model.id) && escalated > dialCap) {
        retriedSoftFail.add(model.id);
        pendingRetry = { model, cap: escalated };
      }
      continue;
    }
    log(`prompt ${promptIndex}: served by ${model.id} after ${attempts} attempt(s), ${failures.length} refusal(s)`);
    return { kind: 'served', model, translated, failures };
  }
  log(`prompt ${promptIndex}: gave up after ${attempts} attempt(s), ${failures.length} refusal(s)`);
  return { kind: 'given-up', failures, lastStatus };
}

function sendJson(res: ServerResponse, status: number, body: unknown, trail: string | null): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(status, headers);
  res.end(text);
}

function sendSse(res: ServerResponse, text: string, trail: string | null): void {
  const headers: Record<string, string> = {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': String(Buffer.byteLength(text)),
  };
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(200, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, trail: string | null = null): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), trail);
  else sendJson(res, status, openAIError(message, type, null), trail);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list), null);
  else sendJson(res, 200, openAIModelList(list), null);
}

/** Answer one completion request in the client's wire, from the outcome servePrompt reached. */
function answerOutcome(res: ServerResponse, wire: 'openai' | 'anthropic', outcome: PromptOutcome): void {
  if (outcome.kind === 'no-model') {
    if (wire === 'anthropic') sendError(res, wire, 503, 'overloaded_error', 'no free model in the brief meets this request');
    else sendError(res, wire, 503, 'no_free_model', 'no free model in the brief meets this request');
    return;
  }
  const trail = trailHeaderValue(outcome.failures);
  if (outcome.kind === 'given-up') {
    const message = `the gateway refused every free model tried (last HTTP ${outcome.lastStatus})`;
    if (wire === 'anthropic') sendError(res, wire, 502, 'api_error', message, trail);
    else sendError(res, wire, 502, 'upstream_error', message, trail);
    return;
  }
  if (outcome.translated.sseText !== null) {
    sendSse(res, outcome.translated.sseText, trail);
    return;
  }
  sendJson(res, 200, outcome.translated.jsonBody ?? {}, trail);
}

/** POST /v1/chat/completions and POST /v1/messages, through the serial completion queue. */
async function completion(req: IncomingMessage, res: ServerResponse, wire: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const task = completionQueue.then(async () => {
    const outcome = await servePrompt(wire, body);
    answerOutcome(res, wire, outcome);
  });
  // The queue never rejects: a failed task must not wedge the prompts behind it.
  completionQueue = task.catch(() => undefined);
  await task;
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await completion(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await completion(req, res, 'anthropic');
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

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * POST /register with the port and the one catalog; the env answers with the trial's prompt count.
 * Retried on a network error or a 5xx — the body is identical every time, so a retry changes
 * nothing the replay compares — and fatal on a 4xx, which says the body itself is wrong.
 */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: Error | null = null;
  for (let attempt = 1; attempt <= REGISTER_ATTEMPTS; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (res.ok) {
        const reply: unknown = await res.json();
        if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
        return;
      }
      const text = (await res.text()).slice(0, 500);
      lastError = new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${text}`);
      if (res.status < 500) throw lastError;
    } catch (err) {
      lastError = err as Error;
      if (/answered HTTP 4/.test(lastError.message)) throw lastError;
    }
    if (attempt < REGISTER_ATTEMPTS) await wait(REGISTER_RETRY_WAIT_MS);
  }
  throw lastError ?? new Error('POST /register failed');
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
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
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
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
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
