/**
 * NeMo-Gym LanguageModelV3 implementation.
 *
 * The opencode `processor.ts` agentic loop is unmodified — this is the only
 * piece that swaps. Internally we POST to NeMo Gym's `/v1/chat/completions`
 * non-streaming, capture token IDs (`prompt_token_ids` / `generation_token_ids`
 * / `generation_log_probs`) from the response, and emit a single-shot synthetic
 * stream so opencode's streaming handler is happy.
 *
 * Why non-streaming? RL training requires contiguous, exact token IDs across
 * turns. Streaming has them drip in across SSE chunks; non-streaming returns
 * them in the final response cleanly. The opencode loop doesn't notice — it
 * receives all stream parts at once.
 *
 * Trajectory dump: every doStream call writes
 * `<completionsDir>/<instanceId>/<turn>.json` BEFORE the stream finishes,
 * so a tool crash later cannot lose this turn's token IDs. The shape matches
 * openhands' `llm_completions/<id>/*.json` exactly so gym's
 * `get_openhands_trajectory_from_completions` reads it without changes.
 *
 * Replay: `cfg.replayTurns` drives the root session and `cfg.replayManifest`
 * supplies one queue per recorded child. A live child is bound by its
 * recorded parent plus the exact task call ID that created it, so parallel
 * siblings and nested agents cannot consume one another's turns. Tool-bearing
 * calls are answered from that session's scripted queue instead of a real
 * HTTP call — same synthesized-stream shape as a real response, so
 * opencode's *real* tool-execution path (streamText -> the AI-SDK `tool.execute` closures
 * built in session/prompt.ts's resolveTools()) runs each replayed tool call
 * for real against the live sandbox. This is required for correctness:
 * SWE-bench agents mutate a git workspace and the final patch comes from
 * `git diff`, so a replayed "edit"/"bash" call must actually happen on the
 * fresh container, not just be asserted via stale recorded output text.
 * Once the queue is exhausted, doStream/doGenerate fall through unchanged to
 * the real HTTP path and the agent continues live. Scripted turns are never
 * dumped to completionsDir: gym derives the replay/live boundary from the
 * FIRST dumped completion's cumulative `messages` length, which must be the
 * first live call (by then the replay prefix is already in session
 * history), not a scripted one.
 *
 * Subsequent user messages (anything after the trajectory's first user
 * message) are real request content, not an action to replay — there's
 * nothing to "re-execute" for a plain user turn. Everything the caller sent
 * has to be part of what the model sees, starting with the first live call
 * and on every call after that. Since opencode's own session storage isn't
 * writable from here (see bench/replay.ts's module docblock), these aren't
 * persisted anywhere — `_buildRequestParams` re-splices them into the
 * outgoing message list, at a fixed position relative to the replayed
 * assistant turns, on every live doStream/doGenerate call.
 */

import {
  type LanguageModelV3,
  type LanguageModelV3CallOptions,
  type LanguageModelV3StreamPart,
  type LanguageModelV3Content,
  type SharedV3ProviderMetadata,
  type SharedV3Warning,
} from "@ai-sdk/provider"
import { promises as fs } from "node:fs"
import path from "node:path"
import { convertToOpenAICompatibleChatMessages } from "../copilot/chat/convert-to-openai-compatible-chat-messages"
import { prepareTools } from "../copilot/chat/openai-compatible-prepare-tools"

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

interface ChatRequestMessage {
  role: "system" | "user" | "assistant" | "tool"
  content?: string | Array<unknown> | null
  tool_calls?: Array<{
    id: string
    type: "function"
    function: { name: string; arguments: string }
  }>
  tool_call_id?: string
  name?: string
  prompt_token_ids?: number[]
  generation_token_ids?: number[]
  generation_log_probs?: number[]
  [key: string]: unknown
}

interface ChatResponseChoice {
  index?: number
  finish_reason?: string | null
  message: {
    role: string
    content?: string | null
    reasoning_text?: string | null
    tool_calls?: Array<{
      id?: string
      type?: string
      function: { name: string; arguments: string }
    }>
    prompt_token_ids?: number[]
    generation_token_ids?: number[]
    generation_log_probs?: number[]
    [key: string]: unknown
  }
}

interface ChatResponseUsage {
  prompt_tokens?: number | null
  completion_tokens?: number | null
  total_tokens?: number | null
}

interface ChatResponse {
  id?: string
  model?: string
  created?: number
  choices: ChatResponseChoice[]
  usage?: ChatResponseUsage
}

function contextOverflowStreamError(message: string): string {
  return JSON.stringify({
    type: "error",
    error: {
      code: "context_length_exceeded",
      message: message.slice(0, 2_000) || "Input exceeds context window of this model",
    },
  })
}

function isGymContextOverflowCompletion(choice: ChatResponseChoice): boolean {
  // Gym's vLLM wrapper translates an upstream context-overflow HTTP 400 into
  // a successful empty completion. The stable signal it returns is exactly
  // this pair. `content == null` alone is not enough because valid tool-call
  // completions also normally carry null assistant content.
  return choice.finish_reason === "length" && choice.message?.content == null
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const TOKEN_ID_FIELDS = ["prompt_token_ids", "generation_token_ids", "generation_log_probs"] as const

/** A single scripted assistant turn to replay before falling through to live model calls. */
export interface NemoGymReplayTurn {
  content: string | null
  toolCalls?: Array<{ id: string; name: string; arguments: string }>
  /**
   * Subsequent user messages that occurred, in the original trajectory,
   * immediately before this turn was originally generated. Not replayed as
   * part of the scripted turn itself (a user message isn't an action) —
   * spliced into every live request's message list once replay reaches this
   * point. See the module docblock.
   */
  precedingUserTexts?: string[]
}

export interface NemoGymReplaySession {
  sessionId: string
  parentSessionId: string
  spawnCallId: string
  spawnIndex: number
  subagentType?: string
  messageCount: number
  replayTurns: NemoGymReplayTurn[]
  replayTrailingUserTexts?: string[]
}

export interface NemoGymReplayManifest {
  version: 1
  rootSessionId: string
  sessions: NemoGymReplaySession[]
}

export interface NemoGymLanguageModelConfig {
  /** Provider id used to namespace providerMetadata. Defaults to "nemo-gym". */
  provider: string
  /** Full base URL of the model server (e.g. `http://gym-host:18086`). */
  baseURL: string
  /** Optional gym head-server-style model server name; informational only. */
  modelServerName?: string
  /** Custom request headers (auth, etc). */
  headers?: () => Record<string, string | undefined>
  /** Per-call HTTP timeout in ms. */
  requestTimeoutMs?: number
  /** Number of HTTP retry attempts on transient errors. */
  retries?: number
  /**
   * Forced sampling params for RL training. NeMo-RL's vLLM worker asserts
   * that every request's temperature/top_p exactly match the training
   * generation config (on-policy requirement) — when set, these override
   * whatever the session/agent layer picked, for ALL sessions including
   * subagents.
   */
  temperature?: number
  topP?: number
  /**
   * Optional forced max_tokens. When unset (the default), requests carry NO
   * max_tokens and vLLM generates up to the remaining context — opencode's
   * session-level output cap is deliberately ignored.
   */
  maxTokens?: number
  /**
   * Where per-call llm_completions JSONs land. The bench harness builds this
   * path; it must match what gym's host-side glob expects. If unset, no
   * trajectory dump happens (useful for dev/test).
   */
  completionsDir?: string
  /** instance_id for the dump file naming + path. Required when completionsDir set. */
  instanceId?: string
  /** Optional sink that the bench harness uses to count turns globally. */
  turnCounter?: { next(): number }
  /** Optional callback fired after each successful chat completion. */
  onCompletion?: (info: {
    turn: number
    messages: ChatRequestMessage[]
    response: ChatResponse
    providerSpecificFields: Record<string, unknown>
    requestParams: Record<string, unknown>
  }) => void | Promise<void>
  /**
   * Scripted assistant turns to replay in the root session before falling
   * through to live HTTP calls. See the module docblock for why replayed
   * tool calls must run for real rather than just replaying recorded text.
   */
  replayTurns?: NemoGymReplayTurn[]
  /**
   * Subsequent user messages after the last replayed assistant turn, with
   * nothing recorded after them in the trajectory (trajectory ends on a
   * user message). Spliced into every live request's message list, same as
   * NemoGymReplayTurn.precedingUserTexts. See the module docblock.
   */
  replayTrailingUserTexts?: string[]
  /** Causal parent-task-call -> recorded child-session replay graph. */
  replayManifest?: NemoGymReplayManifest
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

interface ReplayState {
  recordedSessionID: string
  recordedParentSessionID?: string
  spawnCallID?: string
  spawnIndex?: number
  subagentType?: string
  messageCount: number
  turns: NemoGymReplayTurn[]
  trailingUserTexts?: string[]
  index: number
  pendingUserInjections: Array<{ beforeAssistantOrdinal: number; text: string }>
  livePrefixMessageCount?: number
}

interface SessionHeaders {
  sessionID: string
  parentSessionID: string | undefined
  parentToolCallID: string | undefined
  agentName: string | undefined
}

export class NemoGymLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3"
  readonly modelId: string
  readonly provider: string

  private readonly cfg: NemoGymLanguageModelConfig
  private cookies: Record<string, string> = {}
  // Per-session turn counter. opencode's session header is `x-session-affinity`;
  // subagents spawned via the task tool get their own sessionID, so keeping
  // a Map keeps their dump filenames from clobbering the main session's.
  private readonly turnCounters: Map<string, number> = new Map()
  private readonly replayByRecordedSession = new Map<string, ReplayState>()
  private readonly replayChildBySpawn = new Map<string, string>()
  private readonly liveToRecordedSession = new Map<string, string>()
  private readonly recordedToLiveSession = new Map<string, string>()
  private readonly rootRecordedSessionID: string | undefined
  private globalTurn = 0
  private readonly sessionStartGlobalTurn = new Map<string, number>()

  constructor(modelId: string, cfg: NemoGymLanguageModelConfig) {
    this.modelId = modelId
    this.provider = cfg.provider
    this.cfg = {
      ...cfg,
      requestTimeoutMs: cfg.requestTimeoutMs ?? 600_000,
      retries: cfg.retries ?? 3,
    }
    this.rootRecordedSessionID = cfg.replayManifest?.rootSessionId ?? (cfg.replayTurns ? "__main__" : undefined)
    if (this.rootRecordedSessionID) {
      this.replayByRecordedSession.set(
        this.rootRecordedSessionID,
        this._makeReplayState({
          recordedSessionID: this.rootRecordedSessionID,
          messageCount: 0,
          turns: cfg.replayTurns ?? [],
          trailingUserTexts: cfg.replayTrailingUserTexts,
        }),
      )
    }
    for (const session of cfg.replayManifest?.sessions ?? []) {
      this.replayByRecordedSession.set(
        session.sessionId,
        this._makeReplayState({
          recordedSessionID: session.sessionId,
          recordedParentSessionID: session.parentSessionId,
          spawnCallID: session.spawnCallId,
          spawnIndex: session.spawnIndex,
          subagentType: session.subagentType,
          messageCount: session.messageCount,
          turns: session.replayTurns,
          trailingUserTexts: session.replayTrailingUserTexts,
        }),
      )
      this.replayChildBySpawn.set(this._spawnKey(session.parentSessionId, session.spawnCallId), session.sessionId)
    }
  }

  private _nextTurn(sessionID: string): number {
    const n = (this.turnCounters.get(sessionID) ?? -1) + 1
    this.turnCounters.set(sessionID, n)
    return n
  }

  private _nextGlobalTurn(sessionID: string): { globalTurn: number; sessionStartGlobalTurn: number } {
    const globalTurn = this.cfg.turnCounter?.next() ?? this.globalTurn++
    const sessionStartGlobalTurn = this.sessionStartGlobalTurn.get(sessionID) ?? globalTurn
    this.sessionStartGlobalTurn.set(sessionID, sessionStartGlobalTurn)
    return { globalTurn, sessionStartGlobalTurn }
  }

  private _makeReplayState(
    input: Omit<ReplayState, "index" | "pendingUserInjections" | "livePrefixMessageCount">,
  ): ReplayState {
    return {
      ...input,
      index: 0,
      pendingUserInjections: [
        ...input.turns.flatMap((turn, i) =>
          (turn.precedingUserTexts ?? []).map((text) => ({ beforeAssistantOrdinal: i, text })),
        ),
        ...(input.trailingUserTexts ?? []).map((text) => ({
          beforeAssistantOrdinal: input.turns.length,
          text,
        })),
      ],
    }
  }

  private _spawnKey(parentSessionID: string, callID: string): string {
    return `${parentSessionID}\u0000${callID}`
  }

  private _sessionFromHeaders(headers: unknown): SessionHeaders {
    let sid = ""
    let pid: string | undefined
    let callID: string | undefined
    let agentName: string | undefined
    if (headers && typeof headers === "object") {
      const h = headers as Record<string, unknown>
      const v = h["x-session-affinity"] ?? h["X-Session-Affinity"]
      if (typeof v === "string") sid = v
      const p = h["x-parent-session-id"] ?? h["X-Parent-Session-Id"]
      if (typeof p === "string") pid = p
      const c = h["x-parent-tool-call-id"] ?? h["X-Parent-Tool-Call-Id"]
      if (typeof c === "string") callID = c
      const a = h["x-opencode-agent"] ?? h["X-Opencode-Agent"]
      if (typeof a === "string") agentName = a
    }
    return { sessionID: sid || "main", parentSessionID: pid, parentToolCallID: callID, agentName }
  }

  private _bindReplaySession(liveSessionID: string, recordedSessionID: string): ReplayState | undefined {
    const existing = this.liveToRecordedSession.get(liveSessionID)
    if (existing && existing !== recordedSessionID) {
      throw new Error(
        `nemo-gym replay: live session ${liveSessionID} is already bound to ${existing}, cannot bind ${recordedSessionID}`,
      )
    }
    const otherLive = this.recordedToLiveSession.get(recordedSessionID)
    if (otherLive && otherLive !== liveSessionID) {
      throw new Error(
        `nemo-gym replay: recorded session ${recordedSessionID} is already bound to ${otherLive}, cannot bind ${liveSessionID}`,
      )
    }
    this.liveToRecordedSession.set(liveSessionID, recordedSessionID)
    this.recordedToLiveSession.set(recordedSessionID, liveSessionID)
    return this.replayByRecordedSession.get(recordedSessionID)
  }

  private _replayState(session: SessionHeaders): ReplayState | undefined {
    const recorded = this.liveToRecordedSession.get(session.sessionID)
    if (recorded) return this.replayByRecordedSession.get(recorded)
    if (!session.parentSessionID) {
      if (!this.rootRecordedSessionID) return undefined
      return this._bindReplaySession(session.sessionID, this.rootRecordedSessionID)
    }
    if (!session.parentToolCallID) return undefined
    const recordedParent = this.liveToRecordedSession.get(session.parentSessionID)
    if (!recordedParent) return undefined
    const recordedChild = this.replayChildBySpawn.get(this._spawnKey(recordedParent, session.parentToolCallID))
    if (!recordedChild) return undefined
    return this._bindReplaySession(session.sessionID, recordedChild)
  }

  // Auxiliary title/summary calls never pass tools, so they must not consume
  // a session's scripted agentic-loop turns.
  private _popReplayTurn(session: SessionHeaders, hasTools: boolean): NemoGymReplayTurn | undefined {
    if (!hasTools) return undefined
    const replay = this._replayState(session)
    if (!replay || replay.index >= replay.turns.length) return undefined
    return replay.turns[replay.index++]
  }

  private _rewriteTaskResumeArguments(tool: NonNullable<NemoGymReplayTurn["toolCalls"]>[number]): string {
    if (tool.name !== "task" || !tool.arguments.includes('"task_id"')) return tool.arguments
    let parsed: unknown
    try {
      parsed = JSON.parse(tool.arguments)
    } catch {
      return tool.arguments
    }
    if (!parsed || typeof parsed !== "object") return tool.arguments
    const input = parsed as Record<string, unknown>
    if (typeof input.task_id !== "string") return tool.arguments
    const liveSessionID = this.recordedToLiveSession.get(input.task_id)
    if (!liveSessionID) return tool.arguments
    return JSON.stringify({ ...input, task_id: liveSessionID })
  }

  private _messageFromReplayTurn(turn: NemoGymReplayTurn): ChatResponseChoice["message"] {
    return {
      role: "assistant",
      content: turn.content,
      tool_calls: turn.toolCalls?.map((tc) => ({
        id: tc.id,
        type: "function" as const,
        function: { name: tc.name, arguments: this._rewriteTaskResumeArguments(tc) },
      })),
    }
  }

  private _replayFinishReason(turn: NemoGymReplayTurn): string {
    return turn.toolCalls && turn.toolCalls.length > 0 ? "tool_calls" : "stop"
  }

  // Shared by the real HTTP path and the replay path so both stay in sync:
  // reasoning -> text -> tool-call stream parts for one assistant turn.
  private _enqueueMessageParts(
    controller: ReadableStreamDefaultController<LanguageModelV3StreamPart>,
    msg: Pick<ChatResponseChoice["message"], "content" | "reasoning_text" | "tool_calls">,
    providerMetadata: SharedV3ProviderMetadata,
  ) {
    // Reasoning content. providerMetadata on the *-end events is persisted by
    // opencode's processor as part.metadata and replayed on the next request
    // as part.providerOptions["nemo-gym"] — this is how per-turn token IDs
    // round-trip so EVERY assistant turn (not just the last) carries them for
    // RL training reconstruction. Replay turns carry no token IDs (empty
    // providerMetadata), which is fine — gym never reads a dump for them.
    if (msg.reasoning_text) {
      controller.enqueue({ type: "reasoning-start", id: "reasoning-0" })
      controller.enqueue({ type: "reasoning-delta", id: "reasoning-0", delta: msg.reasoning_text })
      controller.enqueue({ type: "reasoning-end", id: "reasoning-0", providerMetadata })
    }

    // Text content.
    if (msg.content) {
      controller.enqueue({ type: "text-start", id: "txt-0" })
      controller.enqueue({ type: "text-delta", id: "txt-0", delta: msg.content })
      controller.enqueue({ type: "text-end", id: "txt-0", providerMetadata })
    }

    // Tool calls. Replayed tool-call IDs are reused verbatim (see
    // _messageFromReplayTurn) so gym's replay/live boundary matching by
    // call_id still lines up.
    if (msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        const tcId = tc.id ?? `call_${Math.random().toString(36).slice(2, 10)}`
        controller.enqueue({
          type: "tool-input-start",
          id: tcId,
          toolName: tc.function.name,
        })
        controller.enqueue({
          type: "tool-input-delta",
          id: tcId,
          delta: tc.function.arguments,
        })
        controller.enqueue({ type: "tool-input-end", id: tcId })
        controller.enqueue({
          type: "tool-call",
          toolCallId: tcId,
          toolName: tc.function.name,
          input: tc.function.arguments,
        })
      }
    }
  }

  // Re-splices subsequent-user-message text from the replay trajectory into
  // the outgoing message list, in place. Processed in descending ordinal
  // order so inserting a later text doesn't shift the index about to be
  // looked up for an earlier one.
  private _injectPendingUserMessages(messages: ChatRequestMessage[], replay: ReplayState | undefined): void {
    if (!replay?.pendingUserInjections.length) return
    const byOrdinal = new Map<number, string[]>()
    for (const { beforeAssistantOrdinal, text } of replay.pendingUserInjections) {
      const list = byOrdinal.get(beforeAssistantOrdinal) ?? []
      list.push(text)
      byOrdinal.set(beforeAssistantOrdinal, list)
    }
    const ordinals = [...byOrdinal.keys()].sort((a, b) => b - a)
    for (const ordinal of ordinals) {
      const idx = this._findAssistantOrdinalIndex(messages, ordinal)
      const texts = byOrdinal.get(ordinal)!
      messages.splice(idx, 0, ...texts.map((text) => ({ role: "user" as const, content: text })))
    }
  }

  // Index right before the `ordinal`-th (0-based) assistant-role message, or
  // messages.length if fewer than `ordinal` assistant messages exist yet.
  private _findAssistantOrdinalIndex(messages: ChatRequestMessage[], ordinal: number): number {
    let seen = 0
    for (let i = 0; i < messages.length; i++) {
      if (messages[i].role === "assistant") {
        if (seen === ordinal) return i
        seen++
      }
    }
    return messages.length
  }

  get supportedUrls() {
    return {} as Record<string, RegExp[]>
  }

  // The streamText path in `session/llm.ts` only calls doStream. We still
  // implement doGenerate for completeness / future direct-use.
  async doGenerate(options: LanguageModelV3CallOptions) {
    const session = this._sessionFromHeaders(options.headers)
    const replayTurn = this._popReplayTurn(session, Boolean(options.tools && options.tools.length > 0))

    if (replayTurn) {
      const msg = this._messageFromReplayTurn(replayTurn)
      const providerMetadata = this._buildProviderMetadata({})
      const content: LanguageModelV3Content[] = []
      if (msg.content) content.push({ type: "text", text: msg.content, providerMetadata })
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          content.push({
            type: "tool-call",
            toolCallId: tc.id ?? `call_${Math.random().toString(36).slice(2, 10)}`,
            toolName: tc.function.name,
            input: tc.function.arguments,
          })
        }
      }
      return {
        content,
        finishReason: this._mapFinishReason(this._replayFinishReason(replayTurn)),
        usage: this._mapUsage(undefined),
        providerMetadata,
        request: { body: "{}" },
        response: { body: {} },
        warnings: [],
      }
    }

    const { warnings, loggedMessages, requestParams, globalTurn, sessionStartGlobalTurn } =
      await this._buildRequestParams(options, session)
    const { responseJson } = await this._postChat(requestParams)

    const choice = responseJson.choices[0]
    if (!choice) throw new Error("nemo-gym: empty choices in response")
    if (isGymContextOverflowCompletion(choice)) {
      throw new Error(
        contextOverflowStreamError("NeMo Gym returned an empty length completion for an overlong context"),
      )
    }
    const msg: ChatResponseChoice["message"] =
      choice.message ?? ({ role: "assistant" } as ChatResponseChoice["message"])

    const providerSpecificFields = this._extractProviderFields(msg)
    const providerMetadata = this._buildProviderMetadata(providerSpecificFields)

    const content: LanguageModelV3Content[] = []
    // Part-level providerMetadata round-trips per-turn token IDs (see doStream).
    if (msg.content) content.push({ type: "text", text: msg.content, providerMetadata })
    if (msg.reasoning_text) content.push({ type: "reasoning", text: msg.reasoning_text, providerMetadata })
    if (msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        content.push({
          type: "tool-call",
          toolCallId: tc.id ?? `call_${Math.random().toString(36).slice(2, 10)}`,
          toolName: tc.function.name,
          input: tc.function.arguments,
        })
      }
    }

    await this._dumpAndNotify({
      messages: loggedMessages,
      response: responseJson,
      providerSpecificFields,
      requestParams,
      session,
      globalTurn,
      sessionStartGlobalTurn,
    })

    return {
      content,
      finishReason: this._mapFinishReason(choice.finish_reason ?? null),
      usage: this._mapUsage(responseJson.usage),
      providerMetadata,
      request: { body: JSON.stringify(requestParams) },
      response: { body: responseJson },
      warnings,
    }
  }

  async doStream(options: LanguageModelV3CallOptions) {
    const session = this._sessionFromHeaders(options.headers)
    const replayTurn = this._popReplayTurn(session, Boolean(options.tools && options.tools.length > 0))

    if (replayTurn) {
      const msg = this._messageFromReplayTurn(replayTurn)
      const finishReasonRaw = this._replayFinishReason(replayTurn)
      const self = this
      const stream = new ReadableStream<LanguageModelV3StreamPart>({
        start(controller) {
          controller.enqueue({ type: "stream-start", warnings: [] })
          const providerMetadata = self._buildProviderMetadata({})
          self._enqueueMessageParts(controller, msg, providerMetadata)
          // No _dumpAndNotify here — see the module docblock: gym derives the
          // replay/live boundary from the FIRST dumped completion, which must
          // be the first live call.
          controller.enqueue({
            type: "finish",
            finishReason: self._mapFinishReason(finishReasonRaw),
            usage: self._mapUsage(undefined),
            providerMetadata,
          })
          controller.close()
        },
      })
      return { stream, request: { body: "{}" }, response: {} }
    }

    const { warnings, loggedMessages, requestParams, globalTurn, sessionStartGlobalTurn } =
      await this._buildRequestParams(options, session)

    // Fire the HTTP call eagerly so any error surfaces synchronously when the
    // stream is consumed. We then synthesize parts in `start`.
    const self = this

    const stream = new ReadableStream<LanguageModelV3StreamPart>({
      async start(controller) {
        controller.enqueue({ type: "stream-start", warnings })

        try {
          const { responseJson } = await self._postChat(requestParams)

          const choice = responseJson.choices[0]
          if (!choice) throw new Error("nemo-gym: empty choices in response")
          if (isGymContextOverflowCompletion(choice)) {
            throw new Error(
              contextOverflowStreamError("NeMo Gym returned an empty length completion for an overlong context"),
            )
          }
          const msg: ChatResponseChoice["message"] =
            choice.message ?? ({ role: "assistant" } as ChatResponseChoice["message"])

          const providerSpecificFields = self._extractProviderFields(msg)
          const providerMetadata = self._buildProviderMetadata(providerSpecificFields)

          // Emit response-metadata first.
          controller.enqueue({
            type: "response-metadata",
            id: responseJson.id,
            modelId: responseJson.model,
            timestamp: responseJson.created ? new Date(responseJson.created * 1000) : undefined,
          })

          self._enqueueMessageParts(controller, msg, providerMetadata)

          // Persist trajectory BEFORE finishing so a downstream tool crash
          // cannot lose this turn's token IDs.
          await self._dumpAndNotify({
            messages: loggedMessages,
            response: responseJson,
            providerSpecificFields,
            requestParams,
            session,
            globalTurn,
            sessionStartGlobalTurn,
          })

          controller.enqueue({
            type: "finish",
            finishReason: self._mapFinishReason(choice.finish_reason ?? null),
            usage: self._mapUsage(responseJson.usage),
            providerMetadata,
          })

          controller.close()
        } catch (err) {
          controller.enqueue({ type: "error", error: err instanceof Error ? err.message : String(err) })
          controller.enqueue({
            type: "finish",
            finishReason: { unified: "error", raw: undefined },
            usage: self._mapUsage(undefined),
            providerMetadata: {},
          })
          controller.close()
        }
      },
    })

    return {
      stream,
      request: { body: JSON.stringify(requestParams) },
      response: {},
    }
  }

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------

  private async _buildRequestParams(
    options: LanguageModelV3CallOptions,
    session: SessionHeaders,
  ): Promise<{
    warnings: SharedV3Warning[]
    messages: ChatRequestMessage[]
    loggedMessages: ChatRequestMessage[]
    tools: unknown
    toolChoice: unknown
    requestParams: Record<string, unknown>
    globalTurn: number
    sessionStartGlobalTurn: number
  }> {
    const warnings: SharedV3Warning[] = []
    // Reuse opencode's existing OpenAI-compatible message converter so all
    // tool-call / multi-content shapes map identically to the rest of opencode.
    const messages = convertToOpenAICompatibleChatMessages(options.prompt) as unknown as ChatRequestMessage[]

    // The gym's vllm proxy (NeMoGymEasyInputMessage) only accepts plain-string
    // content on non-assistant messages; the converter emits an ARRAY of parts
    // for multi-part user turns (e.g. the synthetic "Attached image(s) from
    // tool result:" message) which fails validation server-side with a 500.
    // The policy model is text-only anyway, so flatten arrays to a single
    // string and stub out non-text parts.
    for (const m of messages as Array<Record<string, unknown>>) {
      const content = m["content"]
      if (Array.isArray(content)) {
        m["content"] = (content as Array<Record<string, unknown>>)
          .map((p) =>
            p?.["type"] === "text" ? String(p["text"] ?? "") : `[${String(p?.["type"] ?? "unknown")} part omitted]`,
          )
          .join("\n")
      }
    }

    // Trajectory resume: splice in any subsequent user messages from the
    // replay trajectory (see bench/replay.ts + the module docblock for why
    // these can't just be persisted into opencode's own session storage).
    // Done here, before loggedMessages is cloned from messages below, so the
    // wire request and the trajectory dump gym reads both include them
    // identically. Re-applied on every call — nothing else carries them
    // forward — at a position fixed relative to the replayed assistant
    // turns, so they land in the same chronological spot every time.
    const replay = this._replayState(session)
    this._injectPendingUserMessages(messages as ChatRequestMessage[], replay)

    // Token-ID handling, mirroring OpenHands' nemo_gym_client.py exactly:
    //   - WIRE request: token IDs on the MOST RECENT assistant message only
    //     (the last turn's prompt_token_ids embed the exact token stream of
    //     the whole conversation; the vllm proxy uses it to verify continuity
    //     and avoid retokenization drift).
    //   - LOGGED trajectory (llm_completions/*.json): token IDs on EVERY
    //     assistant turn — swe_agents app.py materializes the training
    //     episode from the logged messages, and NeMo-RL needs per-turn
    //     generation_token_ids/log_probs to build the loss mask over all
    //     model-generated spans, not just the final turn.
    // The IDs round-trip through opencode's part metadata: doStream emits
    // providerMetadata on text-end / reasoning-end -> processor stores
    // part.metadata -> replay attaches part.providerOptions["nemo-gym"].
    // (The generic converter drops the fields, so we restore them here.)
    // Shallow-clone each message so wire-side attach/strip never leaks into
    // the logged copy and vice versa.
    const loggedMessages = (messages as Array<Record<string, unknown>>).map((m) => ({
      ...m,
    })) as unknown as ChatRequestMessage[]
    if (replay && options.tools?.length && replay.livePrefixMessageCount === undefined) {
      replay.livePrefixMessageCount = loggedMessages.length
    }
    {
      const promptAssistants = options.prompt.filter((m) => m.role === "assistant")
      const wireAssistants = (messages as Array<Record<string, unknown>>).filter((m) => m["role"] === "assistant")
      const loggedAssistants = (loggedMessages as unknown as Array<Record<string, unknown>>).filter(
        (m) => m["role"] === "assistant",
      )
      const n = Math.min(promptAssistants.length, wireAssistants.length)
      let mostRecentAttached = false
      for (let i = n - 1; i >= 0; i--) {
        const parts = promptAssistants[i].content
        if (!Array.isArray(parts)) continue
        for (const part of parts as Array<{ providerOptions?: Record<string, Record<string, unknown>> }>) {
          const md = part.providerOptions?.[this.provider]
          if (md && TOKEN_ID_FIELDS.every((f) => f in md)) {
            for (const f of TOKEN_ID_FIELDS) loggedAssistants[i][f] = md[f]
            if (!mostRecentAttached) {
              for (const f of TOKEN_ID_FIELDS) wireAssistants[i][f] = md[f]
              mostRecentAttached = true
            }
            break
          }
        }
      }
    }

    const { tools, toolChoice, toolWarnings } = prepareTools({
      tools: options.tools,
      toolChoice: options.toolChoice,
    })
    warnings.push(...toolWarnings)

    // Safeguard: strip token-ID fields from all assistant messages EXCEPT the
    // most recent (mirrors nemo_gym_client.py:85-97). With the single-message
    // attach above this is normally a no-op, but it keeps the wire contract
    // if an upstream change ever attaches more.
    {
      let lastSeen = false
      for (let i = messages.length - 1; i >= 0; i--) {
        const m = messages[i] as Record<string, unknown>
        const hasAll = TOKEN_ID_FIELDS.every((f) => f in m)
        if (lastSeen) {
          for (const f of TOKEN_ID_FIELDS) delete m[f]
        } else if (hasAll) {
          lastSeen = true
        }
      }
    }

    const requestParams: Record<string, unknown> = {
      messages,
      // max_tokens is intentionally OMITTED unless the gym config forces one:
      // without it vLLM generates up to the remaining context
      // (max_model_len - prompt), i.e. "unlimited" output. Sending opencode's
      // session-level cap (OUTPUT_TOKEN_MAX=32k) both truncated long turns and
      // shrank the usable input window (vLLM rejects input+max_tokens>context).
      ...(this.cfg.maxTokens ? { max_tokens: this.cfg.maxTokens } : {}),
      // Forced training params (cfg) win over session/agent-level choices:
      // NeMo-RL asserts exact temperature/top_p equality on every request.
      temperature: this.cfg.temperature ?? options.temperature,
      top_p: this.cfg.topP ?? options.topP,
      stop: options.stopSequences,
      seed: options.seed,
    }
    // Only include `model` when the caller-supplied modelId is a real value.
    // opencode's session resolver falls back to its sentinel `"default"`
    // (and to empty string with some misconfigured agents) when no model is
    // pinned. We DO NOT want either of those leaking through to OpenAI as a
    // literal `model: "default"` — the gym openai_model server's
    // `body_dict.setdefault("model", self.config.openai_model)` will fill in
    // the policy-configured model name when we omit it instead.
    if (this.modelId && this.modelId !== "default") {
      requestParams.model = this.modelId
    }
    if (tools && (tools as unknown[]).length) requestParams.tools = tools
    if (toolChoice) requestParams.tool_choice = toolChoice

    // Strip undefineds — vllm errors on null/undefined keys.
    for (const k of Object.keys(requestParams)) {
      if (requestParams[k] === undefined) delete requestParams[k]
    }

    return {
      warnings,
      messages,
      loggedMessages,
      tools,
      toolChoice,
      requestParams,
      ...this._nextGlobalTurn(session.sessionID),
    }
  }

  private async _postChat(params: Record<string, unknown>): Promise<{ responseJson: ChatResponse }> {
    const url = this._urlFor("/v1/chat/completions")
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "application/json",
    }
    // opencode's bundled-provider loader can pass `headers` as either a
    // function (matching upstream openai-compatible's schema) OR a plain
    // object (when opencode injects defaults from its provider merge layer).
    // Handle both — `?.()` would throw on a non-callable object.
    let cfgHeaders: Record<string, string | undefined> | undefined
    const rawHeaders = this.cfg.headers as unknown
    if (typeof rawHeaders === "function") {
      cfgHeaders = (rawHeaders as () => Record<string, string | undefined>)()
    } else if (rawHeaders && typeof rawHeaders === "object") {
      cfgHeaders = rawHeaders as Record<string, string | undefined>
    }
    if (cfgHeaders) {
      for (const [k, v] of Object.entries(cfgHeaders)) if (v != null) headers[k] = v
    }
    if (Object.keys(this.cookies).length) {
      headers.Cookie = Object.entries(this.cookies)
        .map(([k, v]) => `${k}=${v}`)
        .join("; ")
    }

    const retries = this.cfg.retries ?? 3
    const timeoutMs = this.cfg.requestTimeoutMs ?? 0
    let lastErr: unknown = null
    for (let attempt = 0; attempt < retries; attempt++) {
      const ac = new AbortController()
      // timeoutMs<=0 means "no timeout" — don't install the abort timer.
      const timer = timeoutMs > 0 ? setTimeout(() => ac.abort(), timeoutMs) : null
      try {
        const res = await fetch(url, {
          method: "POST",
          headers,
          body: JSON.stringify(params),
          signal: ac.signal,
        })
        if (timer) clearTimeout(timer)
        if (!res.ok) {
          const text = await res.text().catch(() => "")
          throw new Error(`NeMoGym ${url} ${res.status}: ${text.slice(0, 500)}`)
        }
        const setCookie = res.headers.get("set-cookie")
        if (setCookie) {
          for (const part of setCookie.split(/,(?=[^;]+=)/)) {
            const [kv] = part.split(";")
            const [k, v] = kv.split("=")
            if (k && v) this.cookies[k.trim()] = v.trim()
          }
        }
        const responseJson = (await res.json()) as ChatResponse
        return { responseJson }
      } catch (err) {
        if (timer) clearTimeout(timer)
        lastErr = err
        if (attempt === retries - 1) break
        // Cap exponential backoff at 60s so unlimited-retries configs don't blow up the delay.
        const backoffMs = Math.min(1000 * 2 ** attempt, 60_000)
        await new Promise((r) => setTimeout(r, backoffMs))
      }
    }
    throw new Error(`NeMoGym chat completions failed after ${retries} attempts: ${String(lastErr)}`)
  }

  private _urlFor(p: string): string {
    const base = this.cfg.baseURL.endsWith("/") ? this.cfg.baseURL : `${this.cfg.baseURL}/`
    return new URL(p.replace(/^\//, ""), base).toString()
  }

  private _extractProviderFields(msg: ChatResponseChoice["message"]): Record<string, unknown> {
    const out: Record<string, unknown> = {}
    if (Array.isArray(msg.prompt_token_ids)) {
      for (const f of TOKEN_ID_FIELDS) {
        const v = (msg as Record<string, unknown>)[f]
        if (v !== undefined) out[f] = v
      }
    }
    return out
  }

  private _buildProviderMetadata(providerSpecific: Record<string, unknown>): SharedV3ProviderMetadata {
    const md: SharedV3ProviderMetadata = { [this.provider]: {} }
    for (const [k, v] of Object.entries(providerSpecific)) {
      ;(md[this.provider] as Record<string, unknown>)[k] = v as never
    }
    return md
  }

  private _mapFinishReason(raw: string | null): {
    unified: "stop" | "length" | "tool-calls" | "error" | "other"
    raw: string | undefined
  } {
    if (!raw) return { unified: "other", raw: undefined }
    switch (raw) {
      case "stop":
        return { unified: "stop", raw }
      case "length":
        return { unified: "length", raw }
      case "tool_calls":
      case "function_call":
        return { unified: "tool-calls", raw }
      default:
        return { unified: "other", raw }
    }
  }

  private _mapUsage(raw?: ChatResponseUsage) {
    return {
      inputTokens: {
        total: raw?.prompt_tokens ?? undefined,
        noCache: raw?.prompt_tokens ?? undefined,
        cacheRead: undefined,
        cacheWrite: undefined,
      },
      outputTokens: {
        total: raw?.completion_tokens ?? undefined,
        text: raw?.completion_tokens ?? undefined,
        reasoning: undefined,
      },
    }
  }

  private async _dumpAndNotify(args: {
    messages: ChatRequestMessage[]
    response: ChatResponse
    providerSpecificFields: Record<string, unknown>
    requestParams: Record<string, unknown>
    session: SessionHeaders
    globalTurn: number
    sessionStartGlobalTurn: number
  }) {
    const turn = this._nextTurn(args.session.sessionID)
    const replay = this._replayState(args.session)
    const recordedParentSessionID =
      replay?.recordedParentSessionID ??
      (args.session.parentSessionID ? this.liveToRecordedSession.get(args.session.parentSessionID) : undefined)
    if (this.cfg.onCompletion) {
      try {
        await this.cfg.onCompletion({ turn, ...args })
      } catch (err) {
        console.warn(`[nemo-gym] onCompletion hook threw: ${String(err)}`)
      }
    }

    if (!this.cfg.completionsDir || !this.cfg.instanceId) return

    try {
      await fs.mkdir(this.cfg.completionsDir, { recursive: true })
      const turnStr = String(turn).padStart(4, "0")
      const safeModel = this.modelId.replace(/\//g, "__")
      // sessionID is part of the filename so subagent dumps don't clobber the
      // main session's. Sanitized for filesystem safety.
      const safeSession = args.session.sessionID.replace(/[^A-Za-z0-9_-]/g, "_")
      const fname = `${safeModel}-${safeSession}-${turnStr}-${Date.now()}.json`
      const fpath = path.join(this.cfg.completionsDir, fname)
      const kwargs: Record<string, unknown> = {}
      for (const [k, v] of Object.entries(args.requestParams)) {
        if (k !== "messages") kwargs[k] = v
      }
      const payload = {
        messages: args.messages,
        response: args.response,
        provider_specific_fields: args.providerSpecificFields,
        kwargs,
        session_id: args.session.sessionID,
        parent_session_id: args.session.parentSessionID ?? null,
        recorded_session_id: replay?.recordedSessionID ?? null,
        recorded_parent_session_id: recordedParentSessionID ?? null,
        spawn_call_id: replay?.spawnCallID ?? args.session.parentToolCallID ?? null,
        spawn_index: replay?.spawnIndex ?? null,
        subagent_type: replay?.subagentType ?? args.session.agentName ?? null,
        replay_prefix_message_count: replay?.livePrefixMessageCount ?? null,
        turn,
        global_turn: args.globalTurn,
        session_start_global_turn: args.sessionStartGlobalTurn,
        timestamp: Date.now() / 1000,
      }
      const tmp = `${fpath}.tmp`
      await fs.writeFile(tmp, JSON.stringify(payload))
      await fs.rename(tmp, fpath)
    } catch (err) {
      console.warn(`[nemo-gym] failed to dump completion: ${String(err)}`)
    }
  }
}
