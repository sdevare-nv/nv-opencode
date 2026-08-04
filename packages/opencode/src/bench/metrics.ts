import { promises as fs } from "node:fs"

export interface ResponseLatencyMetric {
  model: string
  latency: number
  response_id: string
  request_kind: "agent" | "title" | "subagent"
  session_id: string
  parent_session_id: string | null
  session_turn: number
  start_timestamp: string
  timestamp: string
}

export interface ActionExecutionLatencyMetric {
  observation_type: string
  observation_id: string
  session_id: string
  child_session_id?: string
  input?: Record<string, unknown>
  output?: string
  latency: number
  message: string
  start_timestamp: string
  timestamp: string
}

export interface TokenUsageMetric {
  model: string
  prompt_tokens: number
  completion_tokens: number
  reasoning_tokens?: number
  cache_read_tokens: number
  cache_write_tokens: number
  context_window: number
  per_turn_token: number
  response_id: string
}

export interface CompletionMetrics {
  responseLatencies: ResponseLatencyMetric[]
  tokenUsages: TokenUsageMetric[]
}

interface CompletionDump {
  response?: {
    id?: unknown
    model?: unknown
    usage?: {
      prompt_tokens?: unknown
      completion_tokens?: unknown
      completion_tokens_details?: {
        reasoning_tokens?: unknown
      } | null
      prompt_tokens_details?: {
        cached_tokens?: unknown
      } | null
    }
  }
  latency?: unknown
  request_kind?: unknown
  request_started_at?: unknown
  session_id?: unknown
  parent_session_id?: unknown
  turn?: unknown
  timestamp?: unknown
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined
}

function nonNegativeInteger(value: unknown): number {
  const number = finiteNumber(value)
  return number === undefined ? 0 : Math.max(0, Math.trunc(number))
}

function optionalNonNegativeInteger(value: unknown): number | undefined {
  const number = finiteNumber(value)
  return number === undefined || number < 0 ? undefined : Math.trunc(number)
}

function isoTimestamp(seconds: number): string {
  return new Date(seconds * 1000).toISOString()
}

export function parseToolExecutionMetric(line: string): ActionExecutionLatencyMetric | undefined {
  let event: Record<string, any>
  try {
    event = JSON.parse(line)
  } catch {
    return undefined
  }

  const part = event.part
  const state = part?.state
  if (
    event.type !== "tool_use" ||
    part?.type !== "tool" ||
    (state?.status !== "completed" && state?.status !== "error")
  ) {
    return undefined
  }

  const recordedStart = finiteNumber(event.toolStart)
  const end = finiteNumber(state.time?.end)
  const callID = typeof part.callID === "string" ? part.callID : typeof part.id === "string" ? part.id : undefined
  if (recordedStart === undefined || end === undefined || end < recordedStart || !callID) return undefined

  const title = typeof state.title === "string" ? state.title : ""
  const error = typeof state.error === "string" ? state.error : ""
  const metadata =
    "metadata" in state && state.metadata && typeof state.metadata === "object"
      ? (state.metadata as Record<string, unknown>)
      : undefined
  const childSessionID =
    part.tool === "task" && typeof metadata?.sessionId === "string" ? metadata.sessionId : undefined
  const input =
    state.input && typeof state.input === "object" && !Array.isArray(state.input)
      ? (state.input as Record<string, unknown>)
      : undefined
  const output = typeof state.output === "string" ? state.output : undefined
  return {
    observation_type: typeof part.tool === "string" ? part.tool : "opencode_tool",
    observation_id: callID,
    session_id: typeof event.sessionID === "string" ? event.sessionID : "",
    ...(childSessionID ? { child_session_id: childSessionID } : {}),
    ...(input ? { input } : {}),
    ...(output === undefined ? {} : { output }),
    latency: (end - recordedStart) / 1000,
    message: title || error,
    start_timestamp: new Date(recordedStart).toISOString(),
    timestamp: new Date(end).toISOString(),
  }
}

export async function collectCompletionMetrics(
  completionsDir: string,
  fallbackModel: string,
): Promise<CompletionMetrics> {
  const records: Array<{
    startedAtSeconds: number
    timestampSeconds: number
    responseLatency: ResponseLatencyMetric
    tokenUsage: TokenUsageMetric
  }> = []

  for (const name of await fs.readdir(completionsDir)) {
    if (!name.endsWith(".json")) continue

    let dump: CompletionDump
    try {
      dump = JSON.parse(await fs.readFile(`${completionsDir}/${name}`, "utf8"))
    } catch {
      continue
    }

    const latency = finiteNumber(dump.latency)
    const requestStartedAtSeconds = finiteNumber(dump.request_started_at)
    const timestampSeconds = finiteNumber(dump.timestamp)
    const responseID = typeof dump.response?.id === "string" ? dump.response.id : ""
    if (
      latency === undefined ||
      latency < 0 ||
      requestStartedAtSeconds === undefined ||
      timestampSeconds === undefined ||
      timestampSeconds < requestStartedAtSeconds ||
      !responseID
    )
      continue

    const model = typeof dump.response?.model === "string" ? dump.response.model : fallbackModel
    const requestKind = dump.request_kind === "title" || dump.request_kind === "subagent" ? dump.request_kind : "agent"
    const sessionID = typeof dump.session_id === "string" ? dump.session_id : ""
    const parentSessionID = typeof dump.parent_session_id === "string" ? dump.parent_session_id : null
    const sessionTurn = nonNegativeInteger(dump.turn)
    const promptTokens = nonNegativeInteger(dump.response?.usage?.prompt_tokens)
    const completionTokens = nonNegativeInteger(dump.response?.usage?.completion_tokens)
    const reasoningTokens = optionalNonNegativeInteger(
      dump.response?.usage?.completion_tokens_details?.reasoning_tokens,
    )
    const cacheReadTokens = nonNegativeInteger(dump.response?.usage?.prompt_tokens_details?.cached_tokens)

    records.push({
      startedAtSeconds: requestStartedAtSeconds,
      timestampSeconds,
      responseLatency: {
        model,
        latency,
        response_id: responseID,
        request_kind: requestKind,
        session_id: sessionID,
        parent_session_id: parentSessionID,
        session_turn: sessionTurn,
        start_timestamp: isoTimestamp(requestStartedAtSeconds),
        timestamp: isoTimestamp(timestampSeconds),
      },
      tokenUsage: {
        model,
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        ...(reasoningTokens === undefined ? {} : { reasoning_tokens: reasoningTokens }),
        cache_read_tokens: cacheReadTokens,
        cache_write_tokens: 0,
        context_window: 0,
        per_turn_token: promptTokens + completionTokens,
        response_id: responseID,
      },
    })
  }

  // Subagent requests can overlap, so completion order is not turn-start order.
  records.sort(
    (a, b) =>
      a.startedAtSeconds - b.startedAtSeconds ||
      a.timestampSeconds - b.timestampSeconds ||
      a.responseLatency.response_id.localeCompare(b.responseLatency.response_id),
  )
  return {
    responseLatencies: records.map((record) => record.responseLatency),
    tokenUsages: records.map((record) => record.tokenUsage),
  }
}

export async function updateNemoGymMetrics(
  metricsPath: string | undefined,
  update: Record<string, unknown>,
): Promise<void> {
  if (!metricsPath) return

  let existing: Record<string, unknown> = {}
  try {
    existing = JSON.parse(await fs.readFile(metricsPath, "utf8"))
  } catch {}

  const tmpPath = `${metricsPath}.tmp.${process.pid}.${Date.now()}`
  await fs.writeFile(tmpPath, JSON.stringify({ ...existing, ...update }))
  await fs.rename(tmpPath, metricsPath)
}
