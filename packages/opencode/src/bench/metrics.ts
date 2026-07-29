import { promises as fs } from "node:fs"

export interface ResponseLatencyMetric {
  model: string
  latency: number
  response_id: string
  timestamp: string
}

export interface ActionExecutionLatencyMetric {
  observation_type: string
  observation_id: string
  latency: number
  message: string
  timestamp: string
}

export interface TokenUsageMetric {
  model: string
  prompt_tokens: number
  completion_tokens: number
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
      prompt_tokens_details?: {
        cached_tokens?: unknown
      } | null
    }
  }
  latency?: unknown
  timestamp?: unknown
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined
}

function nonNegativeInteger(value: unknown): number {
  const number = finiteNumber(value)
  return number === undefined ? 0 : Math.max(0, Math.trunc(number))
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

  const start = finiteNumber(state.time?.start)
  const end = finiteNumber(state.time?.end)
  const callID = typeof part.callID === "string" ? part.callID : typeof part.id === "string" ? part.id : undefined
  if (start === undefined || end === undefined || end < start || !callID) return undefined

  const title = typeof state.title === "string" ? state.title : ""
  const error = typeof state.error === "string" ? state.error : ""
  return {
    observation_type: typeof part.tool === "string" ? part.tool : "opencode_tool",
    observation_id: callID,
    latency: (end - start) / 1000,
    message: title || error,
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
    const timestampSeconds = finiteNumber(dump.timestamp)
    const responseID = typeof dump.response?.id === "string" ? dump.response.id : ""
    if (latency === undefined || latency < 0 || timestampSeconds === undefined || !responseID) continue

    const model = typeof dump.response?.model === "string" ? dump.response.model : fallbackModel
    const promptTokens = nonNegativeInteger(dump.response?.usage?.prompt_tokens)
    const completionTokens = nonNegativeInteger(dump.response?.usage?.completion_tokens)
    const cacheReadTokens = nonNegativeInteger(dump.response?.usage?.prompt_tokens_details?.cached_tokens)

    records.push({
      startedAtSeconds: timestampSeconds - latency,
      timestampSeconds,
      responseLatency: {
        model,
        latency,
        response_id: responseID,
        timestamp: isoTimestamp(timestampSeconds),
      },
      tokenUsage: {
        model,
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
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
