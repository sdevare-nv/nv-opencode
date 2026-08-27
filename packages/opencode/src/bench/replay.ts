/**
 * Parses a replay-messages file (prior chat-completion-format trajectory) for
 * the SWE-bench bench harness. Split out of `bench/cli.ts` so it's importable
 * without triggering `cli.ts`'s top-level `main()` (which calls
 * `process.exit` on missing/invalid CLI args — not test-friendly).
 *
 * The first user message becomes the task instruction, assistant messages
 * become scripted turns replayed in order by the nemo-gym provider (see
 * language-model.ts). System / tool messages are skipped — system content is
 * handled separately (pinned as the agent's system prompt by gym), and tool
 * output is regenerated fresh by actually re-executing each replayed tool
 * call against the sandbox, not replayed from recorded text.
 *
 * A *subsequent* user message (anything after the first) is real request
 * content — everything the caller sent must be part of what the model sees.
 * Unlike assistant/tool content it isn't reconstructed by replaying it
 * through the agent loop (it's inert text, not an action), so each one is
 * attached to the replay turn it immediately precedes as `precedingUserTexts`
 * (or, if it trails the very last replayed turn with nothing recorded after
 * it, returned separately as `trailingUserTexts`). `NemoGymLanguageModel`
 * splices these into the outgoing message list on every live model call from
 * the first one onward — they're not persisted in opencode's own session
 * storage (there's no cheap way to do that without restructuring the bench
 * harness to drive multiple `session.prompt()` calls against one long-lived,
 * disk-backed session), so the provider re-applies them on every request
 * instead of relying on session history to carry them forward.
 */

import type { NemoGymReplayManifest, NemoGymReplayTurn } from "../provider/sdk/nemo-gym/language-model"

export interface ReplayChatMessage {
  role: "system" | "user" | "assistant" | "tool"
  content?: string | Array<{ type?: string; text?: string }> | null
  tool_calls?: Array<{ id: string; type?: string; function: { name: string; arguments: string } }>
}

export function replayMessageText(content: ReplayChatMessage["content"]): string {
  if (typeof content === "string") return content
  if (Array.isArray(content)) {
    return content
      .filter((part) => part?.type === "text" || typeof part?.text === "string")
      .map((part) => part.text ?? "")
      .join("\n")
  }
  return ""
}

export interface ParsedReplay {
  initialUserText: string
  replayTurns: NemoGymReplayTurn[]
  /** Subsequent user messages after the last replayed assistant turn (trajectory ends on a user message). */
  trailingUserTexts?: string[]
}

export function parseReplayMessages(raw: string): ParsedReplay {
  const messages = JSON.parse(raw) as ReplayChatMessage[]

  let initialUserText: string | undefined
  const replayTurns: NemoGymReplayTurn[] = []
  let pendingUserTexts: string[] = []

  for (const msg of messages) {
    if (msg.role === "system" || msg.role === "tool") continue
    if (msg.role === "user") {
      const text = replayMessageText(msg.content)
      if (initialUserText === undefined) {
        initialUserText = text
      } else if (text) {
        pendingUserTexts.push(text)
      }
      continue
    }
    if (msg.role === "assistant") {
      replayTurns.push({
        content: typeof msg.content === "string" ? msg.content : replayMessageText(msg.content) || null,
        toolCalls: msg.tool_calls?.map((tc) => ({ id: tc.id, name: tc.function.name, arguments: tc.function.arguments })),
        ...(pendingUserTexts.length ? { precedingUserTexts: pendingUserTexts } : {}),
      })
      pendingUserTexts = []
    }
  }

  if (initialUserText === undefined) {
    throw new Error("replay-messages-file: no user message found (expected at least one task-instruction message)")
  }

  return {
    initialUserText,
    replayTurns,
    ...(pendingUserTexts.length ? { trailingUserTexts: pendingUserTexts } : {}),
  }
}

function replayManifestError(message: string): never {
  throw new Error(`replay-subagents-file: ${message}`)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

/** Parse Gym's snake_case causal subagent manifest into provider replay queues. */
export function parseReplayManifest(raw: string): NemoGymReplayManifest {
  const input: unknown = JSON.parse(raw)
  if (!isRecord(input)) return replayManifestError("expected an object")
  if (input.version !== 1) return replayManifestError("version must be 1")
  if (typeof input.root_session_id !== "string" || !input.root_session_id) {
    return replayManifestError("root_session_id must be a non-empty string")
  }
  if (!Array.isArray(input.sessions)) return replayManifestError("sessions must be an array")

  const seen = new Set<string>()
  const seenSpawns = new Set<string>()
  const sessions = input.sessions.map((value, index) => {
    if (!isRecord(value)) return replayManifestError(`sessions[${index}] must be an object`)
    const sessionId = value.session_id
    const parentSessionId = value.parent_session_id
    const spawnCallId = value.spawn_call_id
    const spawnIndex = value.spawn_index
    if (typeof sessionId !== "string" || !sessionId) {
      return replayManifestError(`sessions[${index}].session_id must be a non-empty string`)
    }
    if (sessionId === input.root_session_id) {
      return replayManifestError(`sessions[${index}].session_id duplicates root_session_id`)
    }
    if (seen.has(sessionId)) return replayManifestError(`duplicate session_id ${sessionId}`)
    seen.add(sessionId)
    if (typeof parentSessionId !== "string" || !parentSessionId) {
      return replayManifestError(`sessions[${index}].parent_session_id must be a non-empty string`)
    }
    if (typeof spawnCallId !== "string" || !spawnCallId) {
      return replayManifestError(`sessions[${index}].spawn_call_id must be a non-empty string`)
    }
    if (!Number.isInteger(spawnIndex) || (spawnIndex as number) < 0) {
      return replayManifestError(`sessions[${index}].spawn_index must be a non-negative integer`)
    }
    const spawnKey = `${parentSessionId}\u0000${spawnCallId}`
    if (seenSpawns.has(spawnKey)) {
      return replayManifestError(`duplicate spawn_call_id ${spawnCallId} in parent ${parentSessionId}`)
    }
    seenSpawns.add(spawnKey)
    if (!Array.isArray(value.messages)) {
      return replayManifestError(`sessions[${index}].messages must be an array`)
    }

    const parsed = parseReplayMessages(JSON.stringify(value.messages))
    return {
      sessionId,
      parentSessionId,
      spawnCallId,
      spawnIndex: spawnIndex as number,
      ...(typeof value.subagent_type === "string" ? { subagentType: value.subagent_type } : {}),
      messageCount: value.messages.length,
      // Unlike the root session, every later child user message is recreated
      // by replaying its parent's task(task_id=...) call. Injecting the text
      // here as well would duplicate resumed-task prompts.
      replayTurns: parsed.replayTurns.map(({ content, toolCalls }) => ({
        content,
        ...(toolCalls ? { toolCalls } : {}),
      })),
    }
  })

  const knownParents = new Set([input.root_session_id, ...sessions.map((session) => session.sessionId)])
  for (const session of sessions) {
    if (!knownParents.has(session.parentSessionId)) {
      return replayManifestError(
        `session ${session.sessionId} references unknown parent_session_id ${session.parentSessionId}`,
      )
    }
  }

  const reachable = new Set([input.root_session_id])
  let changed = true
  while (changed) {
    changed = false
    for (const session of sessions) {
      if (reachable.has(session.sessionId) || !reachable.has(session.parentSessionId)) continue
      reachable.add(session.sessionId)
      changed = true
    }
  }
  const unreachable = sessions.find((session) => !reachable.has(session.sessionId))
  if (unreachable) {
    return replayManifestError(`session ${unreachable.sessionId} is not reachable from root_session_id`)
  }

  return {
    version: 1,
    rootSessionId: input.root_session_id,
    sessions,
  }
}
