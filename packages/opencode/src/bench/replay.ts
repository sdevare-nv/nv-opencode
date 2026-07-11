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

import type { NemoGymReplayTurn } from "../provider/sdk/nemo-gym/language-model"

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
