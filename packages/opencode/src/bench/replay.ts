/**
 * Parses a replay-messages file (prior chat-completion-format trajectory) for
 * the SWE-bench bench harness. Split out of `bench/cli.ts` so it's importable
 * without triggering `cli.ts`'s top-level `main()` (which calls
 * `process.exit` on missing/invalid CLI args — not test-friendly).
 *
 * Mirrors OpenHands' `messages_to_replay_events()` skip rules (see
 * `temp/nv-OpenHands/evaluation/benchmarks/swe_bench/replay_utils.py`):
 * system / tool / subsequent-user messages are skipped — the first user
 * message becomes the task instruction, assistant messages become scripted
 * turns replayed in order by the nemo-gym provider (see language-model.ts).
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

export function parseReplayMessages(raw: string): { initialUserText: string; replayTurns: NemoGymReplayTurn[] } {
  const messages = JSON.parse(raw) as ReplayChatMessage[]

  let initialUserText: string | undefined
  const replayTurns: NemoGymReplayTurn[] = []

  for (const msg of messages) {
    if (msg.role === "system" || msg.role === "tool") continue
    if (msg.role === "user") {
      if (initialUserText === undefined) initialUserText = replayMessageText(msg.content)
      continue
    }
    if (msg.role === "assistant") {
      replayTurns.push({
        content: typeof msg.content === "string" ? msg.content : replayMessageText(msg.content) || null,
        toolCalls: msg.tool_calls?.map((tc) => ({ id: tc.id, name: tc.function.name, arguments: tc.function.arguments })),
      })
    }
  }

  if (initialUserText === undefined) {
    throw new Error("replay-messages-file: no user message found (expected at least one task-instruction message)")
  }

  return { initialUserText, replayTurns }
}
