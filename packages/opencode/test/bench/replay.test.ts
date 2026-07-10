import { describe, test, expect } from "bun:test"
import { parseReplayMessages, replayMessageText } from "@/bench/replay"

describe("replayMessageText", () => {
  test("returns string content verbatim", () => {
    expect(replayMessageText("hello")).toBe("hello")
  })

  test("joins text parts from array content", () => {
    expect(
      replayMessageText([
        { type: "text", text: "line one" },
        { type: "text", text: "line two" },
      ]),
    ).toBe("line one\nline two")
  })

  test("returns empty string for null/undefined content", () => {
    expect(replayMessageText(null)).toBe("")
    expect(replayMessageText(undefined)).toBe("")
  })
})

describe("parseReplayMessages", () => {
  test("extracts the first user message as the initial task instruction", () => {
    const raw = JSON.stringify([
      { role: "system", content: "sys prompt" },
      { role: "user", content: "fix the bug" },
    ])
    const { initialUserText, replayTurns } = parseReplayMessages(raw)
    expect(initialUserText).toBe("fix the bug")
    expect(replayTurns).toEqual([])
  })

  test("skips system, tool, and subsequent user messages", () => {
    const raw = JSON.stringify([
      { role: "system", content: "sys prompt" },
      { role: "user", content: "fix the bug" },
      {
        role: "assistant",
        content: null,
        tool_calls: [{ id: "call_1", type: "function", function: { name: "bash", arguments: '{"cmd":"ls"}' } }],
      },
      { role: "tool", content: "file1.py\n", tool_call_id: "call_1" },
      { role: "user", content: "please continue" },
      { role: "assistant", content: "Done.", tool_calls: undefined },
    ])
    const { initialUserText, replayTurns } = parseReplayMessages(raw)
    expect(initialUserText).toBe("fix the bug")
    expect(replayTurns).toEqual([
      { content: null, toolCalls: [{ id: "call_1", name: "bash", arguments: '{"cmd":"ls"}' }] },
      { content: "Done.", toolCalls: undefined },
    ])
  })

  test("preserves tool_call ids verbatim, including multiple calls in one turn", () => {
    const raw = JSON.stringify([
      { role: "user", content: "fix the bug" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          { id: "call_abc", type: "function", function: { name: "read", arguments: '{"path":"a.py"}' } },
          { id: "call_def", type: "function", function: { name: "read", arguments: '{"path":"b.py"}' } },
        ],
      },
    ])
    const { replayTurns } = parseReplayMessages(raw)
    expect(replayTurns[0].toolCalls?.map((tc) => tc.id)).toEqual(["call_abc", "call_def"])
  })

  test("joins array-of-parts user content for the initial instruction", () => {
    const raw = JSON.stringify([
      { role: "user", content: [{ type: "text", text: "part one" }, { type: "text", text: "part two" }] },
    ])
    const { initialUserText } = parseReplayMessages(raw)
    expect(initialUserText).toBe("part one\npart two")
  })

  test("throws when no user message is present", () => {
    const raw = JSON.stringify([{ role: "system", content: "sys prompt" }])
    expect(() => parseReplayMessages(raw)).toThrow(/no user message/)
  })
})
