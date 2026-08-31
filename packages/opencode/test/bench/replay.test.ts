import { describe, test, expect } from "bun:test"
import { parseReplayManifest, parseReplayMessages, replayMessageText } from "@/bench/replay"

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

  test("skips system and tool messages", () => {
    const raw = JSON.stringify([
      { role: "system", content: "sys prompt" },
      { role: "user", content: "fix the bug" },
      {
        role: "assistant",
        content: null,
        tool_calls: [{ id: "call_1", type: "function", function: { name: "bash", arguments: '{"cmd":"ls"}' } }],
      },
      { role: "tool", content: "file1.py\n", tool_call_id: "call_1" },
      { role: "assistant", content: "Done.", tool_calls: undefined },
    ])
    const { initialUserText, replayTurns } = parseReplayMessages(raw)
    expect(initialUserText).toBe("fix the bug")
    expect(replayTurns).toEqual([
      { content: null, toolCalls: [{ id: "call_1", name: "bash", arguments: '{"cmd":"ls"}' }] },
      { content: "Done.", toolCalls: undefined },
    ])
  })

  test("attaches a subsequent user message to the turn it precedes, not dropped", () => {
    const raw = JSON.stringify([
      { role: "user", content: "fix the bug" },
      {
        role: "assistant",
        content: null,
        tool_calls: [{ id: "call_1", type: "function", function: { name: "bash", arguments: '{"cmd":"ls"}' } }],
      },
      { role: "tool", content: "file1.py\n", tool_call_id: "call_1" },
      { role: "user", content: "please also fix the other bug" },
      { role: "assistant", content: "Done.", tool_calls: undefined },
    ])
    const { initialUserText, replayTurns } = parseReplayMessages(raw)
    expect(initialUserText).toBe("fix the bug")
    expect(replayTurns).toEqual([
      { content: null, toolCalls: [{ id: "call_1", name: "bash", arguments: '{"cmd":"ls"}' }] },
      { content: "Done.", toolCalls: undefined, precedingUserTexts: ["please also fix the other bug"] },
    ])
  })

  test("collects multiple consecutive subsequent user messages onto the same turn, in order", () => {
    const raw = JSON.stringify([
      { role: "user", content: "fix the bug" },
      { role: "assistant", content: "ok" },
      { role: "user", content: "also do X" },
      { role: "user", content: "and Y" },
      { role: "assistant", content: "done" },
    ])
    const { replayTurns } = parseReplayMessages(raw)
    expect(replayTurns[1].precedingUserTexts).toEqual(["also do X", "and Y"])
  })

  test("returns trailing user messages separately when the trajectory ends on a user turn", () => {
    const raw = JSON.stringify([
      { role: "user", content: "fix the bug" },
      {
        role: "assistant",
        content: null,
        tool_calls: [{ id: "call_1", type: "function", function: { name: "bash", arguments: '{"cmd":"ls"}' } }],
      },
      { role: "tool", content: "file1.py\n", tool_call_id: "call_1" },
      { role: "user", content: "now also check the tests" },
    ])
    const { replayTurns, trailingUserTexts } = parseReplayMessages(raw)
    expect(replayTurns).toEqual([
      { content: null, toolCalls: [{ id: "call_1", name: "bash", arguments: '{"cmd":"ls"}' }] },
    ])
    expect(trailingUserTexts).toEqual(["now also check the tests"])
  })

  test("omits empty subsequent user message text", () => {
    const raw = JSON.stringify([
      { role: "user", content: "fix the bug" },
      { role: "assistant", content: "ok" },
      { role: "user", content: "" },
      { role: "assistant", content: "done" },
    ])
    const { replayTurns } = parseReplayMessages(raw)
    expect(replayTurns[1].precedingUserTexts).toBeUndefined()
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
      {
        role: "user",
        content: [
          { type: "text", text: "part one" },
          { type: "text", text: "part two" },
        ],
      },
    ])
    const { initialUserText } = parseReplayMessages(raw)
    expect(initialUserText).toBe("part one\npart two")
  })

  test("throws when no user message is present", () => {
    const raw = JSON.stringify([{ role: "system", content: "sys prompt" }])
    expect(() => parseReplayMessages(raw)).toThrow(/no user message/)
  })
})

describe("parseReplayManifest", () => {
  test("parses child and nested-child replay queues without relying on array order", () => {
    const manifest = parseReplayManifest(
      JSON.stringify({
        version: 1,
        root_session_id: "recorded-root",
        sessions: [
          {
            session_id: "recorded-grandchild",
            parent_session_id: "recorded-child",
            spawn_call_id: "call_nested",
            spawn_index: 0,
            subagent_type: "explore",
            messages: [
              { role: "user", content: "nested work" },
              { role: "assistant", content: "nested result" },
            ],
          },
          {
            session_id: "recorded-child",
            parent_session_id: "recorded-root",
            spawn_call_id: "call_child",
            spawn_index: 1,
            subagent_type: "general",
            messages: [
              { role: "user", content: "child work" },
              {
                role: "assistant",
                content: null,
                tool_calls: [
                  {
                    id: "call_nested",
                    function: { name: "task", arguments: '{"prompt":"nested work"}' },
                  },
                ],
              },
            ],
          },
        ],
      }),
    )

    expect(manifest.rootSessionId).toBe("recorded-root")
    expect(manifest.sessions.map((session) => session.sessionId)).toEqual(["recorded-grandchild", "recorded-child"])
    expect(manifest.sessions[0]).toMatchObject({
      parentSessionId: "recorded-child",
      spawnCallId: "call_nested",
      messageCount: 2,
      replayTurns: [{ content: "nested result" }],
    })
  })

  test("rejects an unlinked parent", () => {
    expect(() =>
      parseReplayManifest(
        JSON.stringify({
          version: 1,
          root_session_id: "root",
          sessions: [
            {
              session_id: "child",
              parent_session_id: "missing",
              spawn_call_id: "call_1",
              spawn_index: 0,
              messages: [{ role: "user", content: "work" }],
            },
          ],
        }),
      ),
    ).toThrow(/unknown parent_session_id/)
  })
})
