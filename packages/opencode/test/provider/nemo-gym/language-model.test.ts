import { describe, test, expect, mock } from "bun:test"
import { NemoGymLanguageModel } from "@/provider/sdk/nemo-gym/language-model"
import type { LanguageModelV3CallOptions, LanguageModelV3StreamPart } from "@ai-sdk/provider"

async function drain(stream: ReadableStream<LanguageModelV3StreamPart>): Promise<LanguageModelV3StreamPart[]> {
  const reader = stream.getReader()
  const parts: LanguageModelV3StreamPart[] = []
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    parts.push(value)
  }
  return parts
}

// Real agentic-loop calls always carry the resolved tool registry
// (session/prompt.ts's resolveTools()) — this is what actually distinguishes
// them from auxiliary same-session calls like title/summary generation,
// which never pass tools. See _popReplayTurn's docblock in language-model.ts.
const AGENT_TOOLS: LanguageModelV3CallOptions["tools"] = [
  { type: "function", name: "bash", inputSchema: { type: "object", properties: {} } },
]

const CALL_OPTIONS: LanguageModelV3CallOptions = {
  prompt: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
  tools: AGENT_TOOLS,
}

describe("NemoGymLanguageModel replay", () => {
  test("doStream replays a scripted tool-call turn without hitting the network", async () => {
    const fetchSpy = mock(async () => {
      throw new Error("network should not be called during replay")
    })
    // @ts-expect-error test override
    globalThis.fetch = fetchSpy

    const model = new NemoGymLanguageModel("test-model", {
      provider: "nemo-gym",
      baseURL: "http://unused.invalid",
      replayTurns: [{ content: null, toolCalls: [{ id: "call_1", name: "bash", arguments: '{"cmd":"ls"}' }] }],
    })

    const { stream } = await model.doStream(CALL_OPTIONS)
    const parts = await drain(stream)

    expect(fetchSpy).not.toHaveBeenCalled()

    const toolCall = parts.find((p) => p.type === "tool-call")
    expect(toolCall).toMatchObject({ toolCallId: "call_1", toolName: "bash", input: '{"cmd":"ls"}' })

    const finish = parts.find((p) => p.type === "finish")
    expect(finish).toMatchObject({ finishReason: { unified: "tool-calls" } })
  })

  test("doStream replays a scripted text-only turn (no tool calls) as finishReason stop", async () => {
    const model = new NemoGymLanguageModel("test-model", {
      provider: "nemo-gym",
      baseURL: "http://unused.invalid",
      replayTurns: [{ content: "All done." }],
    })

    const { stream } = await model.doStream(CALL_OPTIONS)
    const parts = await drain(stream)

    expect(parts.some((p) => p.type === "tool-call")).toBe(false)
    const textDelta = parts.find((p) => p.type === "text-delta")
    expect(textDelta).toMatchObject({ delta: "All done." })
    const finish = parts.find((p) => p.type === "finish")
    expect(finish).toMatchObject({ finishReason: { unified: "stop" } })
  })

  test("doStream falls through to the real HTTP path once the replay queue is exhausted", async () => {
    const fetchSpy = mock(
      async () =>
        new Response(
          JSON.stringify({
            id: "resp_1",
            model: "test-model",
            choices: [{ finish_reason: "stop", message: { role: "assistant", content: "live turn" } }],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
    )
    // @ts-expect-error test override
    globalThis.fetch = fetchSpy

    const model = new NemoGymLanguageModel("test-model", {
      provider: "nemo-gym",
      baseURL: "http://unused.invalid",
      replayTurns: [{ content: "scripted turn" }],
    })

    // First call: replay (no fetch).
    await drain((await model.doStream(CALL_OPTIONS)).stream)
    expect(fetchSpy).not.toHaveBeenCalled()

    // Second call: replay queue exhausted -> real HTTP path.
    const { stream } = await model.doStream(CALL_OPTIONS)
    const parts = await drain(stream)
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    const textDelta = parts.find((p) => p.type === "text-delta")
    expect(textDelta).toMatchObject({ delta: "live turn" })
  })

  test("doStream recognizes Gym's null-content length completion as context overflow", async () => {
    const fetchSpy = mock(
      async () =>
        new Response(
          JSON.stringify({
            id: "chtcmpl-123",
            model: "test-model",
            choices: [
              {
                index: 0,
                finish_reason: "length",
                message: { role: "assistant", content: null, tool_calls: null },
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
    )
    // @ts-expect-error test override
    globalThis.fetch = fetchSpy

    const model = new NemoGymLanguageModel("test-model", {
      provider: "nemo-gym",
      baseURL: "http://unused.invalid",
      retries: Number.MAX_SAFE_INTEGER,
    })

    const parts = await drain((await model.doStream(CALL_OPTIONS)).stream)
    expect(fetchSpy).toHaveBeenCalledTimes(1)

    const error = parts.find((part) => part.type === "error")
    expect(error?.type).toBe("error")
    if (error?.type !== "error" || typeof error.error !== "string") throw new Error("missing stream error")
    expect(JSON.parse(error.error)).toMatchObject({
      type: "error",
      error: { code: "context_length_exceeded" },
    })
  })

  test("replay is scoped to the top-level session — a subagent session (x-parent-session-id set) calls through to HTTP", async () => {
    const fetchSpy = mock(
      async () =>
        new Response(
          JSON.stringify({
            id: "resp_1",
            model: "test-model",
            choices: [{ finish_reason: "stop", message: { role: "assistant", content: "subagent turn" } }],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
    )
    // @ts-expect-error test override
    globalThis.fetch = fetchSpy

    const model = new NemoGymLanguageModel("test-model", {
      provider: "nemo-gym",
      baseURL: "http://unused.invalid",
      replayTurns: [{ content: "scripted turn" }],
    })

    // A subagent session carries its OWN session id plus x-parent-session-id
    // pointing at the main session — session/llm.ts sets both unconditionally
    // for every call, subagent or not, so it's parentSessionID's presence
    // that identifies a subagent, not the session id string itself.
    const subagentOptions: LanguageModelV3CallOptions = {
      ...CALL_OPTIONS,
      headers: { "x-session-affinity": "ses_subagent_1", "x-parent-session-id": "ses_main" },
    }
    const { stream } = await model.doStream(subagentOptions)
    const parts = await drain(stream)
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(parts.find((p) => p.type === "text-delta")).toMatchObject({ delta: "subagent turn" })
  })

  test("an auxiliary no-tool call on the main session (e.g. opencode's own title/summary generation) does not consume the replay queue", async () => {
    const fetchSpy = mock(
      async () =>
        new Response(
          JSON.stringify({
            id: "resp_1",
            model: "test-model",
            choices: [{ finish_reason: "stop", message: { role: "assistant", content: "Fix the parser bug" } }],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
    )
    // @ts-expect-error test override
    globalThis.fetch = fetchSpy

    const model = new NemoGymLanguageModel("test-model", {
      provider: "nemo-gym",
      baseURL: "http://unused.invalid",
      replayTurns: [{ content: null, toolCalls: [{ id: "call_1", name: "bash", arguments: "{}" }] }],
    })

    // session/prompt.ts's runLoop forks off a title-generation call on step 1,
    // using the SAME session id as the real agentic loop but never passing
    // tools. Real x-session-affinity value from opencode: same session,
    // no x-parent-session-id (it's not a subagent), no tools.
    const titleGenOptions: LanguageModelV3CallOptions = {
      prompt: [{ role: "user", content: [{ type: "text", text: "Generate a title" }] }],
      headers: { "x-session-affinity": "ses_main" },
      // no tools
    }
    const { stream } = await model.doStream(titleGenOptions)
    await drain(stream)
    expect(fetchSpy).toHaveBeenCalledTimes(1) // went straight to HTTP, not replayed

    // The scripted turn is still there for the real agent's own next call
    // (same session id, this time with tools).
    const realOptions: LanguageModelV3CallOptions = {
      ...CALL_OPTIONS,
      headers: { "x-session-affinity": "ses_main" },
    }
    const realFetchSpy = mock(async () => {
      throw new Error("network should not be called — replay turn should still be available")
    })
    // @ts-expect-error test override
    globalThis.fetch = realFetchSpy
    const { stream: realStream } = await model.doStream(realOptions)
    const parts = await drain(realStream)
    expect(realFetchSpy).not.toHaveBeenCalled()
    expect(parts.find((p) => p.type === "tool-call")).toMatchObject({ toolCallId: "call_1" })
  })

  function fetchSpyCapturingBody() {
    const bodies: Array<{ messages: unknown }> = []
    const fetchSpy = mock(async (_url: unknown, init: { body?: string }) => {
      bodies.push(JSON.parse(init.body ?? "{}"))
      return new Response(
        JSON.stringify({
          id: "resp_1",
          model: "test-model",
          choices: [{ finish_reason: "stop", message: { role: "assistant", content: "live turn" } }],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      )
    })
    // @ts-expect-error test override
    globalThis.fetch = fetchSpy
    return { fetchSpy, bodies }
  }

  test("subsequent user message (precedingUserTexts) is spliced at the right position relative to already-replayed turns", async () => {
    const { bodies } = fetchSpyCapturingBody()

    const model = new NemoGymLanguageModel("test-model", {
      provider: "nemo-gym",
      baseURL: "http://unused.invalid",
      // Attached to turn 0 (ordinal 0): must land BEFORE the first assistant
      // message in the eventual wire request, i.e. right after the initial
      // user message.
      replayTurns: [
        {
          content: null,
          toolCalls: [{ id: "call_1", name: "bash", arguments: "{}" }],
          precedingUserTexts: ["please also fix the other bug"],
        },
      ],
    })

    // Turn 0 replays without hitting the network.
    await drain((await model.doStream(CALL_OPTIONS)).stream)
    expect(bodies).toHaveLength(0)

    // Live call, with a realistic prompt reflecting what the session would
    // actually contain by now: the initial user message, then the assistant
    // turn + tool result that were just replayed for real.
    const grownPrompt: LanguageModelV3CallOptions["prompt"] = [
      { role: "user", content: [{ type: "text", text: "fix the bug" }] },
      { role: "assistant", content: [{ type: "tool-call", toolCallId: "call_1", toolName: "bash", input: {} }] },
      {
        role: "tool",
        content: [
          { type: "tool-result", toolCallId: "call_1", toolName: "bash", output: { type: "text", value: "ok" } },
        ],
      },
    ]
    await drain((await model.doStream({ prompt: grownPrompt, tools: AGENT_TOOLS })).stream)
    expect(bodies).toHaveLength(1)

    const messages = bodies[0].messages as unknown as Array<{ role: string; content?: unknown }>
    expect(messages[0]).toMatchObject({ role: "user", content: "fix the bug" })
    const injectedIdx = messages.findIndex((m) => m.role === "user" && m.content === "please also fix the other bug")
    const firstAssistantIdx = messages.findIndex((m) => m.role === "assistant")
    expect(injectedIdx).toBeGreaterThan(-1)
    expect(firstAssistantIdx).toBeGreaterThan(-1)
    // Injected immediately before the assistant turn it originally
    // preceded, not appended somewhere arbitrary — and strictly after the
    // initial task-instruction message.
    expect(injectedIdx).toBe(firstAssistantIdx - 1)
    expect(injectedIdx).toBeGreaterThan(0)
  })

  test("injected user message persists on every subsequent live call, not just the first", async () => {
    const { bodies } = fetchSpyCapturingBody()

    const model = new NemoGymLanguageModel("test-model", {
      provider: "nemo-gym",
      baseURL: "http://unused.invalid",
      replayTurns: [{ content: "ok", precedingUserTexts: ["please also fix the other bug"] }],
    })

    await drain((await model.doStream(CALL_OPTIONS)).stream) // scripted
    await drain((await model.doStream(CALL_OPTIONS)).stream) // live #1
    await drain((await model.doStream(CALL_OPTIONS)).stream) // live #2

    expect(bodies).toHaveLength(2)
    for (const body of bodies) {
      const messages = body.messages as Array<{ role: string; content?: unknown }>
      expect(messages.some((m) => m.role === "user" && m.content === "please also fix the other bug")).toBe(true)
    }
  })

  test("replayTrailingUserTexts is appended once the replay queue is fully drained", async () => {
    const { bodies } = fetchSpyCapturingBody()

    const model = new NemoGymLanguageModel("test-model", {
      provider: "nemo-gym",
      baseURL: "http://unused.invalid",
      replayTurns: [{ content: null, toolCalls: [{ id: "call_1", name: "bash", arguments: "{}" }] }],
      replayTrailingUserTexts: ["now also check the tests"],
    })

    await drain((await model.doStream(CALL_OPTIONS)).stream) // scripted
    await drain((await model.doStream(CALL_OPTIONS)).stream) // live

    expect(bodies).toHaveLength(1)
    const messages = bodies[0].messages as unknown as Array<{ role: string; content?: unknown }>
    expect(messages[messages.length - 1]).toMatchObject({ role: "user", content: "now also check the tests" })
  })

  test("binds parallel child queues by parent task call id, not manifest or execution order", async () => {
    const fetchSpy = mock(async () => {
      throw new Error("network should not be called during replay")
    })
    // @ts-expect-error test override
    globalThis.fetch = fetchSpy

    const model = new NemoGymLanguageModel("test-model", {
      provider: "nemo-gym",
      baseURL: "http://unused.invalid",
      replayTurns: [
        {
          content: null,
          toolCalls: [
            { id: "call_a", name: "task", arguments: '{"prompt":"A","subagent_type":"general"}' },
            { id: "call_b", name: "task", arguments: '{"prompt":"B","subagent_type":"explore"}' },
          ],
        },
      ],
      replayManifest: {
        version: 1,
        rootSessionId: "recorded-root",
        // Deliberately opposite the parent task-call order.
        sessions: [
          {
            sessionId: "recorded-b",
            parentSessionId: "recorded-root",
            spawnCallId: "call_b",
            spawnIndex: 1,
            messageCount: 2,
            replayTurns: [{ content: "result B" }],
          },
          {
            sessionId: "recorded-a",
            parentSessionId: "recorded-root",
            spawnCallId: "call_a",
            spawnIndex: 0,
            messageCount: 2,
            replayTurns: [{ content: "result A" }],
          },
        ],
      },
    })

    await drain((await model.doStream({ ...CALL_OPTIONS, headers: { "x-session-affinity": "live-root" } })).stream)

    const childB = await drain(
      (
        await model.doStream({
          ...CALL_OPTIONS,
          headers: {
            "x-session-affinity": "live-b",
            "x-parent-session-id": "live-root",
            "x-parent-tool-call-id": "call_b",
          },
        })
      ).stream,
    )
    const childA = await drain(
      (
        await model.doStream({
          ...CALL_OPTIONS,
          headers: {
            "x-session-affinity": "live-a",
            "x-parent-session-id": "live-root",
            "x-parent-tool-call-id": "call_a",
          },
        })
      ).stream,
    )

    expect(childB.find((part) => part.type === "text-delta")).toMatchObject({ delta: "result B" })
    expect(childA.find((part) => part.type === "text-delta")).toMatchObject({ delta: "result A" })
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  test("recursively binds a nested subagent to the task call in its recorded parent", async () => {
    const model = new NemoGymLanguageModel("test-model", {
      provider: "nemo-gym",
      baseURL: "http://unused.invalid",
      replayTurns: [
        {
          content: null,
          toolCalls: [{ id: "call_child", name: "task", arguments: "{}" }],
        },
      ],
      replayManifest: {
        version: 1,
        rootSessionId: "recorded-root",
        sessions: [
          {
            sessionId: "recorded-child",
            parentSessionId: "recorded-root",
            spawnCallId: "call_child",
            spawnIndex: 0,
            messageCount: 3,
            replayTurns: [
              {
                content: null,
                toolCalls: [{ id: "call_grandchild", name: "task", arguments: "{}" }],
              },
            ],
          },
          {
            sessionId: "recorded-grandchild",
            parentSessionId: "recorded-child",
            spawnCallId: "call_grandchild",
            spawnIndex: 0,
            messageCount: 2,
            replayTurns: [{ content: "nested result" }],
          },
        ],
      },
    })

    await drain((await model.doStream({ ...CALL_OPTIONS, headers: { "x-session-affinity": "live-root" } })).stream)
    await drain(
      (
        await model.doStream({
          ...CALL_OPTIONS,
          headers: {
            "x-session-affinity": "live-child",
            "x-parent-session-id": "live-root",
            "x-parent-tool-call-id": "call_child",
          },
        })
      ).stream,
    )
    const nested = await drain(
      (
        await model.doStream({
          ...CALL_OPTIONS,
          headers: {
            "x-session-affinity": "live-grandchild",
            "x-parent-session-id": "live-child",
            "x-parent-tool-call-id": "call_grandchild",
          },
        })
      ).stream,
    )

    expect(nested.find((part) => part.type === "text-delta")).toMatchObject({ delta: "nested result" })
  })

  test("rewrites a recorded task_id to the bound live child session id when resuming it", async () => {
    const model = new NemoGymLanguageModel("test-model", {
      provider: "nemo-gym",
      baseURL: "http://unused.invalid",
      replayTurns: [
        { content: null, toolCalls: [{ id: "call_child", name: "task", arguments: "{}" }] },
        {
          content: null,
          toolCalls: [
            {
              id: "call_resume",
              name: "task",
              arguments: '{"task_id":"recorded-child","prompt":"continue"}',
            },
          ],
        },
      ],
      replayManifest: {
        version: 1,
        rootSessionId: "recorded-root",
        sessions: [
          {
            sessionId: "recorded-child",
            parentSessionId: "recorded-root",
            spawnCallId: "call_child",
            spawnIndex: 0,
            messageCount: 2,
            replayTurns: [{ content: "first child result" }],
          },
        ],
      },
    })

    const rootHeaders = { "x-session-affinity": "live-root" }
    await drain((await model.doStream({ ...CALL_OPTIONS, headers: rootHeaders })).stream)
    await drain(
      (
        await model.doStream({
          ...CALL_OPTIONS,
          headers: {
            "x-session-affinity": "live-child",
            "x-parent-session-id": "live-root",
            "x-parent-tool-call-id": "call_child",
          },
        })
      ).stream,
    )
    const resumed = await drain((await model.doStream({ ...CALL_OPTIONS, headers: rootHeaders })).stream)
    const call = resumed.find((part) => part.type === "tool-call")
    expect(call).toMatchObject({
      toolCallId: "call_resume",
      input: '{"task_id":"live-child","prompt":"continue"}',
    })
  })
})
