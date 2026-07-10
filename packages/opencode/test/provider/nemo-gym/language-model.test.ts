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

const CALL_OPTIONS: LanguageModelV3CallOptions = {
  prompt: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
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
    const fetchSpy = mock(async () =>
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

  test("replay is scoped to the main session — subagent sessions call through to HTTP", async () => {
    const fetchSpy = mock(async () =>
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

    const subagentOptions: LanguageModelV3CallOptions = {
      ...CALL_OPTIONS,
      headers: { "x-session-affinity": "ses_subagent_1" },
    }
    const { stream } = await model.doStream(subagentOptions)
    const parts = await drain(stream)
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(parts.find((p) => p.type === "text-delta")).toMatchObject({ delta: "subagent turn" })
  })
})
