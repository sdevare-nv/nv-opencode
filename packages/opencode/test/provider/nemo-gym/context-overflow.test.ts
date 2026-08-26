import { describe, expect, mock, test } from "bun:test"
import type { LanguageModelV3CallOptions, LanguageModelV3StreamPart } from "@ai-sdk/provider"
import { NemoGymLanguageModel } from "@/provider/sdk/nemo-gym/language-model"

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
  tools: [{ type: "function", name: "bash", inputSchema: { type: "object", properties: {} } }],
}

describe("NemoGymLanguageModel context overflow", () => {
  test("recognizes Gym's null-content length completion as context overflow", async () => {
    const originalFetch = globalThis.fetch
    const fetchSpy = mock(
      async () =>
        new Response(
          JSON.stringify({
            id: "chatcmpl-123",
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

    try {
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
    } finally {
      globalThis.fetch = originalFetch
    }
  })
})
