import { NemoGymLanguageModel } from "@/provider/sdk/nemo-gym/language-model"
import { describe, test, expect, mock, beforeEach, afterEach } from "bun:test"
import type { LanguageModelV3CallOptions, LanguageModelV3Prompt } from "@ai-sdk/provider"
import { promises as fs } from "node:fs"
import os from "node:os"
import path from "node:path"

// Covers the on-policy segment tracking added for opencode compaction support
// (language-model.ts _nextSegment / _sessionFromHeaders). A compaction
// summarization call (tagged via the "x-turn-kind: compaction" header set in
// session/llm.ts when agent.name === "compaction") rebuilds its prompt from
// stored history rather than continuing the prior turn's token stream, and
// the turn after it conditions on the rewritten/summarized session — neither
// edge is prefix-contiguous with what came before, so both must land in a
// fresh segment. See responses_api_agents/swe_agents app.py
// get_all_session_trajectories_from_completions for the Python-side consumer.

const TEST_PROMPT: LanguageModelV3Prompt = [{ role: "user", content: [{ type: "text", text: "hi" }] }]

type Dump = {
  session_id: string
  parent_session_id: string | null
  turn: number
  segment_index: number
  segment_boundary_reason: string | null
}

async function drain(stream: ReadableStream<unknown>) {
  const reader = stream.getReader()
  while (true) {
    const { done } = await reader.read()
    if (done) break
  }
}

function fakeChatResponse(text: string) {
  return new Response(
    JSON.stringify({
      id: "resp-1",
      model: "test-model",
      created: 0,
      choices: [{ index: 0, finish_reason: "stop", message: { role: "assistant", content: text } }],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    }),
    { status: 200, headers: { "Content-Type": "application/json" } },
  )
}

function callOptions(headers: Record<string, string>): LanguageModelV3CallOptions {
  return { prompt: TEST_PROMPT, headers } as LanguageModelV3CallOptions
}

describe("NemoGymLanguageModel segment tracking", () => {
  let completionsDir: string
  let originalFetch: typeof fetch

  beforeEach(async () => {
    completionsDir = await fs.mkdtemp(path.join(os.tmpdir(), "nemo-gym-seg-"))
    originalFetch = global.fetch
    global.fetch = mock(async () => fakeChatResponse("ok")) as unknown as typeof fetch
  })

  afterEach(() => {
    global.fetch = originalFetch
  })

  function makeModel() {
    return new NemoGymLanguageModel("test-model", {
      provider: "nemo-gym",
      baseURL: "http://localhost:1",
      completionsDir,
      instanceId: "inst-1",
    })
  }

  async function readDumps(): Promise<Dump[]> {
    const files = await fs.readdir(completionsDir)
    const rows = await Promise.all(
      files
        .filter((f) => f.endsWith(".json"))
        .map(async (f) => JSON.parse(await fs.readFile(path.join(completionsDir, f), "utf8")) as Dump),
    )
    rows.sort((a, b) => a.turn - b.turn)
    return rows
  }

  test("ordinary turns in one session stay in segment 0 with no boundary", async () => {
    const model = makeModel()
    for (let i = 0; i < 3; i++) {
      await drain((await model.doStream(callOptions({ "x-session-affinity": "ses-a" }))).stream)
    }
    const dumps = await readDumps()
    expect(dumps.map((d) => d.segment_index)).toEqual([0, 0, 0])
    expect(dumps.every((d) => d.segment_boundary_reason === null)).toBe(true)
  })

  test("a compaction turn isolates itself and starts a fresh segment right after it", async () => {
    const model = makeModel()
    const sid = "ses-b"
    await drain((await model.doStream(callOptions({ "x-session-affinity": sid }))).stream) // turn 0
    await drain(
      (await model.doStream(callOptions({ "x-session-affinity": sid, "x-turn-kind": "compaction" }))).stream,
    ) // turn 1 (compaction call itself)
    await drain((await model.doStream(callOptions({ "x-session-affinity": sid }))).stream) // turn 2 (first post-compaction turn)
    await drain((await model.doStream(callOptions({ "x-session-affinity": sid }))).stream) // turn 3 (stays in the new segment)

    const dumps = await readDumps()
    expect(dumps.map((d) => d.segment_index)).toEqual([0, 1, 2, 2])
    expect(dumps.map((d) => d.segment_boundary_reason)).toEqual([null, "compaction", "post_compaction", null])
  })

  test("repeated compaction in one session keeps bumping the segment", async () => {
    const model = makeModel()
    const sid = "ses-c"
    await drain((await model.doStream(callOptions({ "x-session-affinity": sid }))).stream) // seg 0
    await drain(
      (await model.doStream(callOptions({ "x-session-affinity": sid, "x-turn-kind": "compaction" }))).stream,
    ) // seg 1
    await drain((await model.doStream(callOptions({ "x-session-affinity": sid }))).stream) // seg 2
    await drain(
      (await model.doStream(callOptions({ "x-session-affinity": sid, "x-turn-kind": "compaction" }))).stream,
    ) // seg 3
    await drain((await model.doStream(callOptions({ "x-session-affinity": sid }))).stream) // seg 4

    const dumps = await readDumps()
    expect(dumps.map((d) => d.segment_index)).toEqual([0, 1, 2, 3, 4])
  })

  test("segment counters are independent per session", async () => {
    const model = makeModel()
    await drain(
      (await model.doStream(callOptions({ "x-session-affinity": "main", "x-turn-kind": "compaction" }))).stream,
    )
    await drain((await model.doStream(callOptions({ "x-session-affinity": "sub-1" }))).stream)

    const dumps = await readDumps()
    const bySession = Object.fromEntries(dumps.map((d) => [d.session_id, d.segment_index]))
    expect(bySession["main"]).toBe(1)
    expect(bySession["sub-1"]).toBe(0)
  })

  test("title-generation turns are excluded entirely and don't consume a turn/segment slot", async () => {
    const model = makeModel()
    const sid = "ses-title"
    // opencode fires the title call under the same sessionID before the real
    // conversation's own seed (session/prompt.ts:192-212) — no tools, a
    // completely different system prompt, sharing no token continuity.
    await drain((await model.doStream(callOptions({ "x-session-affinity": sid, "x-turn-kind": "title" }))).stream)
    await drain((await model.doStream(callOptions({ "x-session-affinity": sid }))).stream) // real turn 0
    await drain((await model.doStream(callOptions({ "x-session-affinity": sid }))).stream) // real turn 1

    const dumps = await readDumps()
    expect(dumps.length).toBe(2)
    expect(dumps.map((d) => d.turn)).toEqual([0, 1])
    expect(dumps.every((d) => d.segment_index === 0)).toBe(true)
  })
})
