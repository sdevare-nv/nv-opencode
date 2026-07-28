import { describe, expect, test } from "bun:test"
import fs from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import { collectCompletionMetrics, parseToolExecutionMetric, updateNemoGymMetrics } from "../../src/bench/metrics"

async function withTempDir(run: (directory: string) => Promise<void>) {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "opencode-metrics-"))
  try {
    await run(directory)
  } finally {
    await fs.rm(directory, { recursive: true, force: true })
  }
}

describe("bench metrics", () => {
  test("extracts an exact completed tool span", () => {
    const metric = parseToolExecutionMetric(
      JSON.stringify({
        type: "tool_use",
        part: {
          type: "tool",
          tool: "bash",
          callID: "call-1",
          state: {
            status: "completed",
            title: "Run focused tests",
            time: { start: 1_000, end: 3_500 },
          },
        },
      }),
    )

    expect(metric).toEqual({
      observation_type: "bash",
      observation_id: "call-1",
      latency: 2.5,
      message: "Run focused tests",
      timestamp: "1970-01-01T00:00:03.500Z",
    })
    expect(parseToolExecutionMetric("not json")).toBeUndefined()
  })

  test("collects and chronologically sorts completion timing and usage", async () => {
    await withTempDir(async (directory) => {
      const later = {
        response: {
          id: "response-2",
          model: "model-b",
          usage: { prompt_tokens: 20, completion_tokens: 4 },
        },
        latency: 2,
        timestamp: 20,
      }
      const earlier = {
        response: {
          id: "response-1",
          usage: {
            prompt_tokens: 10,
            completion_tokens: 3,
            prompt_tokens_details: { cached_tokens: 2 },
          },
        },
        latency: 1.25,
        timestamp: 10,
      }
      await fs.writeFile(path.join(directory, "later.json"), JSON.stringify(later))
      await fs.writeFile(path.join(directory, "earlier.json"), JSON.stringify(earlier))
      await fs.writeFile(path.join(directory, "incomplete.json"), JSON.stringify({ response: {} }))

      const metrics = await collectCompletionMetrics(directory, "fallback-model")

      expect(metrics.responseLatencies).toEqual([
        {
          model: "fallback-model",
          latency: 1.25,
          response_id: "response-1",
          timestamp: "1970-01-01T00:00:10.000Z",
        },
        {
          model: "model-b",
          latency: 2,
          response_id: "response-2",
          timestamp: "1970-01-01T00:00:20.000Z",
        },
      ])
      expect(metrics.tokenUsages).toEqual([
        {
          model: "fallback-model",
          prompt_tokens: 10,
          completion_tokens: 3,
          cache_read_tokens: 2,
          cache_write_tokens: 0,
          context_window: 0,
          per_turn_token: 13,
          response_id: "response-1",
        },
        {
          model: "model-b",
          prompt_tokens: 20,
          completion_tokens: 4,
          cache_read_tokens: 0,
          cache_write_tokens: 0,
          context_window: 0,
          per_turn_token: 24,
          response_id: "response-2",
        },
      ])
    })
  })

  test("atomically merges the NeMo Gym metrics file", async () => {
    await withTempDir(async (directory) => {
      const metricsPath = path.join(directory, "nemo_gym_metrics.json")
      await fs.writeFile(metricsPath, JSON.stringify({ ray_queue_time: 1.5 }))

      await updateNemoGymMetrics(metricsPath, {
        total_model_call_time: 3.25,
        create_runtime_time: 0,
      })

      expect(JSON.parse(await fs.readFile(metricsPath, "utf8"))).toEqual({
        ray_queue_time: 1.5,
        total_model_call_time: 3.25,
        create_runtime_time: 0,
      })
    })
  })
})
