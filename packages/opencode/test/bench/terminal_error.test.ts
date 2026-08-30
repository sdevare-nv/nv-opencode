import { describe, expect, test } from "bun:test"
import * as BenchTerminalError from "@/bench/terminal_error"

describe("bench terminal error signals", () => {
  test("detects max-iteration and context-window markers", () => {
    expect(BenchTerminalError.detect(`before ${BenchTerminalError.encode("max_iteration")} after`)).toBe(
      "max_iteration",
    )
    expect(BenchTerminalError.detect(BenchTerminalError.encode("context_window"))).toBe("context_window")
    expect(BenchTerminalError.detect("ordinary opencode stderr")).toBeUndefined()
  })

  test("prefers context overflow when both terminal states occur", () => {
    expect(BenchTerminalError.prefer("max_iteration", "context_window")).toBe("context_window")
    expect(BenchTerminalError.prefer("context_window", "max_iteration")).toBe("context_window")
  })

  test("writes errors that Gym classifies and preserves ordinary exit errors", () => {
    expect(BenchTerminalError.toGymError(0, "max_iteration")).toBe("maximum iteration reached")
    expect(BenchTerminalError.toGymError(0, "context_window")).toBe("context window exceeded")
    expect(BenchTerminalError.toGymError(17)).toBe("opencode_exit_17")
    expect(BenchTerminalError.toGymError(0)).toBeNull()
  })

  test("keeps terminal trajectories even when opencode exits nonzero", () => {
    expect(BenchTerminalError.shouldExitSuccessfully(1, "context_window")).toBeTrue()
    expect(BenchTerminalError.shouldExitSuccessfully(1, "max_iteration")).toBeTrue()
    expect(BenchTerminalError.shouldExitSuccessfully(1)).toBeFalse()
  })
})
