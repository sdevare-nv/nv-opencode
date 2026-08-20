export type Kind = "max_iteration" | "context_window"

export const ENV = "OPENCODE_BENCH_TERMINAL_SIGNALS"
export const PREFIX = "[opencode-bench-terminal] "

export function encode(kind: Kind): string {
  return PREFIX + kind
}

/** Report terminal agent states from any session, including subagents. */
export function report(kind: Kind): void {
  if (process.env[ENV] !== "1") return
  process.stderr.write(encode(kind) + "\n")
}

export function detect(text: string): Kind | undefined {
  if (text.includes(encode("context_window"))) return "context_window"
  if (text.includes(encode("max_iteration"))) return "max_iteration"
  return undefined
}

/** Context overflow wins when the forced final max-step call also overflows. */
export function prefer(current: Kind | undefined, incoming: Kind | undefined): Kind | undefined {
  if (!current) return incoming
  if (!incoming) return current
  if (current === "context_window" || incoming === "context_window") return "context_window"
  return "max_iteration"
}

export function toGymError(exitCode: number, kind?: Kind): string | null {
  if (kind === "max_iteration") return "maximum iteration reached"
  if (kind === "context_window") return "context window exceeded"
  return exitCode === 0 ? null : `opencode_exit_${exitCode}`
}

export function shouldExitSuccessfully(exitCode: number, kind?: Kind): boolean {
  return exitCode === 0 || kind !== undefined
}
