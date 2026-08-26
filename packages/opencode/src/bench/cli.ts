/**
 * SWE-bench bench CLI driver.
 *
 * Drives a single SWE-bench instance to completion using opencode's REAL
 * agentic loop. We spawn `bun .../src/index.ts run` as a subprocess (with a
 * per-instance opencode config that registers our `nemo-gym` provider, a
 * SWE-bench agent, and disables compaction) and let it run to idle.
 *
 * Why subprocess instead of in-process Server.Default? Subprocess is the
 * model the user-facing `opencode run` already uses (cli/cmd/run.ts:670–675
 * also uses an in-process fetch but the public entry is `bun .../index.ts`).
 * A subprocess gives us:
 *   - clean process isolation per instance (matters for many parallel SIFs)
 *   - identical bootstrapping path to `opencode run`, so we don't drift
 *   - a compact event-type stream on stdout
 *
 * Trajectory capture: the nemo-gym provider (registered via this config)
 * writes `<completionsDir>/<turn>.json` per LLM call BEFORE returning. On
 * exit we capture `git diff` and write `output.jsonl`.
 */

import { existsSync, promises as fs, readFileSync } from "node:fs"
import path from "node:path"
import os from "node:os"
import { spawn } from "node:child_process"
import { runDeepReset } from "./deep_reset"
import { bootstrapRepoIfMissing } from "./bootstrap_repo"
import * as BenchTerminalError from "./terminal_error"
// opencode's built-in anthropic system prompt — Bun bundles .txt as a string.
// Used as the default when no --system-prompt override is passed.
import PROMPT_ANTHROPIC from "../session/prompt/anthropic.txt"

interface CliArgs {
  instanceDictPath: string
  outputDir: string
  config: string
  maxTurns: number
  agentCls: string
  dataset: string
  split: string
  selectedId: string
  /** Resolved repo path inside the SIF — gym side decided based on dataset_name. */
  workspaceRoot: string
  /** Pre-rendered user message file (workspace_path baked in by gym). */
  userMessageFile: string
  systemPromptPath?: string
  /** Enable opencode's `task` tool (spawns subagent sessions). */
  enableSubagents: boolean
}

function parseArgs(argv: string[]): CliArgs {
  const out: Partial<CliArgs> = {
    maxTurns: 100,
    agentCls: "OpenCodeAgent",
    dataset: "",
    split: "test",
    enableSubagents: false,
  }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    const next = () => argv[++i]
    switch (a) {
      case "--instance-dict-path":
        out.instanceDictPath = next()
        break
      case "--output-dir":
        out.outputDir = next()
        break
      case "--config":
        out.config = next()
        break
      case "--max-turns":
        out.maxTurns = parseInt(next(), 10)
        break
      case "--agent-cls":
        out.agentCls = next()
        break
      case "--dataset":
        out.dataset = next()
        break
      case "--split":
        out.split = next()
        break
      case "--selected-id":
        out.selectedId = next()
        break
      case "--workspace-root":
        out.workspaceRoot = next()
        break
      case "--user-message-file":
        out.userMessageFile = next()
        break
      case "--system-prompt":
        out.systemPromptPath = next()
        break
      case "--enable-subagents":
        out.enableSubagents = true
        break
      default:
        if (a.startsWith("--")) throw new Error(`Unknown flag: ${a}`)
    }
  }
  for (const required of [
    "instanceDictPath",
    "outputDir",
    "config",
    "selectedId",
    "workspaceRoot",
    "userMessageFile",
  ] as const) {
    if (!out[required])
      throw new Error(`Missing required arg --${required.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase())}`)
  }
  return out as CliArgs
}

interface InstanceDict {
  instance_id: string
  problem_statement: string
  repo?: string
  repo_name?: string
  workspace?: string
  base_commit?: string
  [key: string]: unknown
}

async function readInstance(instanceDictPath: string, selectedId: string): Promise<InstanceDict> {
  const text = await fs.readFile(instanceDictPath, "utf8")
  const lines = text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
  const records = lines.map((l) => JSON.parse(l) as InstanceDict)
  const match = records.find((r) => r.instance_id === selectedId) ?? records[0]
  if (!match) throw new Error(`No instance found in ${instanceDictPath}`)
  return match
}

function loadGymConfig(configPath: string): Record<string, unknown> {
  return JSON.parse(readFileSync(configPath, "utf8"))
}

const DEFAULT_SYSTEM_PROMPT = PROMPT_ANTHROPIC

async function buildConfigDir(args: {
  instanceId: string
  modelName: string
  baseURL: string
  completionsDir: string
  maxTurns: number
  systemPromptPath?: string
  enableSubagents: boolean
  /** Forced sampling params (RL on-policy requirement); from gym llm.model config. */
  temperature?: number
  topP?: number
  maxTokens?: number
}): Promise<{ tmpRoot: string; configFile: string }> {
  const tmpRoot = await fs.mkdtemp(path.join(os.tmpdir(), `bench-${args.instanceId}-`))
  await fs.mkdir(tmpRoot, { recursive: true })

  const systemPrompt = args.systemPromptPath ? await fs.readFile(args.systemPromptPath, "utf8") : DEFAULT_SYSTEM_PROMPT

  const cfg: Record<string, unknown> = {
    $schema: "https://opencode.ai/config.json",
    provider: {
      "nemo-gym": {
        npm: "@opencode-ai/nemo-gym",
        options: {
          baseURL: args.baseURL,
          completionsDir: args.completionsDir,
          instanceId: args.instanceId,
          // Unlimited: model-server backpressure can stall a request for
          // tens of minutes at high shard concurrency; we'd rather wait
          // than tear down the session and produce an empty model_patch.
          // requestTimeoutMs<=0 disables the abort timer in the provider.
          retries: Number.MAX_SAFE_INTEGER,
          requestTimeoutMs: 0,
          // Forced sampling params: NeMo-RL's vLLM worker asserts every
          // request's temperature/top_p match the training generation config
          // exactly (on-policy). Passed by gym via the llm.model config block.
          ...(args.temperature !== undefined ? { temperature: args.temperature } : {}),
          ...(args.topP !== undefined ? { topP: args.topP } : {}),
          ...(args.maxTokens !== undefined ? { maxTokens: args.maxTokens } : {}),
        },
        models: {
          [args.modelName]: {
            id: args.modelName,
            name: args.modelName,
            limit: { context: 131072, output: 32768 },
            tool_call: true,
            temperature: true,
          },
        },
      },
    },
    agent: {
      title: {
        disable: true,
      },
      "swe-bench": {
        mode: "primary",
        model: `nemo-gym/${args.modelName}`,
        prompt: systemPrompt,
        // Allow the read+write tool set; disable web/skill/task to keep the
        // agent focused on local code editing.
        permission: {
          edit: { "**": "allow" },
          bash: {
            "*": "allow",

            // process termination
            "*killall*": "deny",
            "*pkill*": "deny",
            "*kill -1*": "deny",
            "*kill 0*": "deny",

            // filesystem destruction
            "*rm -rf /": "deny",
            "*rm -rf /*": "deny",
            "*rm -rf /bin*": "deny",
            "*rm -rf /usr*": "deny",
            "*rm -rf /etc*": "deny",
            "*rm -rf /var*": "deny",
            "*rm -rf /home*": "deny",
            "*rm -rf /root*": "deny",
            "*rm -rf /opt*": "deny",
            "*rm -rf /lib*": "deny",
            "*rm -rf /lib64*": "deny",
            "*rm -rf /sbin*": "deny",
            "*rm -rf /boot*": "deny",
            "*rm -rf /dev*": "deny",
            "*rm -rf /proc*": "deny",
            "*rm -rf /sys*": "deny",

            // system control
            // "*shutdown*": "deny",
            // "*reboot*": "deny",
            // "*poweroff*": "deny",
            // "*halt*": "deny",
            // "init 0*": "deny",
            // "init 6*": "deny",

            // disk devices
            "dd *of=/dev/sd*": "deny",
            "dd *of=/dev/nvme*": "deny",
            "dd *of=/dev/hd*": "deny",
            "dd *of=/dev/null*": "deny",

            // git network
            "*git fetch*": "deny",
            "*git pull*": "deny",
            "*git clone*": "deny",
            "*git ls-remote*": "deny",
            "*git remote add*": "deny",
            "*git remote set-url*": "deny",
            "*git remote set-head*": "deny",
            "*git remote update*": "deny",
            "*git remote rename*": "deny",
            "*git remote set-branches*": "deny",
            "*git submodule add*": "deny",
            "*git submodule update*": "deny",
            "*git submodule sync*": "deny",
            "*git submodule init*": "deny",
            "*git archive*--remote*": "deny",
            "*git *://*": "deny",
            "*git *@*:*": "deny",

            // git history mining
            "*git log*--all*": "deny",
            "*git log*--branches*": "deny",
            "*git log*--remotes*": "deny",
            "*git log*--walk-reflogs*": "deny",
            "*git log*--grep*": "deny",
            "*git rev-list*--all*": "deny",
            "*git rev-list*--branches*": "deny",
            "*git rev-list*--remotes*": "deny",
            "*git rev-list*--grep*": "deny",
            "*git shortlog*--all*": "deny",
            "*git reflog*": "deny",
            "*git cat-file*": "deny",
            "*git fsck*": "deny",
            "*git verify-pack*": "deny",
            "*git unpack-objects*": "deny",
            "*git cherry*": "deny",
            "*git show*": "deny",
            "*git merge-base*--is-ancestor*": "deny",
            "*git branch*--contains*": "deny",
            "*git tag*--contains*": "deny",
            "*git for-each-ref*--contains*": "deny",

            // git internals (substring match on path)
            "*.git/logs*": "deny",
            "*.git/packed-refs*": "deny",
            "*.git/ORIG_HEAD*": "deny",
            "*.git/FETCH_HEAD*": "deny",
            "*.git/refs*": "deny",

            // online lookups
            "*curl *github.com*": "deny",
            "*wget *github.com*": "deny",
            "*curl *githubusercontent.com*": "deny",
            "*wget *githubusercontent.com*": "deny",
            "*curl *github.io*": "deny",
            "*wget *github.io*": "deny",
          },
        },
        tools: {
          bash: true,
          edit: true,
          read: true,
          glob: true,
          grep: true,
          write: true,
          apply_patch: true,
          webfetch: false,
          websearch: false,
          task: args.enableSubagents,
          skill: false,
          todowrite: true,
        },
        steps: args.maxTurns,
        options: {},
      },
    },
    compaction: { auto: false },
    share: "manual",
  }

  const configFile = path.join(tmpRoot, "opencode.jsonc")
  await fs.writeFile(configFile, JSON.stringify(cfg, null, 2))

  return { tmpRoot, configFile }
}

// Some SIFs ship with bare PATH lookups that ENOENT on bare program names
// through Bun's posix_spawn. Resolve to an absolute path up front for any
// binary we shell out to.
function detectBin(candidates: string[]): string | null {
  for (const p of candidates) {
    if (existsSync(p)) return p
  }
  return null
}

function runOpencode(args: {
  workspaceRoot: string
  modelName: string
  message: string
  env: NodeJS.ProcessEnv
  opencodeBin: string
  agent: string
}): Promise<{ exitCode: number; stdout: string; stderr: string; terminalError?: BenchTerminalError.Kind }> {
  // Use the same bun binary that's currently running — guaranteed to exist
  // and avoids PATH lookup quirks under Bun's posix_spawn.
  const bunPath = process.execPath
  return new Promise((resolve) => {
    // Don't set spawn's `cwd` — Bun's posix_spawn on some minimal apptainer
    // images ENOENTs whenever cwd is set (libc lacks addchdir_np). Opencode's
    // own `--dir <workspaceRoot>` flag changes the working directory
    // internally, so we don't need spawn-level cwd.
    const child = spawn(
      bunPath,
      [
        args.opencodeBin,
        "run",
        args.message,
        "--agent",
        args.agent,
        "--model",
        `nemo-gym/${args.modelName}`,
        "--format",
        "json",
        "--dir",
        args.workspaceRoot,
      ],
      {
        env: args.env,
        stdio: ["ignore", "pipe", "pipe"],
      },
    )
    let stdout = ""
    let stderr = ""
    let terminalError: BenchTerminalError.Kind | undefined
    let terminalSignalBuffer = ""
    const observeTerminalSignal = (chunk: string) => {
      // Retain enough overlap to recognize a marker split across pipe chunks.
      terminalSignalBuffer = (terminalSignalBuffer + chunk).slice(-256)
      terminalError = BenchTerminalError.prefer(terminalError, BenchTerminalError.detect(terminalSignalBuffer))
    }
    const MAX_KEEP = 256 * 1024 // keep only a bounded tail for error reporting
    let lineBuf = ""
    child.stdout?.on("data", (b) => {
      const chunk = b.toString("utf8")
      observeTerminalSignal(chunk)
      lineBuf += chunk
      let idx: number
      while ((idx = lineBuf.indexOf("\n")) >= 0) {
        const line = lineBuf.slice(0, idx)
        lineBuf = lineBuf.slice(idx + 1)
        // Forward the event type so the gym log captures progress cheaply.
        process.stdout.write(line + "\n")
        stdout = (stdout + line + "\n").slice(-MAX_KEEP)
      }
    })
    child.stderr?.on("data", (b) => {
      const chunk = b.toString("utf8")
      observeTerminalSignal(chunk)
      stderr = (stderr + chunk).slice(-MAX_KEEP)
      process.stderr.write(chunk)
    })
    child.on("close", (code) => resolve({ exitCode: code ?? 0, stdout, stderr, terminalError }))
    child.on("error", (err) => {
      stderr += String(err)
      resolve({ exitCode: 999, stdout, stderr, terminalError })
    })
  })
}

async function captureGitDiff(workspaceRoot: string): Promise<string> {
  const gitPath = detectBin(["/usr/bin/git", "/bin/git", "/usr/local/bin/git"]) ?? "git"
  const runGit = (args: string[], capture: boolean): Promise<string> =>
    new Promise((resolve) => {
      const child = spawn(gitPath, ["-C", workspaceRoot, ...args], {
        env: { ...process.env, GIT_PAGER: "cat" },
      })
      let stdout = ""
      if (capture) child.stdout?.on("data", (b) => (stdout += b.toString("utf8")))
      child.on("close", () => resolve(stdout))
      child.on("error", () => resolve(""))
    })
  // Mark untracked files as intent-to-add so newly-created files appear in
  // `git diff` without being committed. Plain `git diff` only shows changes
  // to tracked files, which silently drops new-file patches the agent wrote.
  await runGit(["add", "-AN"], false)
  return runGit(["diff", "--binary"], true)
}

interface OutputJsonl {
  instance_id: string
  test_result: { git_patch: string }
  metadata: { llm_config: { model: string } }
  metrics: Record<string, unknown>
  error: string | null
}

async function writeOutputJsonl(evalOutputDir: string, instanceId: string, payload: OutputJsonl): Promise<string> {
  const runDir = path.join(evalOutputDir, instanceId, "bench_run")
  await fs.mkdir(runDir, { recursive: true })
  const outPath = path.join(runDir, "output.jsonl")
  const tmp = `${outPath}.tmp`
  await fs.writeFile(tmp, JSON.stringify(payload) + "\n")
  await fs.rename(tmp, outPath)
  return outPath
}

function completionsDirFor(evalOutputDir: string, instanceId: string): string {
  // Match openhands' on-host glob: <eval_dir>/*/*/*/llm_completions/<instance_id>/*.json
  return path.join(evalOutputDir, instanceId, "bench_run", "llm_completions", instanceId)
}

function detectOpencodeBin(): string {
  // Prefer the pre-bundled artifact at <opencode_root>/.bench-build/opencode.js.
  // Running un-bundled `src/index.ts` triggers cascading runtime resolution
  // failures (TUI JSX runtime not honored, @anthropic-ai/sdk relative .mjs
  // paths failing across the isolated install layout). The bundle inlines
  // every transitive dep and is opencode's intended deployment shape.
  // Falls back to src/index.ts only for dev / when setup_scripts/opencode.sh
  // hasn't run.
  const here = path.dirname(new URL(import.meta.url).pathname)
  // bench/cli.ts → packages/opencode/src/bench → packages/opencode/src → packages/opencode → packages → <root>
  const opencodeRoot = path.resolve(here, "..", "..", "..", "..")
  const bundled = path.resolve(opencodeRoot, ".bench-build", "opencode.js")
  if (existsSync(bundled)) return bundled
  return path.resolve(here, "..", "index.ts")
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  const instance = await readInstance(args.instanceDictPath, args.selectedId)
  // workspaceRoot is decided gym-side based on dataset_name; we use it verbatim.
  const workspaceRoot = args.workspaceRoot
  const gymConfig = loadGymConfig(args.config)
  const llmModelCfg = ((gymConfig as Record<string, Record<string, unknown>>).llm?.model ?? {}) as Record<
    string,
    unknown
  >
  const modelName = String(llmModelCfg.model ?? "unknown-model")
  // Forced sampling params from gym (RL training on-policy requirement).
  const forcedTemperature = typeof llmModelCfg.temperature === "number" ? llmModelCfg.temperature : undefined
  const forcedTopP = typeof llmModelCfg.top_p === "number" ? llmModelCfg.top_p : undefined
  // Optional: force a max_tokens cap; when absent, requests carry none (unlimited).
  const forcedMaxTokens = typeof llmModelCfg.max_tokens === "number" ? llmModelCfg.max_tokens : undefined
  const baseURL = process.env.NEMO_GYM_MODEL_SERVER_BASE_URL
  if (!baseURL) throw new Error("NEMO_GYM_MODEL_SERVER_BASE_URL not set in env (gym harness sets this).")

  const completionsDir = completionsDirFor(args.outputDir, instance.instance_id)
  await fs.mkdir(completionsDir, { recursive: true })

  // The user message is fully rendered by gym (workspace_path baked in based
  // on dataset_name); we just read it as-is and pass it to opencode.
  const userPrompt = await fs.readFile(args.userMessageFile, "utf8")

  const { tmpRoot, configFile } = await buildConfigDir({
    instanceId: instance.instance_id,
    modelName,
    baseURL,
    completionsDir,
    maxTurns: args.maxTurns,
    systemPromptPath: args.systemPromptPath,
    enableSubagents: args.enableSubagents,
    temperature: forcedTemperature,
    topP: forcedTopP,
    maxTokens: forcedMaxTokens,
  })

  const startedAt = Date.now()
  const childEnv: NodeJS.ProcessEnv = {
    ...process.env,
    // Run-isolated opencode state.
    OPENCODE_DB: ":memory:",
    OPENCODE_DATA: path.join(tmpRoot, "data"),
    OPENCODE_CONFIG: configFile,
    // The benchmark already runs inside a SIF sandbox, so make that the
    // security boundary. This final config override applies to subagents too.
    OPENCODE_PERMISSION: JSON.stringify({ "*": "allow" }),

    // Disable opencode's built-in plugin loaders; the bench harness doesn't need them.
    OPENCODE_PURE: "1",
    // Skip the dynamic env block (working dir + Today's date) in the system
    // prompt — keeps the RL prompt-token prefix invariant stable across turns
    // (a midnight rollover would otherwise shift `Today's date: ...`).
    OPENCODE_DISABLE_ENV_PROMPT: "1",
    // Have all agent sessions report terminal states to this bench wrapper.
    // This is bench-only and does not alter normal opencode runs.
    [BenchTerminalError.ENV]: "1",
    // Avoid serializing and piping full event payloads into the gym log.
    OPENCODE_BENCH_EVENT_TYPES_ONLY: "1",
  }

  // Bootstrap a git repo if the SIF shipped a flat source tree (swe-bench-ext
  // and some SWE-rebench variants). Without this, captureGitDiff returns ""
  // and every patch is recorded as 0 bytes.
  const { freshInit } = await bootstrapRepoIfMissing(workspaceRoot)

  // Prune git history past base_commit so the agent can't reach future commits.
  // Skip when we just freshly initialized: the dataset's upstream base_commit
  // SHA doesn't exist in our local repo, so deep_reset would just fail
  // rev-parse and fall through to its nuclear pass. The fresh `HEAD` is
  // already the correct baseline (also tagged `opencode_bench_baseline`).
  if (!freshInit) {
    await runDeepReset(workspaceRoot, String(instance.base_commit ?? ""))
  }

  const opencodeBin = detectOpencodeBin()
  const result = await runOpencode({
    workspaceRoot,
    modelName,
    message: userPrompt,
    env: childEnv,
    opencodeBin,
    agent: "swe-bench",
  })

  const patch = await captureGitDiff(workspaceRoot)
  const benchRunTime = (Date.now() - startedAt) / 1000

  const error = BenchTerminalError.toGymError(result.exitCode, result.terminalError)
  const outPath = await writeOutputJsonl(args.outputDir, instance.instance_id, {
    instance_id: instance.instance_id,
    test_result: { git_patch: patch },
    metadata: { llm_config: { model: modelName } },
    metrics: {
      bench_run_time: benchRunTime,
      opencode_exit_code: result.exitCode,
    },
    error,
  })

  console.log(`[bench] wrote ${outPath} (patch=${patch.length} bytes, error=${error ?? "none"})`)

  // Mirror opencode's exit code explicitly. Falling off the end of main() and
  // letting Bun drain the event loop produced a flaky exit=1 even when the
  // bench wrote output.jsonl cleanly (sqlite migration handles, residual
  // child-stdio pipes from the opencode subprocess). Gym's runner treats any
  // non-zero apptainer exit as `Agent command failed` and discards the
  // already-written patch, so we MUST exit 0 deterministically on success.
  process.exit(BenchTerminalError.shouldExitSuccessfully(result.exitCode, result.terminalError) ? 0 : 1)
}

main().catch((err) => {
  console.error(`[bench] fatal: ${err?.stack ?? err}`)
  process.exit(2)
})
