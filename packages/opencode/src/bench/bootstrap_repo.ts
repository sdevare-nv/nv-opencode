/**
 * Bootstrap a git repository inside the workspace when the SIF ships a flat
 * source tree without a `.git` directory.
 *
 * Some dataset SIFs (notably `swe-bench-ext`, and certain SWE-rebench variants)
 * copy the repo contents into `/workspace/repo` (or the dataset-specific path)
 * without preserving git history. Without `.git`, `runDeepReset` is a silent
 * no-op (its `git rev-parse` fails under the outer `|| true`) and
 * `captureGitDiff` returns "" — every rollout is recorded as `patch=0 bytes`
 * regardless of what the agent did. Port of nv-OpenHands'
 * `evaluation/benchmarks/swe_bench/run_infer.py:1142-1156`.
 *
 * If `.git` already exists, this is a no-op. Otherwise a pristine baseline
 * commit is created and tagged `opencode_bench_baseline`. Callers should skip
 * `runDeepReset` when this returns `{ freshInit: true }` — the dataset's
 * upstream `base_commit` SHA does not exist in the fresh repo, so deep_reset
 * would just fail rev-parse and noisily fall through to its nuclear pass.
 */

import { spawn } from "node:child_process"
import { existsSync } from "node:fs"
import path from "node:path"

function detectShell(): string | null {
  for (const p of ["/bin/bash", "/usr/bin/bash", "/bin/sh", "/usr/bin/sh"]) {
    if (existsSync(p)) return p
  }
  return null
}

function shellQuote(s: string): string {
  return `'${s.replace(/'/g, `'\\''`)}'`
}

function buildBootstrapCmd(workspaceRoot: string): string {
  const q = shellQuote(workspaceRoot)
  return (
    `cd ${q} && ` +
    `echo "[bootstrap_repo] initializing git repo at ${workspaceRoot}" && ` +
    `git config --global --add safe.directory ${q} && ` +
    `git init -q && ` +
    `git config user.email 'bench@opencode.local' && ` +
    `git config user.name 'opencode bench' && ` +
    `git add -A && ` +
    `git commit -q --allow-empty -m 'opencode bench baseline' && ` +
    `git tag -f opencode_bench_baseline HEAD && ` +
    `echo "[bootstrap_repo] done; HEAD=$(git rev-parse --short HEAD)"`
  )
}

export interface BootstrapResult {
  freshInit: boolean
}

export async function bootstrapRepoIfMissing(workspaceRoot: string): Promise<BootstrapResult> {
  if (existsSync(path.join(workspaceRoot, ".git"))) {
    return { freshInit: false }
  }
  const shell = detectShell()
  if (!shell) {
    console.warn(`[bench] bootstrap_repo skipped: no shell found at /bin/{bash,sh} or /usr/bin/{bash,sh}`)
    return { freshInit: false }
  }
  const cmd = buildBootstrapCmd(workspaceRoot)
  console.log(`[bench] bootstrap_repo workspace=${workspaceRoot} shell=${shell}`)
  const exitCode = await new Promise<number>((resolve) => {
    const child = spawn(shell, ["-c", cmd], {
      stdio: ["ignore", "inherit", "inherit"],
    })
    child.on("close", (code) => resolve(code ?? 0))
    child.on("error", (err) => {
      console.warn(`[bench] bootstrap_repo spawn error: ${err}`)
      resolve(1)
    })
  })
  console.log(`[bench] bootstrap_repo exit=${exitCode}`)
  return { freshInit: exitCode === 0 }
}
