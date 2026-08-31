/**
 * Model-patch capture for the bench driver.
 *
 * Two modes, selected by `--patch-mode` (gym passes it through the
 * `PATCH_MODE` env var in run_infer.sh):
 *
 *   - `worktree` (DEFAULT, the historical behaviour): mark untracked files
 *     intent-to-add and take `git diff` of the working tree. Correct for the
 *     SWE-bench-style prompts that explicitly tell the agent *not* to commit.
 *
 *   - `committed`: ignore the working tree and extract what the agent
 *     COMMITTED. Required by task families whose problem statement ends with
 *     "work on this in a new branch from main and commit everything when you
 *     are done" (e.g. the DeepSWE set). There, the agent commits its solution,
 *     leaving a clean tree — `git diff` returns "" and every rollout would be
 *     recorded as a 0-byte patch.
 *
 * In `committed` mode we diff `baseline..tip`, where `baseline` is the HEAD sha
 * captured *before* the agent starts (after bootstrap/deep-reset, so it is the
 * dataset's base commit) and `tip` is the most-advanced commit the agent left
 * behind. `tip` is searched across HEAD *and* every local branch, because the
 * agent may commit on a side branch and then switch back to main — HEAD alone
 * would silently yield an empty patch.
 *
 * The resulting patch is still a plain `baseline -> final tree` unified diff,
 * so the eval side (`git reset --hard <base_commit>` + `git apply`) is
 * unchanged regardless of mode.
 */

import { spawn } from "node:child_process"
import { existsSync } from "node:fs"

export type PatchMode = "worktree" | "committed"

export const PATCH_MODES: PatchMode[] = ["worktree", "committed"]

export const DEFAULT_PATCH_MODE: PatchMode = "worktree"

export function parsePatchMode(raw: string | undefined): PatchMode {
  if (!raw) return DEFAULT_PATCH_MODE
  const v = raw.trim().toLowerCase()
  if ((PATCH_MODES as string[]).includes(v)) return v as PatchMode
  throw new Error(`Invalid --patch-mode "${raw}" (expected one of: ${PATCH_MODES.join(", ")})`)
}

// git's canonical empty-tree object id. Used as the diff base when the repo has
// no commits at all (unborn HEAD), so a first commit still produces a patch.
const EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

// Same rationale as cli.ts/deep_reset.ts: some SIFs ENOENT on bare program
// names through Bun's posix_spawn, so resolve an absolute path up front.
function detectGit(): string {
  for (const p of ["/usr/bin/git", "/bin/git", "/usr/local/bin/git"]) {
    if (existsSync(p)) return p
  }
  return "git"
}

interface GitResult {
  stdout: string
  exitCode: number
}

function git(workspaceRoot: string, args: string[]): Promise<GitResult> {
  const gitPath = detectGit()
  return new Promise((resolve) => {
    const child = spawn(gitPath, ["-C", workspaceRoot, ...args], {
      env: { ...process.env, GIT_PAGER: "cat" },
      stdio: ["ignore", "pipe", "pipe"],
    })
    let stdout = ""
    child.stdout?.on("data", (b) => (stdout += b.toString("utf8")))
    // Swallow stderr: every call here is best-effort and a noisy `git` on a
    // half-broken repo must not pollute the gym log or fail the rollout.
    child.stderr?.on("data", () => {})
    child.on("close", (code) => resolve({ stdout, exitCode: code ?? 0 }))
    child.on("error", () => resolve({ stdout: "", exitCode: 1 }))
  })
}

/**
 * Snapshot HEAD before the agent runs. Returns "" when the repo has no commit
 * yet (unborn HEAD) or isn't a repo at all; `capturePatch` then falls back to
 * the empty tree.
 */
export async function recordBaselineCommit(workspaceRoot: string): Promise<string> {
  const res = await git(workspaceRoot, ["rev-parse", "--verify", "--quiet", "HEAD^{commit}"])
  return res.exitCode === 0 ? res.stdout.trim() : ""
}

/**
 * Give git a committer identity if the image doesn't ship one. Without this the
 * agent's very first `git commit` dies with "Author identity unknown" and the
 * whole rollout scores zero for a reason that has nothing to do with the model.
 * Repo-local (`--local`) so we don't mutate anything outside the workspace, and
 * only when unset, so a task-provided identity always wins.
 */
export async function ensureCommitIdentity(workspaceRoot: string): Promise<void> {
  for (const [key, value] of [
    ["user.email", "agent@opencode.local"],
    ["user.name", "opencode agent"],
  ]) {
    const existing = await git(workspaceRoot, ["config", "--get", key])
    if (existing.exitCode === 0 && existing.stdout.trim()) continue
    await git(workspaceRoot, ["config", "--local", key, value])
  }
}

async function worktreePatch(workspaceRoot: string): Promise<string> {
  // Mark untracked files as intent-to-add so newly-created files appear in
  // `git diff` without being committed. Plain `git diff` only shows changes
  // to tracked files, which silently drops new-file patches the agent wrote.
  await git(workspaceRoot, ["add", "-AN"])
  const res = await git(workspaceRoot, ["diff", "--binary"])
  return res.stdout
}

interface Candidate {
  /** commit sha of the ref tip */
  sha: string
  /** human label for logging: "HEAD" or the branch name */
  label: string
  /** commits reachable from `sha` but not from the baseline */
  ahead: number
}

/**
 * Every commit the agent could have left its work on: HEAD (covers detached
 * HEAD and "still on the branch it committed to") plus every local branch
 * (covers "committed on a side branch, then checked main back out").
 */
async function candidateTips(workspaceRoot: string, baseline: string): Promise<Candidate[]> {
  const tips: { sha: string; label: string }[] = []

  const head = await git(workspaceRoot, ["rev-parse", "--verify", "--quiet", "HEAD^{commit}"])
  if (head.exitCode === 0 && head.stdout.trim()) tips.push({ sha: head.stdout.trim(), label: "HEAD" })

  const branches = await git(workspaceRoot, ["for-each-ref", "--format=%(objectname) %(refname:short)", "refs/heads"])
  for (const line of branches.stdout.split("\n")) {
    const [sha, ...rest] = line.trim().split(" ")
    if (!sha) continue
    if (tips.some((t) => t.sha === sha)) continue // HEAD already covers this tip
    tips.push({ sha, label: rest.join(" ") || sha.slice(0, 8) })
  }

  const out: Candidate[] = []
  for (const tip of tips) {
    // `baseline..tip` counts commits reachable from tip but not from baseline.
    // Deliberately NOT gated on `merge-base --is-ancestor`: an agent that
    // amended or rebased its work leaves a tip that no longer descends from
    // baseline, and we still want that work.
    const res = await git(workspaceRoot, ["rev-list", "--count", `${baseline}..${tip.sha}`])
    const ahead = res.exitCode === 0 ? parseInt(res.stdout.trim(), 10) : 0
    out.push({ ...tip, ahead: Number.isFinite(ahead) ? ahead : 0 })
  }
  return out
}

/** Count of dirty/untracked paths, reported so a dropped worktree is visible in the log. */
async function dirtyPathCount(workspaceRoot: string): Promise<number> {
  const res = await git(workspaceRoot, ["status", "--porcelain", "--untracked-files=all"])
  return res.stdout.split("\n").filter((l) => l.trim()).length
}

async function committedPatch(workspaceRoot: string, baselineCommit: string): Promise<string> {
  const baseline = baselineCommit || EMPTY_TREE
  if (!baselineCommit) {
    console.log(`[bench] patch_mode=committed: no pre-run HEAD; diffing against the empty tree`)
  }

  const candidates = await candidateTips(workspaceRoot, baseline)
  // Most commits past the baseline wins. `candidateTips` puts HEAD first, and
  // Array.prototype.sort is stable in every JS engine we ship on, so HEAD wins
  // ties against a side branch holding the identical work.
  const ranked = candidates.filter((c) => c.ahead > 0).sort((a, b) => b.ahead - a.ahead)
  const dirty = await dirtyPathCount(workspaceRoot)

  if (ranked.length === 0) {
    console.log(
      `[bench] patch_mode=committed: agent left no commit past baseline ${baseline.slice(0, 8)} ` +
        `(refs checked: ${candidates.length}, uncommitted paths: ${dirty}) -> empty patch`,
    )
    return ""
  }

  const chosen = ranked[0]!
  const others = ranked
    .slice(1)
    .map((c) => `${c.label}+${c.ahead}`)
    .join(",")
  console.log(
    `[bench] patch_mode=committed: baseline=${baseline.slice(0, 8)} tip=${chosen.label}@${chosen.sha.slice(0, 8)} ` +
      `commits=${chosen.ahead} uncommitted_paths=${dirty}${others ? ` other_refs=[${others}]` : ""}`,
  )
  if (dirty > 0) {
    console.log(
      `[bench] patch_mode=committed: ${dirty} uncommitted path(s) are NOT in the patch ` +
        `(the task asked the agent to commit its work)`,
    )
  }

  const res = await git(workspaceRoot, ["diff", "--binary", baseline, chosen.sha])
  return res.stdout
}

/**
 * Produce the model patch for `mode`.
 *
 * @param baselineCommit HEAD sha captured before the agent ran; only used by
 *                       `committed` mode.
 */
export async function capturePatch(
  workspaceRoot: string,
  mode: PatchMode,
  baselineCommit: string = "",
): Promise<string> {
  return mode === "committed" ? committedPatch(workspaceRoot, baselineCommit) : worktreePatch(workspaceRoot)
}
