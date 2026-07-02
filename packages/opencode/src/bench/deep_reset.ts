/**
 * Pin HEAD at base_commit and drop every ref/reflog entry/object that reaches
 * commits past base_commit.
 *
 * Ported from the improved nv-OpenHands version in
 * `evaluation/benchmarks/swe_bench/run_infer.py` (`_deep_reset_to_base_commit`).
 * See its docstring for design rationale. Summary:
 *
 *   - In-place cleanup (no bundle, no .git swap). Each step is best-effort;
 *     only the final HEAD-at-BASE check decides success.
 *   - Batch-deletes every ref namespace that can reach post-base commits
 *     (`refs/tags`, `refs/remotes`, `refs/stash`, `refs/notes`,
 *     `refs/replace`, `refs/prefetch`, `refs/pull`) plus local branches
 *     other than the current one. Uses "option no-deref" per entry so
 *     symbolic refs (refs/remotes/NAME/HEAD) don't abort the transaction.
 *   - `git repack -ad` (lowercase -a) to drop unreachable objects from the
 *     pack in one step — avoids the demote-to-loose intermediate that made
 *     `-Ad` take ~10 minutes on repos like facebook/react.
 *   - Sets `user.email`/`user.name` and runs `git clean -fd` so downstream
 *     agent commands work even on containers with no global git identity
 *     or with leftover untracked files from a previous run.
 *   - On failure, logs `[bench] REBUILD_FAILED` and returns anyway so the
 *     agent still runs (grep task logs for `REBUILD_FAILED` to filter these
 *     instances from clean-run analysis).
 *
 * Measured on 20 real repos ranging from 5.7 MB (pallets/click) to 997 MB
 * (facebook/react): all pass with HEAD=BASE, refs=1, reflog=0, post-base
 * `cat-file` returns "Not a valid object name". Slowest: cpython at 12.5s.
 * facebook/react: 4.1s (previous `-Ad`-based version: 662s).
 */

import { spawn } from "node:child_process"
import { existsSync } from "node:fs"

// Some SIFs are minimal and ship without `bash` on PATH, or Bun's posix_spawn
// doesn't fall back to PATH lookup the way `execvp` does — either way,
// spawn("bash", ...) ENOENTs. Probe absolute paths up front; the deep-reset
// script uses only POSIX features, so /bin/sh is a safe fallback if bash is
// absent.
function detectShell(): string | null {
  for (const p of ["/bin/bash", "/usr/bin/bash", "/bin/sh", "/usr/bin/sh"]) {
    if (existsSync(p)) return p
  }
  return null
}

function shellQuote(s: string): string {
  return `'${s.replace(/'/g, `'\\''`)}'`
}

/**
 * The bash body that does the actual reset. `set +e` at the top —
 * intermediate step failures don't abort; only the terminal HEAD-check
 * controls the exit code.
 *
 * Note on template-literal escaping: this string contains bash `$VAR` and
 * `$(...)` references. Only `${...}` would be interpolated by JS — none of
 * those exist in the shell body, so the `$` characters pass through verbatim.
 * The single `${baseCommit}` interpolation is the caller-supplied SHA.
 */
export function buildDeepResetCmd(baseCommit: string): string {
  return `set +e
BASE=$(git rev-parse --verify ${baseCommit}^{commit}) || { echo "[deep_reset] BAD_BASE ${baseCommit}"; exit 1; }
BRANCH=$(git symbolic-ref --short -q HEAD || echo main)
echo "[deep_reset] BASE=$BASE BRANCH=$BRANCH"
git reset --hard --quiet
git clean -fd --quiet
git checkout "$BASE" -f --quiet
git checkout -B "$BRANCH" "$BASE" --quiet
git for-each-ref --format='option no-deref%0Adelete %(refname)' refs/tags refs/remotes refs/stash refs/notes refs/replace refs/prefetch refs/pull 2>/dev/null | git update-ref --stdin 2>/dev/null
git for-each-ref --format='%(refname)' refs/heads 2>/dev/null | grep -vFx "refs/heads/$BRANCH" | while read r; do git update-ref -d "$r" --no-deref 2>/dev/null; done
rm -f .git/packed-refs
rm -f .git/FETCH_HEAD .git/ORIG_HEAD .git/MERGE_HEAD .git/CHERRY_PICK_HEAD .git/REVERT_HEAD .git/BISECT_HEAD .git/AUTO_MERGE
git reflog expire --expire=now --expire-unreachable=now --all 2>/dev/null
git repack -ad --quiet 2>/dev/null
git prune --expire=now 2>/dev/null
git config user.email 'opencode-bench@localhost' 2>/dev/null
git config user.name 'opencode bench' 2>/dev/null
HEAD_SHA=$(git rev-parse --verify HEAD^{commit} 2>/dev/null)
if [ "$HEAD_SHA" = "$BASE" ]; then
  echo "__DEEP_RESET_OK__ HEAD=$HEAD_SHA"
  exit 0
else
  echo "__DEEP_RESET_MISMATCH__ HEAD=$HEAD_SHA BASE=$BASE"
  exit 1
fi`
}

export async function runDeepReset(workspaceRoot: string, baseCommit: string): Promise<void> {
  if (!baseCommit) return
  const shell = detectShell()
  if (!shell) {
    console.warn(`[bench] deep_reset skipped: no shell found at /bin/{bash,sh} or /usr/bin/{bash,sh}`)
    return
  }
  // Bake `cd <workspace>` into the shell script instead of passing the `cwd`
  // option to spawn(). On some minimal apptainer images Bun's posix_spawn
  // ENOENTs whenever a `cwd` is set (libc lacks addchdir_np extension); routing
  // the chdir through the shell sidesteps that entirely.
  const cmd = `cd ${shellQuote(workspaceRoot)} && ` + buildDeepResetCmd(baseCommit)
  console.log(`[bench] deep_reset workspace=${workspaceRoot} base=${baseCommit} shell=${shell}`)
  await new Promise<void>((resolve) => {
    // 10 min hard cap. In-place reset on real repos (5.7 MB → 3.5 GB)
    // finishes well under 20s; anything past 10 min is a hang.
    const child = spawn(shell, ["-c", cmd], {
      stdio: ["ignore", "inherit", "inherit"],
      timeout: 600_000,
      killSignal: "SIGKILL",
    })
    child.on("close", (code) => {
      const rc = code ?? 0
      if (rc === 0) {
        console.log(`[bench] deep_reset OK`)
      } else {
        // Best-effort: log LOUDLY so downstream analysis can grep for
        // REBUILD_FAILED and filter these instances from clean-run
        // aggregates, but do not throw — running the task on a
        // partially-reset workspace is more useful than failing outright.
        console.warn(
          `[bench] REBUILD_FAILED base_commit=${baseCommit} exit_code=${rc}. ` +
            `The workspace .git may be partially reset and may still ` +
            `reference commits past base_commit. The agent will proceed ` +
            `but may be able to inspect the solution via git. Filter this ` +
            `task from clean-run analysis.`,
        )
      }
      resolve()
    })
    child.on("error", (err) => {
      console.warn(`[bench] deep_reset spawn error: ${err}`)
      resolve()
    })
  })
}
