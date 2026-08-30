import { $ } from "bun"
import { afterEach, describe, expect, test } from "bun:test"
import fs from "fs/promises"
import os from "os"
import path from "path"
import { capturePatch, ensureCommitIdentity, parsePatchMode, recordBaselineCommit } from "../../src/bench/patch"

// Deliberately NOT using ../fixture/fixture: src/bench/* is a standalone leaf
// (node:child_process + node:fs only) that runs inside minimal SIF images, and
// this suite should stay runnable without booting the instance/effect stack.
const created: string[] = []

afterEach(async () => {
  await Promise.all(created.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true }).catch(() => {})))
})

async function repo(withBaseline = true) {
  const dir = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), "bench-patch-")))
  created.push(dir)
  await $`git init -q -b main`.cwd(dir).quiet()
  await $`git config user.email bench@opencode.local`.cwd(dir).quiet()
  await $`git config user.name bench`.cwd(dir).quiet()
  await $`git config commit.gpgsign false`.cwd(dir).quiet()
  if (withBaseline) {
    await Bun.write(path.join(dir, "app.py"), "def solve():\n    return 0\n")
    await commit(dir, "baseline")
  }
  return dir
}

async function commit(dir: string, message: string) {
  await $`git add -A`.cwd(dir).quiet()
  await $`git commit -q -m ${message}`.cwd(dir).quiet()
}

describe("bench patch mode", () => {
  test("parsePatchMode defaults to worktree and rejects garbage", () => {
    expect(parsePatchMode(undefined)).toBe("worktree")
    expect(parsePatchMode("")).toBe("worktree")
    expect(parsePatchMode("Committed")).toBe("committed")
    expect(() => parsePatchMode("staged")).toThrow()
  })

  test("worktree mode captures uncommitted edits and new files", async () => {
    const dir = await repo()
    const baseline = await recordBaselineCommit(dir)

    await Bun.write(path.join(dir, "app.py"), "def solve():\n    return 1\n")
    await Bun.write(path.join(dir, "new_file.py"), "X = 1\n")

    const patch = await capturePatch(dir, "worktree", baseline)
    expect(patch).toContain("+    return 1")
    expect(patch).toContain("new_file.py")
  })

  test("worktree mode misses work the agent committed (the DeepSWE failure)", async () => {
    const dir = await repo()
    const baseline = await recordBaselineCommit(dir)

    await Bun.write(path.join(dir, "app.py"), "def solve():\n    return 1\n")
    await commit(dir, "fix")

    expect(await capturePatch(dir, "worktree", baseline)).toBe("")
  })

  test("committed mode captures a side-branch commit from either HEAD position", async () => {
    const dir = await repo()
    const baseline = await recordBaselineCommit(dir)

    await $`git checkout -q -b fix/solve`.cwd(dir).quiet()
    await Bun.write(path.join(dir, "app.py"), "def solve():\n    return 1\n")
    await Bun.write(path.join(dir, "new_file.py"), "X = 1\n")
    await commit(dir, "fix")

    const onBranch = await capturePatch(dir, "committed", baseline)
    expect(onBranch).toContain("+    return 1")
    expect(onBranch).toContain("new_file.py")

    // Agent switched back to main after committing: HEAD sits at the baseline,
    // so only the local-branch scan finds the work.
    await $`git checkout -q main`.cwd(dir).quiet()
    expect(await capturePatch(dir, "committed", baseline)).toBe(onBranch)
  })

  test("committed mode picks the ref with the most commits past baseline", async () => {
    const dir = await repo()
    const baseline = await recordBaselineCommit(dir)

    await $`git checkout -q -b scratch`.cwd(dir).quiet()
    await Bun.write(path.join(dir, "scratch.txt"), "debug\n")
    await commit(dir, "scratch")

    await $`git checkout -q -b fix/solve main`.cwd(dir).quiet()
    for (const n of [1, 2]) {
      await Bun.write(path.join(dir, "app.py"), `def solve():\n    return ${n}\n`)
      await commit(dir, `step${n}`)
    }
    await $`git checkout -q main`.cwd(dir).quiet()

    const patch = await capturePatch(dir, "committed", baseline)
    expect(patch).toContain("+    return 2")
    expect(patch).not.toContain("scratch.txt")
  })

  test("committed mode excludes uncommitted leftovers", async () => {
    const dir = await repo()
    const baseline = await recordBaselineCommit(dir)

    await Bun.write(path.join(dir, "app.py"), "def solve():\n    return 1\n")
    await commit(dir, "fix")
    await Bun.write(path.join(dir, "repro_scratch.py"), "print('debug')\n")

    const patch = await capturePatch(dir, "committed", baseline)
    expect(patch).toContain("+    return 1")
    expect(patch).not.toContain("repro_scratch.py")
  })

  test("committed mode yields an empty patch when the agent never committed", async () => {
    const dir = await repo()
    const baseline = await recordBaselineCommit(dir)

    await Bun.write(path.join(dir, "app.py"), "def solve():\n    return 1\n")

    expect(await capturePatch(dir, "committed", baseline)).toBe("")
  })

  test("committed mode falls back to the empty tree for an unborn HEAD", async () => {
    const dir = await repo(false)
    const baseline = await recordBaselineCommit(dir)
    expect(baseline).toBe("")

    await Bun.write(path.join(dir, "app.py"), "X = 1\n")
    await commit(dir, "first")

    const patch = await capturePatch(dir, "committed", baseline)
    expect(patch).toContain("app.py")
    expect(patch).toContain("+X = 1")
  })

  test("ensureCommitIdentity fills a missing identity and keeps an existing one", async () => {
    const dir = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), "bench-patch-")))
    created.push(dir)
    await $`git init -q -b main`.cwd(dir).quiet()
    // A user-level identity would mask the "missing" case on a dev machine.
    await $`git config --local user.useConfigOnly true`.cwd(dir).quiet()

    await ensureCommitIdentity(dir)
    expect((await $`git config --get user.email`.cwd(dir).quiet().text()).trim()).not.toBe("")

    await $`git config --local user.email task@example.com`.cwd(dir).quiet()
    await ensureCommitIdentity(dir)
    expect((await $`git config --get user.email`.cwd(dir).quiet().text()).trim()).toBe("task@example.com")
  })
})
