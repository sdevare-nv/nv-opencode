# Sanity-run menu — nv-opencode fork (sdd/dev), aarch64 Linux, headless

**Repo root (run everything from here unless noted):** `/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/abhibhag/nv-opencode`

**Global gotchas that apply to several runs**
- Root `bun test` / `bun run test` is deliberately blocked (bunfig `[test] root="./do-not-run-tests-from-root"`); always run tests from inside a package. If you see "do not run tests from root" + exit 1, that is correct behavior, not breakage.
- No `rg` on this node: any run touching grep/glob/ripgrep tests auto-downloads a ripgrep aarch64 tarball from github.com **on every full-suite run** (test preload uses fresh XDG tmpdirs). Put an `rg` binary on PATH to make the whole suite offline.
- From-source version string is `local`; no API keys are needed anywhere (test preload deletes them; models.dev is stubbed by the committed empty `models-snapshot.js`). A background models.dev refresh fetch can still fire on registry-building commands — suppress with `OPENCODE_DISABLE_MODELS_FETCH=1` (non-fatal offline either way).
- Non-TTY `opencode run` reads stdin to EOF — always append `</dev/null` in scripts.
- Without `OPENCODE_DB=':memory:'`, the first real invocation runs a one-time sqlite migration and writes `~/.local/share/opencode/opencode-local.db` (writes outside the repo).

---

## CORE LADDER (in order; each step assumes the previous passed)

### 0. Install-complete gate
```
cd <repo> && ls node_modules/.bin | wc -l && ls -d node_modules/@happy-dom node_modules/@lydell node_modules/@typescript
```
- **Validates:** the concurrent `bun install` actually finished (at exploration time `.bin` was empty).
- **Prereqs:** none. **Expected:** count > 0; all three dirs exist.
- **Failure modes:** missing dirs = install still running/failed; nothing downstream is meaningful yet.

### 1. CLI boots from source
```
cd <repo> && bun packages/opencode/src/index.ts --version && bun packages/opencode/src/index.ts --help
```
- **Validates:** the exact un-bundled module graph the bench subprocess falls back to — fork's TUI-import removal and the offline `models-snapshot` stub. `--help` loads all ~21 subcommand modules (much wider import surface).
- **Prereqs:** step 0. Fully offline (yargs short-circuits before any registry/network work).
- **Expected:** prints `local`, then usage text listing run/serve/models/providers/mcp/export/db etc.; exit 0.
- **Failure modes:** `ResolveMessage` / `react/jsx-dev-runtime` errors = incomplete install or a regression in the fork's TUI trim; if it only fails plain but works via `bun dev -- --help` (root script adds `--conditions=browser`), a browser-conditioned import leaked back into the entry graph — that would also break the bench child spawn.

### 2. Lint (cheapest whole-repo oracle)
```
cd <repo> && bun run lint
```
- **Validates:** native linux-arm64 oxlint + oxlint-tsgolint bindings load and parse the whole monorepo.
- **Prereqs:** none beyond install. Offline.
- **Expected:** "Found N warnings and 0 errors", exit 0 (warnings are fine).
- **Failure modes:** native binding load error = arm64 prebuild problem (same class of risk as node-pty later).

### 3. Typecheck (mirrors typecheck.yml CI)
```
cd <repo> && bun typecheck
```
- **Validates:** turbo-linux-arm64 + tsgo (@typescript/native-preview arm64) binaries, workspace/catalog resolution, TS health of all packages — both are pinned in bun.lock for arm64.
- **Prereqs:** none. Offline. **Expected:** all turbo tasks green, "Tasks: N successful, N total", exit 0.
- **Failure modes:** tsgo binary crash on this 64K-page kernel would show here first (cheap place to catch it).

### 4. Fast hermetic test smoke
```
cd <repo>/packages/opencode && bun test test/tool/edit.test.ts test/util/
```
- **Validates:** bun test runner + test/preload.ts sandbox (isolated XDG tmpdirs, in-memory sqlite, local models fixture, key-deletion) + raw-src workspace resolution, without paying for the full suite.
- **Prereqs:** git user.name/email configured (fixtures create tmp git repos; CI sets a bot identity). Offline, no rg needed for this subset.
- **Expected:** ~27 edit tests + util suites pass in seconds, 0 fail.
- **Failure modes:** failures here = broken install or preload, not flaky tests.

### 5. Full opencode package suite (CI-equivalent core)
```
cd <repo>/packages/opencode && bun run test
```
- **Validates:** 200 test files — server/httpapi loopback servers, session fake-LLM streaming/compaction, lsp, shell, and **pty** (real PTYs via @lydell/node-pty native aarch64 prebuild + `fix-node-pty` postinstall chmod — the most platform-risky piece on this cluster).
- **Prereqs:** git identity; network to github.com **once per run** for the ripgrep download (or `rg` on PATH → fully offline); node on PATH recommended. No keys, no model endpoint.
- **Expected:** all pass in several minutes (30s per-test timeout preset).
- **Failure modes:** `test/pty/*` failures = node-pty arm64 prebuild / postinstall problem; grep/glob/ripgrep failures = blocked github.com egress; avoids turbo's `^build` codegen side effects that `bun turbo test:ci` has.

### 6. Offline provider registry with a vLLM-shaped config
```
cat > /tmp/oc-sanity.json <<'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8000/v1", "apiKey": "dummy" },
      "models": { "MODEL_ID": { "name": "MODEL_ID", "limit": { "context": 131072, "output": 32768 }, "tool_call": true, "temperature": true } }
    }
  },
  "model": "vllm/MODEL_ID"
}
EOF
cd <repo> && OPENCODE_DISABLE_MODELS_FETCH=1 OPENCODE_DB=':memory:' OPENCODE_CONFIG=/tmp/oc-sanity.json bun packages/opencode/src/index.ts models
```
- **Validates:** config parsing + provider merge + that a config-declared OpenAI-compatible provider registers with zero network. Key fork fact: the empty snapshot stub means config-declared providers are the ONLY way to get a model from source (env keys like OPENAI_API_KEY enable nothing).
- **Prereqs:** none — the endpoint does NOT need to be up; MODEL_ID stays a placeholder. Writes only /tmp/oc-sanity.json.
- **Expected:** stdout is exactly `vllm/MODEL_ID`, exit 0.
- **Failure modes:** empty output = config not picked up (check OPENCODE_CONFIG path); any models.dev fetch attempt = env var not applied.

### 7. Headless serve + health curl (no model)
```
cd <repo> && OPENCODE_DISABLE_MODELS_FETCH=1 OPENCODE_DB=':memory:' bun packages/opencode/src/index.ts serve --port 4996 --hostname 127.0.0.1 & sleep 8; curl -sf http://127.0.0.1:4996/global/health; echo; curl -sf "http://127.0.0.1:4996/config/providers?directory=/tmp" | head -c 300; kill %1
```
- **Validates:** the effect-httpapi server backend (default for from-source "local" channel) boots headless, listens, answers unauthenticated (no OPENCODE_SERVER_PASSWORD ⇒ open, warning printed).
- **Prereqs:** a free port; localhost only.
- **Expected:** `opencode server listening on http://127.0.0.1:4996`; `{"healthy":true,"version":"local"}`; JSON providers payload.
- **Failure modes:** if routes look missing, retry with `OPENCODE_EXPERIMENTAL_HTTPAPI=0` (legacy hono backend; also the only backend serving `GET /doc`). Port collision on shared nodes — pick another.

### 8. End-to-end one-prompt run against your vLLM endpoint
```
export SERVED_MODEL_NAME=<your --served-model-name>; sed -i "s|http://127.0.0.1:8000/v1|http://<HOST>:<PORT>/v1|; s/MODEL_ID/$SERVED_MODEL_NAME/g" /tmp/oc-sanity.json
cd <repo> && OPENCODE_DISABLE_MODELS_FETCH=1 OPENCODE_DB=':memory:' OPENCODE_CONFIG=/tmp/oc-sanity.json bun packages/opencode/src/index.ts run -m "vllm/$SERVED_MODEL_NAME" 'Reply with exactly: SANITY_OK' </dev/null
```
- **Validates:** the full path config → bundled @ai-sdk/openai-compatible SDK → chat/completions on your self-hosted endpoint → session event stream → stdout.
- **Prereqs:** a live OpenAI-compatible vLLM endpoint; model key and `-m` arg MUST equal vLLM's `--served-model-name`; baseURL needs the `/v1` suffix; `</dev/null` is mandatory in scripts.
- **Expected:** header line then `SANITY_OK`, exit 0.
- **Failure modes:** `ModelNotFoundError` = served-model-name mismatch; ECONNREFUSED = wrong baseURL (add `--print-logs --log-level DEBUG` to see the outbound request).

---

## OPTIONAL DEEP CHECKS (fork bench machinery, builds, CI parity)

### A. Bench driver argparse smoke (free)
```
cd <repo> && bun packages/opencode/src/bench/cli.ts; echo exit=$?
```
- **Validates:** the fork's 596-line bench driver module graph (deep_reset.ts, bootstrap_repo.ts, bundled anthropic.txt) compiles; arg validation fires.
- **Expected:** stderr `[bench] fatal: Error: Missing required arg --instance-dict-path`, `exit=2` — NOT a compile/resolve error. Note: sdd/dev has NO `--context-limit`/`--patch-mode` (those are on sibling branches only).

### B. nemo-gym provider factory (free)
```
cd <repo> && bun -e 'const m=await import("<repo>/packages/opencode/src/provider/sdk/nemo-gym/index.ts");const lm=m.createNemoGym({baseURL:"http://127.0.0.1:9"}).languageModel("m");console.log(lm.provider,lm.modelId,lm.specificationVersion)'
```
- **Validates:** the fork's LanguageModelV3 token-ID-capture SDK compiles and exposes the ai-sdk shape provider.ts registers. **Expected:** `nemo-gym m v3`, exit 0.

### C. deep_reset git sanitizer on a scratch repo — **writes /tmp/dr_ws**
```
rm -rf /tmp/dr_ws && git init -q /tmp/dr_ws && git -C /tmp/dr_ws -c user.email=a@b -c user.name=t commit -q --allow-empty -m base && BASE=$(git -C /tmp/dr_ws rev-parse HEAD) && git -C /tmp/dr_ws -c user.email=a@b -c user.name=t commit -q --allow-empty -m future && git -C /tmp/dr_ws tag leak && DR_WS=/tmp/dr_ws DR_BASE=$BASE bun -e 'const {runDeepReset}=await import("<repo>/packages/opencode/src/bench/deep_reset.ts"); await runDeepReset(process.env.DR_WS, process.env.DR_BASE)' && echo "HEAD_ok=$([ "$(git -C /tmp/dr_ws rev-parse HEAD)" = "$BASE" ] && echo yes)"; git -C /tmp/dr_ws tag -l | wc -l; git -C /tmp/dr_ws log --oneline --all | cat
```
- **Validates:** the bench history-pruner (careful pass + nuclear fallback, 10-min process-group timeout, `OPENCODE_DEEP_RESET_TIMEOUT_MS` override).
- **Expected:** `[deep_reset:careful] ... done`, HEAD_ok=yes, 0 tags, only the `base` commit reachable.

### D. Offline end-to-end mock-gym bench run — **the definitive fork check; writes /tmp/benchsmoke and leaves /tmp/bench-smoke__t1-\* behind; keep the `HOME=/tmp/benchsmoke` override or bootstrap's `git config --global --add safe.directory` will edit your real ~/.gitconfig**

Use the `offline-e2e-mock-gym` command verbatim from the fork-diff report (python3 stdlib mock OpenAI server on port 18099 asserting flattened string content, temperature=0.6/top_p=0.95 forced, `max_tokens` absent; then `HOME=/tmp/benchsmoke NEMO_GYM_MODEL_SERVER_BASE_URL=http://127.0.0.1:18099 bun packages/opencode/src/bench/cli.ts --instance-dict-path ... --selected-id smoke__t1 --max-turns 3`).
- **Validates:** the FULL bench pipeline with no gym and no model: bootstrap_repo git-init, per-instance opencode.jsonc generation, gym-proxy content flattening, forced sampling params, `OPENCODE_DISABLE_ENV_PROMPT`, `llm_completions/` trajectory dump with `generation_token_ids` preserved, `output.jsonl`, deterministic exit 0.
- **Prereqs:** core step 1 passed (child spawn uses the un-bundled entry unless `.bench-build/opencode.js` exists); python3; free port 18099.
- **Expected:** `[bench] bootstrap_repo exit=0` → `[bench] wrote .../output.jsonl (patch=0 bytes, error=none)` → `bench_exit=0`; exactly one dump file `mock-model-ses_*-0000-*.json` with `token_ids_ok= True`, `env_block_absent= True`; empty mock-server log (no assertion tracebacks).
- **Failure modes:** child `ResolveMessage` errors → run E first (cli.ts auto-prefers `.bench-build/opencode.js`); port in use; hung child killed by the driver's timeout machinery.

### E. Bench-shape bundle build — **writes untracked `.bench-build/` into the repo**
```
cd <repo> && bun build --target=bun --outfile .bench-build/opencode.js packages/opencode/src/index.ts && bun .bench-build/opencode.js --version
```
- **Validates:** the pre-bundled artifact the SIF deployment actually runs, and that the committed empty models-snapshot stub lets `bun build` succeed with zero network (do NOT run script/generate.ts — that fetches models.dev).
- **Expected:** clean build, then `local`. Afterwards run D again — it routes through the bundle automatically.

### F. Sandbox config inspection (after D; reads the leftover tmpdir)
```
grep -c '"deny"' /tmp/bench-smoke__t1-*/opencode.jsonc && python3 -c "import json,glob; c=json.load(open(glob.glob('/tmp/bench-smoke__t1-*/opencode.jsonc')[0])); a=c['agent']['swe-bench']; print('steps=',a['steps'],'compaction=',c['compaction'],'websearch=',a['tools']['websearch'],'git_fetch=',a['permission']['bash']['*git fetch*'])"
```
- **Validates:** the generated per-instance sandbox: 73 bash deny globs, websearch/webfetch off, compaction.auto=false, steps=maxTurns. **Expected:** `73`; `steps= 3 compaction= {'auto': False} websearch= False git_fetch= deny`.

### G. app + ui unit tests (rest of CI unit matrix; offline)
```
cd <repo>/packages/app && bun run test:unit && cd ../ui && bun test src && cd ../core && bun test
```
- **Expected:** all pass, <1–2 min. happy-dom import error = install incomplete. Skip Playwright e2e entirely (needs browsers; headless cluster).

### H. Exact CI parity — **may dirty the tree: `^build` regenerates tracked `packages/sdk/js/src/v2/gen` + plugin dist, writes junit artifacts**
```
cd <repo> && bun turbo test:ci
```
- **Expected:** 3 green tasks; junit at `packages/{opencode,app,ui}/.artifacts/unit/junit.xml`; `git status` after should show a no-op diff at a fixed commit. Prefer core steps 4/5 + G unless you need literal CI parity.

### I. Standalone linux-arm64 binary build (mirrors publish.yml) — **rewrites TRACKED `models-snapshot.{js,d.ts}`; ~1–2 GB disk; restore after**
```
cd <repo> && MODELS_DEV_API_JSON=$PWD/packages/opencode/test/tool/fixtures/models-api.json bun packages/opencode/script/build.ts --single --skip-embed-web-ui --skip-install
git checkout -- packages/opencode/src/provider/models-snapshot.js packages/opencode/src/provider/models-snapshot.d.ts
```
- **Validates:** Bun.build compile for this platform + the binary itself — build.ts self-smoke-tests `--version` and exits 1 on failure; also the strongest check that a static bun binary boots on this 64K-page aarch64 kernel.
- **Expected:** `Smoke test passed: 0.0.0-sdd/dev-<ts>`; binary at `packages/opencode/dist/opencode-linux-arm64/bin/opencode` (then run its `--version`/`--help` by hand). Dropping MODELS_DEV_API_JSON makes it fetch live models.dev (network).

---

## Repo/host mutation summary
- **Writes outside repo:** D (/tmp/benchsmoke + leftover /tmp/bench-\* dirs; `git config --global` unless HOME overridden — keep the override), C (/tmp/dr_ws), 6/8 (/tmp/oc-sanity.json), any run without `OPENCODE_DB=':memory:'` (sqlite DB + migration in ~/.local/share), ripgrep auto-download (per-run XDG tmpdirs).
- **Writes inside repo:** E (untracked .bench-build/), H (tracked sdk codegen + .artifacts junit), I (tracked models-snapshot files — must `git checkout --` after; plus dist/).
- **Network:** only ripgrep download (steps 5/H, github.com), background models.dev refresh when `OPENCODE_DISABLE_MODELS_FETCH` unset (non-fatal), and step 8 to your own vLLM endpoint. Everything else is fully offline; no external LLM API keys used anywhere.