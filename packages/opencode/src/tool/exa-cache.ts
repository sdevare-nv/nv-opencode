// Exa search cache client (2026-09-10). Talks to the same node-local cache
// sidecar the tavily arm uses (Gym swe_agents/cache_sidecar.py, reached via
// TAVILY_CACHE_URL injected into every rollout container by Gym app.py when
// search_cache_dir is configured). The sidecar owns rgala's flat-file
// SearchCache + FTS5/rapidfuzz fuzzy index, so exa runs get exact AND fuzzy
// hits plus write-through — all against the run's own writable cache root.
//
// VALUE SHAPE. The tavily arm caches the provider's raw results LIST; this
// exa arm caches the RENDERED TEXT BLOCK websearch.ts produces (both the MCP
// and REST paths return text). A hit is accepted only when provider=="exa"
// AND results is a string — rgala's historical exa entries store her python
// harness's highlight-only result lists (no `text` field), which would be a
// strictly worse SERP than a live call, so they are deliberately treated as
// misses. Domain filtering (webfilter.ts) runs AFTER the cache on both hit
// and miss paths, so exclusion-registry changes apply to cached entries too.
//
// Every path fails open: sidecar down / timeout / bad JSON => miss, and the
// live call proceeds. One `[exa-cache]` stderr line per lookup/put mirrors
// tavily_cache.mjs's `[tavily-cache]` logging for wire-level verification.

const LOOKUP_TIMEOUT_MS = 10_000
const PUT_TIMEOUT_MS = 15_000

function cacheUrl(): string | undefined {
  const u = process.env["TAVILY_CACHE_URL"]?.trim()
  return u ? u.replace(/\/+$/, "") : undefined
}

async function post(path: string, body: unknown, timeoutMs: number): Promise<any | undefined> {
  const base = cacheUrl()
  if (!base) return undefined
  const ctl = new AbortController()
  const timer = setTimeout(() => ctl.abort(), timeoutMs)
  try {
    const res = await fetch(`${base}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ctl.signal,
    })
    if (!res.ok) return undefined
    return await res.json()
  } catch (e: any) {
    console.error(`[exa-cache] ${path} error: ${e?.name ?? ""} ${e?.message ?? e}`)
    return undefined
  } finally {
    clearTimeout(timer)
  }
}

export function enabled(): boolean {
  return cacheUrl() !== undefined
}

/** Exact-or-fuzzy lookup. Returns the cached rendered SERP text, or undefined on miss. */
export async function lookup(query: string): Promise<string | undefined> {
  const data = await post("/search", { query }, LOOKUP_TIMEOUT_MS)
  if (data?.hit === true && data.provider === "exa" && typeof data.results === "string" && data.results.length > 0) {
    console.error(`[exa-cache] HIT query=${JSON.stringify(query.slice(0, 120))}`)
    return data.results
  }
  if (data !== undefined) console.error(`[exa-cache] MISS query=${JSON.stringify(query.slice(0, 120))}`)
  return undefined
}

/** Write-through after a live exa call. Fire-and-forget semantics; errors are logged only. */
export async function put(query: string, rendered: string): Promise<void> {
  if (!rendered) return
  const data = await post("/search/put", { query, results: rendered, provider: "exa" }, PUT_TIMEOUT_MS)
  if (data?.ok) console.error(`[exa-cache] PUT query=${JSON.stringify(query.slice(0, 120))} chars=${rendered.length}`)
}
