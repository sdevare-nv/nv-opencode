// Web-result guards for BrowseComp eval runs. Two independent concerns, one module:
//
//   1. EXCLUDE-DOMAINS (NV TDM opt-out compliance). The tavily arm enforces the
//      47-domain opt-out registry server-side via the MCP DEFAULT_PARAMETERS
//      `exclude_domains` field. The exa arm had NO equivalent: mcp.exa.ai
//      IGNORES excludeDomains server-side (its schema is query+numResults only;
//      live A/B verified), so exclusion must happen client-side on the response.
//      Fed by OPENCODE_WEBSEARCH_EXCLUDE_DOMAINS (JSON array or comma list),
//      which bench/cli.ts populates from the SAME staged
//      mcp/tavily_default_parameters.json the tavily arm uses -> one registry,
//      both providers.
//
//   2. CONTAMINATION (BrowseComp answer-key leakage). Pattern list from the
//      decontam arm (rgala HANDOFF_contamination.md, 23a600af + 2fcbf2d4).
//      Measured on super_v24_exa_cmp{150,192}k_bc400_stepcap: ~15% of instances
//      had at least one of these in a tool result, `huggingface.co/datasets`
//      and `browsecomp` being the two big ones.
//
// UNSET env => exclude-domains is a no-op, byte-identical to stock. The
// contamination guard is always on (matching the decontam arm, which has no
// config knob).

// Name patterns: matched over the WHOLE record (the Highlights body carries the
// text, so a field whitelist would miss it).
const CONTAMINATION_PATTERNS = [
  "browsecomp",
  "browse_comp",
  "browse-comp",
  "simple-eval",
  "bcplus",
  "bc-plus",
  "bc_plus",
]

// Mirror-hosting domains: matched against the record's URL LINE ONLY. rgala is
// explicit that folding these into the text list "would drop any page that
// merely LINKS to a HF dataset" -- a large false-positive class.
const CONTAMINATED_URL_SUBSTRINGS = ["huggingface.co/datasets", "datasets-server.huggingface.co"]

// rgala gotcha 3: "browse_comp" is a substring of "browse_companies" /
// "browse_companiesmarketcap" — legitimate finance pages, NOT contamination.
const BENIGN_BROWSE_COMP = /browse[-_]?compan/g

/** Neutral empty-result string. Deliberately says nothing about WHY. */
export const EMPTY_RESULT = "No results returned"

let cachedDomains: string[] | undefined

/** Excluded domains from env; parsed once. JSON array or comma/colon list. */
export function excludedDomains(): string[] {
  if (cachedDomains !== undefined) return cachedDomains
  const raw = (process.env["OPENCODE_WEBSEARCH_EXCLUDE_DOMAINS"] ?? "").trim()
  if (!raw) return (cachedDomains = [])
  let list: string[] = []
  if (raw.startsWith("[")) {
    try {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) list = parsed.map(String)
    } catch {
      list = []
    }
  } else {
    list = raw.split(/[,:]/)
  }
  cachedDomains = list.map((d) => d.trim().toLowerCase().replace(/^\*?\./, "")).filter(Boolean)
  return cachedDomains
}

/** True when `url`'s host is, or is a subdomain of, an excluded domain. */
export function isExcludedUrl(url: string): boolean {
  const domains = excludedDomains()
  if (domains.length === 0) return false
  let host: string
  try {
    host = new URL(url).hostname.toLowerCase()
  } catch {
    return false
  }
  // Suffix match on a LABEL boundary: "allure.com" must not match
  // "notallure.com", but must match "www.allure.com".
  return domains.some((d) => host === d || host.endsWith("." + d))
}

/** True when the text carries a BrowseComp answer-key marker. */
export function isContaminated(text: string): string | undefined {
  const low = text.toLowerCase()
  for (const p of CONTAMINATION_PATTERNS) {
    if (!low.includes(p)) continue
    if (p.startsWith("browse") && !low.replace(BENIGN_BROWSE_COMP, "").includes(p)) continue
    return p
  }
  return undefined
}

/**
 * Filter an exa websearch response.
 *
 * The exa MCP returns ONE pre-formatted text blob, not a results array, so
 * per-result work means splitting on the record boundary. Records begin at a
 * line-anchored "Title: " and carry a "URL: " line (verified against captured
 * production responses).
 *
 * Contamination is all-or-nothing (blank the whole response); exclusion is
 * per-record, because dropping an entire search because one hit was a
 * Condé Nast page would gut retrieval.
 */
export function filterSearchOutput(text: string | undefined, fn: string): string | undefined {
  if (!text) return text

  const records = text.split(/^(?=Title: )/m).filter((r) => r.length > 0)
  if (records.length === 0) {
    // Unparseable shape (provider format drift). Fall back to whole-blob
    // contamination so a leak can never slip through un-checked; exclusion is
    // skipped because there is no URL to test.
    const p = isContaminated(text)
    if (p) {
      console.error(`[browsecomp][contamination] fn=${fn} provider=exa unparsed_blanked=1 pattern=${p}`)
      return EMPTY_RESULT
    }
    return text
  }

  const kept: string[] = []
  let droppedContam = 0
  let droppedDomain = 0
  for (const rec of records) {
    const url = rec.match(/^URL:\s*(\S+)/m)?.[1] ?? ""
    const lowUrl = url.toLowerCase()
    // (a) mirror-host substrings -- URL field ONLY
    if (CONTAMINATED_URL_SUBSTRINGS.some((s) => lowUrl.includes(s))) {
      droppedContam++
      continue
    }
    // (b) benchmark name patterns -- whole record, so Highlights text is covered
    if (isContaminated(rec)) {
      droppedContam++
      continue
    }
    // (c) TDM opt-out registry
    if (url && isExcludedUrl(url)) {
      droppedDomain++
      continue
    }
    kept.push(rec)
  }

  if (droppedContam)
    console.error(
      `[browsecomp][contamination] fn=${fn} provider=exa dropped=${droppedContam} kept=${kept.length}`,
    )
  if (droppedDomain)
    console.error(
      `[browsecomp][excludedomains] fn=${fn} provider=exa dropped=${droppedDomain} kept=${kept.length}`,
    )
  if (droppedContam === 0 && droppedDomain === 0) return text
  // All dropped => look like an empty provider response, never explain why.
  return kept.length === 0 ? EMPTY_RESULT : kept.join("")
}

/** Filter fetched page content (webfetch). Domain is checked on the INPUT url. */
export function filterFetchOutput(text: string, fn: string): string {
  const pattern = isContaminated(text)
  if (pattern) {
    console.error(`[browsecomp][contamination] fn=${fn} provider=exa blanked=1 pattern=${pattern}`)
    return EMPTY_RESULT
  }
  return text
}
