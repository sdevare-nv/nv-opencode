import { Effect, Schema } from "effect"
import { HttpClient } from "effect/unstable/http"
import * as Tool from "./tool"
import * as McpExa from "./mcp-exa"
import * as ExaCache from "./exa-cache"
import { filterSearchOutput } from "./webfilter"
import DESCRIPTION from "./websearch.txt"

export const Parameters = Schema.Struct({
  query: Schema.String.annotate({ description: "Websearch query" }),
  numResults: Schema.optional(Schema.Number).annotate({
    description: "Number of search results to return (default: 8)",
  }),
  livecrawl: Schema.optional(Schema.Literals(["fallback", "preferred"])).annotate({
    description:
      "Live crawl mode - 'fallback': use live crawling as backup if cached content unavailable, 'preferred': prioritize live crawling (default: 'fallback')",
  }),
  type: Schema.optional(Schema.Literals(["auto", "fast", "deep"])).annotate({
    description: "Search type - 'auto': balanced search (default), 'fast': quick results, 'deep': comprehensive search",
  }),
  contextMaxCharacters: Schema.optional(Schema.Number).annotate({
    description: "Maximum characters for context string optimized for LLMs (default: 10000)",
  }),
})

export const WebSearchTool = Tool.define(
  "websearch",
  Effect.gen(function* () {
    const http = yield* HttpClient.HttpClient

    return {
      get description() {
        return DESCRIPTION.replace("{{year}}", new Date().getFullYear().toString())
      },
      parameters: Parameters,
      execute: (params: Schema.Schema.Type<typeof Parameters>, ctx: Tool.Context) =>
        Effect.gen(function* () {
          yield* ctx.ask({
            permission: "websearch",
            patterns: [params.query],
            always: ["*"],
            metadata: {
              query: params.query,
              numResults: params.numResults,
              livecrawl: params.livecrawl,
              type: params.type,
              contextMaxCharacters: params.contextMaxCharacters,
            },
          })

          // Cache consult first (sidecar reachable only when Gym configured a
          // search cache; see exa-cache.ts). Cached value is the PRE-filter
          // rendered SERP, so webfilter still applies below on both paths.
          const cached = ExaCache.enabled() ? yield* Effect.promise(() => ExaCache.lookup(params.query)) : undefined
          if (cached !== undefined) {
            return {
              output: filterSearchOutput(cached, "websearch") ?? "No search results found. Please try a different query.",
              title: `Web search: ${params.query}`,
              metadata: {},
            }
          }

          // OPENCODE_WEBSEARCH_TYPE forces every search through Exa's REST
          // /search with that type (mcp.exa.ai drops `type`, so deep modes
          // only exist on the REST path). Unset = stock MCP, byte-identical.
          const forcedType = process.env.OPENCODE_WEBSEARCH_TYPE?.trim()
          const result = forcedType
            ? yield* McpExa.restSearch(http, {
                query: params.query,
                type: forcedType,
                numResults: params.numResults || 8,
                livecrawl: params.livecrawl || "fallback",
                contextMaxCharacters: params.contextMaxCharacters,
              })
            : yield* McpExa.call(
                http,
                "web_search_exa",
                McpExa.SearchArgs,
                {
                  query: params.query,
                  type: params.type || "auto",
                  numResults: params.numResults || 8,
                  livecrawl: params.livecrawl || "fallback",
                  contextMaxCharacters: params.contextMaxCharacters,
                },
                "25 seconds",
              )

          // Write-through: cache the raw rendered SERP (pre-filter) so future
          // exact/fuzzy lookups in this run and later runs are API-free.
          if (ExaCache.enabled() && typeof result === "string" && result.length > 0) {
            yield* Effect.promise(() => ExaCache.put(params.query, result))
          }

          // TDM opt-out exclusion + BrowseComp contamination guard. Applied to
          // the raw provider response BEFORE it reaches the model. No-op for
          // exclusion when OPENCODE_WEBSEARCH_EXCLUDE_DOMAINS is unset.
          return {
            output: filterSearchOutput(result, "websearch") ?? "No search results found. Please try a different query.",
            title: `Web search: ${params.query}`,
            metadata: {},
          }
        }).pipe(Effect.orDie),
    }
  }),
)
