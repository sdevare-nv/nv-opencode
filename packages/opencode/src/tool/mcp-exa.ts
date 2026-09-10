import { Duration, Effect, Schema } from "effect"
import { HttpClient, HttpClientRequest } from "effect/unstable/http"

// Read EXA_API_KEY LAZILY, not at module load. bench/cli.ts picks a per-instance
// key out of the EXA_API_KEYS pool (stable hash over instance_id, mirroring the
// tavily rotation) and assigns process.env.EXA_API_KEY while starting the run --
// which lands AFTER this module is imported. A module-scope const would capture
// the pre-rotation value and silently pin every instance to one key.
const url = () =>
  process.env.EXA_API_KEY
    ? `https://mcp.exa.ai/mcp?exaApiKey=${encodeURIComponent(process.env.EXA_API_KEY)}`
    : "https://mcp.exa.ai/mcp"

const McpResult = Schema.Struct({
  result: Schema.Struct({
    content: Schema.Array(
      Schema.Struct({
        type: Schema.String,
        text: Schema.String,
      }),
    ),
  }),
})

const decode = Schema.decodeUnknownEffect(Schema.fromJsonString(McpResult))

const parseSse = Effect.fn("McpExa.parseSse")(function* (body: string) {
  for (const line of body.split("\n")) {
    if (!line.startsWith("data: ")) continue
    const data = yield* decode(line.substring(6))
    if (data.result.content[0]?.text) return data.result.content[0].text
  }
  return undefined
})

export const SearchArgs = Schema.Struct({
  query: Schema.String,
  type: Schema.String,
  numResults: Schema.Number,
  livecrawl: Schema.String,
  contextMaxCharacters: Schema.optional(Schema.Number),
})

const McpRequest = <F extends Schema.Struct.Fields>(args: Schema.Struct<F>) =>
  Schema.Struct({
    jsonrpc: Schema.Literal("2.0"),
    id: Schema.Literal(1),
    method: Schema.Literal("tools/call"),
    params: Schema.Struct({
      name: Schema.String,
      arguments: args,
    }),
  })

// ---------------------------------------------------------------------------
// REST /search override (2026-09-09). mcp.exa.ai's web_search_exa schema is
// {query, numResults} ONLY — it silently drops type/livecrawl (re-verified
// live 09-09), so Exa's deep modes are unreachable through MCP. When
// OPENCODE_WEBSEARCH_TYPE is set, websearch routes here instead: a direct
// POST to api.exa.ai/search with the forced `type`. Unset env = stock MCP
// path, byte-identical behaviour. Deep types additionally request the
// per-result summary and the output-schema synthesis (they are what the
// multi-step research path actually produces; without them deep ~= auto),
// rendered as a capped [Deep Answer] block ahead of the results.
// Every call logs one `[exa-rest]` line (type/latency/cost/nresults) for
// wire-level verification from the run logs.
// ---------------------------------------------------------------------------
const REST_DEEP_TYPES = ["deep", "deep-lite", "deep-reasoning"]
const REST_DEEP_ANSWER_MAX_CHARS = 5000

const RestBody = Schema.Struct({
  query: Schema.String,
  numResults: Schema.Number,
  type: Schema.String,
  contents: Schema.Struct({
    text: Schema.Struct({ maxCharacters: Schema.Number }),
    livecrawl: Schema.String,
    summary: Schema.optional(Schema.Boolean),
  }),
  outputSchema: Schema.optional(Schema.Struct({ type: Schema.String })),
})

const RestResult = Schema.Struct({
  results: Schema.optional(
    Schema.Array(
      Schema.Struct({
        title: Schema.optional(Schema.NullOr(Schema.String)),
        url: Schema.optional(Schema.NullOr(Schema.String)),
        publishedDate: Schema.optional(Schema.NullOr(Schema.String)),
        text: Schema.optional(Schema.NullOr(Schema.String)),
        summary: Schema.optional(Schema.NullOr(Schema.String)),
      }),
    ),
  ),
  output: Schema.optional(Schema.NullOr(Schema.Struct({ content: Schema.optional(Schema.NullOr(Schema.String)) }))),
  costDollars: Schema.optional(Schema.NullOr(Schema.Struct({ total: Schema.optional(Schema.Number) }))),
})

const decodeRest = Schema.decodeUnknownEffect(Schema.fromJsonString(RestResult))

export const restSearch = (
  http: HttpClient.HttpClient,
  args: {
    query: string
    type: string
    numResults: number
    livecrawl: string
    contextMaxCharacters?: number
  },
) =>
  Effect.gen(function* () {
    const deep = REST_DEEP_TYPES.includes(args.type)
    const attempt = Effect.gen(function* () {
      const started = Date.now()
      const request = yield* HttpClientRequest.post("https://api.exa.ai/search").pipe(
        HttpClientRequest.setHeader("x-api-key", process.env.EXA_API_KEY ?? ""),
        HttpClientRequest.accept("application/json"),
        HttpClientRequest.schemaBodyJson(RestBody)({
          query: args.query,
          numResults: args.numResults,
          type: args.type,
          contents: {
            // per-result cap ~2.5k keeps a 10-result SERP near the MCP path's
            // observed 16-21KB output size
            text: { maxCharacters: args.contextMaxCharacters ?? 2500 },
            livecrawl: args.livecrawl,
            ...(deep ? { summary: true } : {}),
          },
          ...(deep ? { outputSchema: { type: "text" } } : {}),
        }),
      )
      const response = yield* HttpClient.filterStatusOk(http)
        .execute(request)
        .pipe(
          Effect.timeoutOrElse({
            duration: deep ? "120 seconds" : "30 seconds",
            orElse: () => Effect.die(new Error(`exa rest search (type=${args.type}) timed out`)),
          }),
        )
      const body = yield* response.text
      const data = yield* decodeRest(body)
      const elapsed = Date.now() - started
      const results = data.results ?? []
      // stderr, not stdout: every other diagnostic in this tree ([exa-cache],
      // webfilter, webfetch) logs to console.error, and stdout does not
      // reliably surface in the Gym driver log from inside the rollout
      // container -- a console.log here is invisible even when deep fires.
      console.error(
        `[exa-rest] type=${args.type} ms=${elapsed} cost=${data.costDollars?.total ?? "?"} ` +
          `n=${results.length} deep_answer=${data.output?.content ? 1 : 0} query=${JSON.stringify(args.query.slice(0, 120))}`,
      )
      const blocks: string[] = []
      const synthesis = data.output?.content
      if (synthesis) {
        const capped =
          synthesis.length > REST_DEEP_ANSWER_MAX_CHARS
            ? synthesis.slice(0, REST_DEEP_ANSWER_MAX_CHARS) + "\n[...deep answer truncated...]"
            : synthesis
        blocks.push(`[Deep Answer]: ${capped}`)
      }
      for (const r of results) {
        let entry = `Title: ${r.title ?? ""}\nURL: ${r.url ?? ""}\nPublished: ${r.publishedDate ?? "N/A"}`
        if (r.summary) entry += `\nSummary: ${r.summary}`
        if (r.text) entry += `\nText: ${r.text}`
        blocks.push(entry)
      }
      if (blocks.length === 0) return undefined
      return blocks.join("\n\n")
    })
    // one retry keeps transient 429s/5xxs from burning an agent turn
    return yield* attempt.pipe(Effect.orElse(() => attempt))
  })

export const call = <F extends Schema.Struct.Fields>(
  http: HttpClient.HttpClient,
  tool: string,
  args: Schema.Struct<F>,
  value: Schema.Struct.Type<F>,
  timeout: Duration.Input,
) =>
  Effect.gen(function* () {
    const request = yield* HttpClientRequest.post(url()).pipe(
      HttpClientRequest.accept("application/json, text/event-stream"),
      HttpClientRequest.schemaBodyJson(McpRequest(args))({
        jsonrpc: "2.0" as const,
        id: 1 as const,
        method: "tools/call" as const,
        params: { name: tool, arguments: value },
      }),
    )
    const response = yield* HttpClient.filterStatusOk(http)
      .execute(request)
      .pipe(
        Effect.timeoutOrElse({ duration: timeout, orElse: () => Effect.die(new Error(`${tool} request timed out`)) }),
      )
    const body = yield* response.text
    return yield* parseSse(body)
  })
