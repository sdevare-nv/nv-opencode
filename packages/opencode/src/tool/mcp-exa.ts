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
