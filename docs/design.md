# lite-grpc: Minimal gRPC Router for SGLang Benchmarking

## Purpose

A standalone, minimal Rust binary that routes OpenAI-compatible HTTP requests to SGLang backends over gRPC. Built for two goals:

1. **Learning** — understand how gRPC routing to inference backends works, without SMG's full complexity
2. **Benchmarking** — compare latency/throughput of lite-grpc vs full SMG gRPC router vs direct SGLang

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Standalone vs deps | Standalone (copy proto, no SMG deps) | Forces understanding of every piece; clean benchmark binary |
| Streaming support | Both streaming + non-streaming | Complete benchmarking coverage |
| Load balancing | Round-robin across N backends | Simple, fair comparison baseline |
| Parsers | DeepSeek reasoning + DeepSeek tool call | Single model family (DeepSeek-R1), covers both use cases |
| Tool parser format | DeepSeek only (not generic JSON) | Only using DeepSeek-R1 |
| API format | OpenAI-compatible `/v1/chat/completions` | Reuse existing benchmark tools across all three targets |
| Tokenizer source | Fetched from SGLang via `GetTokenizer` RPC | Zero config, guaranteed match with running model |
| Project structure | Flat binary (no workspace) | ~2000-2500 lines, no need for crate separation |

## Architecture

```
                  ┌──────────────────────────────────┐
                  │          HTTP Client              │
                  │  POST /v1/chat/completions        │
                  └──────────────┬───────────────────┘
                                 │ JSON
                                 ▼
                  ┌──────────────────────────────────┐
                  │         Axum HTTP Server          │
                  │  Deserialize → ChatCompletionReq  │
                  └──────────────┬───────────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────────┐
                  │           Router                  │
                  │  1. Tokenize input                │
                  │  2. Round-robin select worker     │
                  │  3. Build proto GenerateRequest   │
                  │  4. Send gRPC to SGLang           │
                  │  5. Detokenize (token IDs → text)  │
                  │  6. Parse decoded text             │
                  │     - DeepSeek reasoning parser   │
                  │     - DeepSeek tool call parser   │
                  │  7. Return OpenAI-format response │
                  └──────────────┬───────────────────┘
                                 │ gRPC (tonic)
                        ┌────────┴────────┐
                        ▼                 ▼
                  ┌──────────┐     ┌──────────┐
                  │ SGLang 0 │     │ SGLang 1 │  ... N workers
                  └──────────┘     └──────────┘
```

## File Structure

```
lite-grpc/
├── Cargo.toml
├── proto/
│   ├── sglang_scheduler.proto     # Trimmed proto (Generate, GetModelInfo, HealthCheck only)
│   └── common.proto               # GetTokenizer RPC + tokenizer messages
├── build.rs                       # tonic-build for proto compilation
└── src/
    ├── main.rs                    # CLI (clap) + Axum server setup + startup flow
    ├── client.rs                  # SglangClient — tonic gRPC client wrapper
    ├── router.rs                  # Request routing: tokenize → select → build → dispatch → respond
    ├── worker.rs                  # Worker, WorkerPool, round-robin selection
    ├── tokenizer.rs               # Tokenizer wrapper loaded via GetTokenizer RPC
    ├── parsers/
    │   ├── mod.rs
    │   ├── reasoning.rs           # DeepSeek <think> tag reasoning parser
    │   └── tool.rs                # DeepSeek tool call marker parser
    ├── streaming.rs               # SSE streaming response builder
    └── types.rs                   # OpenAI-compatible request/response types
```

## Component Details

### gRPC Client (`client.rs`)

Wraps a tonic `SglangSchedulerClient<Channel>`:

```rust
struct SglangClient {
    client: SglangSchedulerClient<Channel>,
    url: String,
}
```

Methods:
- `connect(url)` — establish tonic channel
- `generate(req)` — send `GenerateRequest`, return `Streaming<GenerateResponse>`
- `get_tokenizer()` — fetch tokenizer at startup (see Tokenizer section for details)
- `get_model_info()` — fetch model metadata at startup
- `health_check()` — validate connectivity

### Proto files

Two proto files (mirrors SMG's structure):

**`sglang_scheduler.proto`** — trimmed to include only:
- `SglangScheduler` service: `Generate`, `GetModelInfo`, `HealthCheck`
- `GenerateRequest`, `GenerateResponse`, `GenerateStreamChunk`, `GenerateComplete`
- `SamplingParams`, `TokenizedInput`
- ModelInfo messages
- Imports `common.proto` for tokenizer RPCs

**`common.proto`** — shared messages:
- `GetTokenizer` RPC (returns streaming `GetTokenizerChunk` with zip archive bytes + SHA-256 fingerprint)
- `GetTokenizerRequest`, `GetTokenizerChunk` messages

Strip out from both protos: `Embed`, `SubscribeKvEvents`, `GetLoads`, disaggregated params, multimodal inputs, and all KV cache messages (`KvEventBatch`, `KvCacheEvent`, `KvBlock*`, etc.) from `common.proto`.

**Sampling params caveat:** Proto3 defaults (all zeros) don't match SGLang semantics. Must explicitly set: temperature=1.0, top_p=1.0, top_k=-1, max_new_tokens from request or default 2048. `top_k` is not exposed in the OpenAI request type — always hardcode to -1.

**Stream field:** The proto `GenerateRequest` has a `bool stream` field. Must be set to match the client's HTTP `stream` parameter. The gRPC RPC signature is always `returns (stream GenerateResponse)` regardless, but the field tells SGLang whether to send incremental per-token `GenerateStreamChunk` messages (`stream=true`) or batch the output (`stream=false`).

### Worker Pool (`worker.rs`)

```rust
struct Worker {
    client: SglangClient,
    model_name: String,
    healthy: bool,
}

struct WorkerPool {
    workers: Vec<Worker>,
    next: AtomicUsize,
}
```

- Round-robin via `atomic fetch_add % len`, skip unhealthy workers
- No background health check loop — mark unhealthy on gRPC failure. Recovery: on each select(), if all workers are unhealthy, attempt re-check of all workers before failing.
- Tokenizer fetched once from first healthy worker at startup, stored in `Arc<Tokenizer>` (all workers serve same model)

### Tokenizer (`tokenizer.rs`)

Loading a tokenizer from SGLang's `GetTokenizer` RPC is non-trivial:

1. Call `GetTokenizer` RPC — returns a **streaming** response of `GetTokenizerChunk` messages
2. Each chunk contains a `bytes` field (part of a zip archive) and the final chunk includes a `fingerprint` (SHA-256)
3. Collect all chunks into a single byte buffer
4. Validate SHA-256 of the buffer against the fingerprint
5. Extract the zip archive to a temporary directory
6. Load the HuggingFace `tokenizers` crate from the extracted `tokenizer.json` file

Port the stream-collect + zip-extract logic from SMG's `crates/grpc_client/src/tokenizer_bundle.rs`.

**Chat template:** The `tokenizers` crate (0.19+) has `apply_chat_template()` built in with Jinja rendering. No special feature flags needed since we load from a local file (not HuggingFace Hub). DeepSeek models include a Jinja chat template in their `tokenizer_config.json`. Use this to format messages into the model's expected format before encoding.

**Detokenization:** gRPC `GenerateStreamChunk` returns `token_ids` (not decoded text). The tokenizer must decode these to text before passing to parsers. Use `tokenizer.decode(&chunk.token_ids, true)` for each chunk.

### Parsers (`parsers/`)

Both parsers are stateful per-request (created fresh, no shared mutable state).

**Reasoning parser (`reasoning.rs`):**
- State machine starts in `InsideThink` (DeepSeek-R1 outputs reasoning from the start, before any explicit tag)
- `</think>` transitions to `Outside` (normal content)
- Optional `<think>` can re-enter reasoning mode (rare in practice)
- Splits output into `reasoning_content` (thinking) and `content` (final answer)
- Streaming: each SSE chunk carries either field

**Tool call parser (`tool.rs`):**
- DeepSeek uses a multi-token format with outer and inner markers:
  ```
  <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>func_name
  ```json
  {"arg": "value"}
  ```
  <｜tool▁call▁end｜><｜tool▁calls▁end｜>
  ```
- Key markers: `<｜tool▁calls▁begin｜>` (outer), `<｜tool▁call▁begin｜>` / `<｜tool▁call▁end｜>` (per call), `<｜tool▁sep｜>` (between type and name), `<｜tool▁calls▁end｜>` (outer end)
- Supports multiple tool calls in a single response (multiple inner begin/end pairs)
- Accumulates partial JSON during streaming — buffers until complete
- Emits `tool_calls` array with `id`, `function.name`, `function.arguments`

**Pipeline order:**
```
gRPC token stream → detokenize (token IDs → text) → reasoning parser → tool call parser → response builder
```

### OpenAI-Compatible API (`types.rs`)

**Request** (subset of OpenAI spec):
- `model`, `messages` (role + content), `temperature`, `top_p`, `max_tokens`, `stream`, `stop`, `tools`, `frequency_penalty`, `presence_penalty`

**Non-streaming response:**
- `ChatCompletionResponse` with `choices[0].message.content`, `reasoning_content`, `tool_calls`, `usage`

**Streaming response:**
- SSE chunks: `data: {"choices":[{"delta":{"content":"..."}}]}`
- Final chunk: `finish_reason` set
- Stream end: `data: [DONE]`

### HTTP Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | Main inference endpoint |
| `GET /health` | 200 if any worker healthy |
| `GET /v1/models` | Returns model name from `GetModelInfo` |

### CLI

```
lite-grpc --backend http://gpu1:30000 --backend http://gpu2:30000 --port 8080
```

## Request Flow

```
1. Axum receives POST /v1/chat/completions
2. Deserialize JSON → ChatCompletionRequest
3. WorkerPool.select() → round-robin pick healthy worker
4. Tokenize: apply chat template + encode via cached tokenizer → token IDs + text
5. Build proto GenerateRequest:
   - request_id: UUIDv7
   - tokenized: { text, token_ids }
   - sampling_params: { temperature, top_p, max_new_tokens, stop }
6. worker.client.generate(request) → gRPC streaming call
7a. If stream=true:
    - For each gRPC chunk: detokenize token_ids → text via tokenizer
    - Pipe decoded text through reasoning + tool parsers
    - Convert each parsed chunk → SSE ChatCompletionChunk
    - Return Axum streaming response
7b. If stream=false:
    - Collect all gRPC chunks, detokenize token_ids → full text
    - Run reasoning parser → split think/content
    - Run tool parser → extract tool_calls
    - Build ChatCompletionResponse with usage stats
    - Return JSON
8. On gRPC error: mark worker unhealthy, return 502
```

## Intentionally Excluded

- No retry logic (keep benchmarks deterministic)
- No circuit breakers
- No metrics/tracing (use external tools)
- No multimodal support
- No Harmony/PD modes
- No MCP/Responses API
- No WASM plugins
- No auth
- No chat history / storage

## Benchmark Strategy

Run the same benchmark script (e.g., `sglang.bench_serving` or custom OpenAI SDK client) against three targets:

1. **Direct SGLang** — `http://gpu:30000/v1/chat/completions`
2. **lite-grpc** — `http://router:8080/v1/chat/completions`
3. **Full SMG** — `http://smg:8080/v1/chat/completions`

Compare: TTFT, token throughput, p50/p99 latency, overhead per request.

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| `axum` | HTTP server |
| `tonic` / `prost` | gRPC client + proto codegen |
| `tonic-build` | Build-time proto compilation |
| `tokio` | Async runtime |
| `serde` / `serde_json` | JSON serialization |
| `clap` | CLI argument parsing |
| `uuid` | UUIDv7 request IDs |
| `tokenizers` | HuggingFace tokenizers (base crate, `apply_chat_template` works out of the box in 0.19+) |
| `zip` | Extract tokenizer zip archive from GetTokenizer RPC |
| `sha2` | SHA-256 validation of tokenizer bundle |
| `tempfile` | Temporary directory for tokenizer extraction |
| `anyhow` | Error handling |
