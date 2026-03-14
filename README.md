# lite-grpc

Minimal Rust gRPC router that proxies OpenAI-compatible HTTP requests to [SGLang](https://github.com/sgl-project/sglang) backends. Built for learning and benchmarking.

## Features

- OpenAI-compatible `/v1/chat/completions` API (streaming and non-streaming)
- gRPC client for SGLang scheduler backends
- Tokenizer loading from SGLang's `GetTokenizer` RPC (zip stream, SHA-256 validation)
- Jinja chat template rendering via minijinja
- DeepSeek-R1 reasoning parser (`<think>`/`</think>` extraction)
- DeepSeek tool call parser (Unicode marker format)
- Round-robin worker pool with health checking
- Multiple backend support

## Usage

```sh
cargo build --release

# Single backend
./target/release/lite-grpc --backend http://localhost:30000

# Multiple backends with custom port
./target/release/lite-grpc --backend http://host1:30000 --backend http://host2:30000 --port 9000
```

## API

### POST /v1/chat/completions

```sh
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": false,
    "max_tokens": 200
  }'
```

### GET /v1/models

Returns available models from the connected backends.

### GET /health

Returns `200 OK` if any backend is healthy.

## Architecture

```
HTTP request → Axum → tokenize (chat template + HF tokenizer)
  → select worker (round-robin) → gRPC GenerateRequest
  → SGLang backend → stream/collect response
  → detokenize → parse reasoning + tool calls
  → OpenAI-format JSON/SSE response
```

## Tech Stack

Rust, Axum, Tonic/Prost, HuggingFace `tokenizers`, minijinja, clap, serde
