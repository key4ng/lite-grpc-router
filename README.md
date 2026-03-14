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

## Prerequisites: Start an SGLang Backend

lite-grpc connects to SGLang backends running in gRPC mode. Start one with:

```sh
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --host 0.0.0.0 \
  --port 30000 \
  --context-length 8192 \
  --mem-fraction-static 0.6\
  --tp 2 \
  --grpc-mode
```

> **Note:** Adjust `--tp` to match your number of available GPUs. For the full DeepSeek-R1 671B model, you'll need significantly more VRAM (e.g., 8x 80GB GPUs with FP8).

For multiple backends (e.g., prefill-decode disaggregation), start additional instances on different ports and pass them all via `--backend`.

## Usage

```sh
cargo build --release

# Single backend
./target/release/lite-grpc --backend http://localhost:30000

# Multiple backends with custom port
./target/release/lite-grpc --backend http://host1:30000 --backend http://host2:30000 --port 9000
```

### Using SMG (Shepherd Model Gateway)

Alternatively, use [SMG](https://github.com/lightseekorg/smg) as the router:

```sh
pip install smg

smg launch --worker-urls grpc://localhost:30000 --port 8080 \
  --tokenizer-path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
```

## API

### POST /v1/chat/completions

```sh
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": false,
    "max_tokens": 200
  }'
```

### GET /v1/models

Returns available models from the connected backends.

### GET /health

Returns `200 OK` if any backend is healthy.

## Benchmarking

Use [genai-bench](https://github.com/sgl-project/genai-bench) to benchmark the router:

```sh
pip install genai-bench

genai-bench benchmark \
  --api-backend sglang \
  --api-base "http://localhost:8080" \
  --api-key "none" \
  --api-model-name "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --model-tokenizer "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --task text-to-text \
  --num-concurrency 256 \
  --traffic-scenario "D(1000,1000)" \
  --max-time-per-run 5 \
  --max-requests-per-run 2560
```

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
