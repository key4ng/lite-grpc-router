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
./target/release/lite-grpc --backend http://localhost:30000 \
--tokenizer deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

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
  --num-concurrency 64 \
  --num-concurrency 128 \
  --num-concurrency 256 \
  --num-concurrency 512 \
  --traffic-scenario "D(100,100)" \
  --max-time-per-run 5 \
  --max-requests-per-run 2560
```

### Results: SGLang HTTP vs lite-grpc vs SMG

Tested with DeepSeek-R1-Distill-Qwen-32B, D(100,100), 2x H100 TP=2. **Bold** = best, positive % = improvement over HTTP.

**TTFT (s)** — lower is better

| Concurrency | Metric | HTTP | lite-grpc | SMG | lite-grpc vs HTTP | SMG vs HTTP |
|---:|:---|---:|---:|---:|---:|---:|
| **64** | Mean | **0.351** | 0.418 | 0.420 | -19.1% | -19.9% |
|  | P50 | **0.345** | 0.417 | 0.421 | -20.9% | -22.1% |
|  | P95 | 0.585 | 0.473 | **0.470** | +19.2% | +19.7% |
|  | P99 | 0.886 | 0.583 | **0.582** | +34.2% | +34.4% |
| **128** | Mean | **0.626** | 0.678 | 0.694 | -8.5% | -10.9% |
|  | P50 | **0.687** | 0.688 | 0.693 | -0.1% | -0.9% |
|  | P95 | 0.846 | **0.747** | 0.843 | +11.7% | +0.4% |
|  | P99 | 1.048 | **0.760** | 0.887 | +27.5% | +15.3% |
| **256** | Mean | **1.122** | 1.127 | 1.173 | -0.4% | -4.6% |
|  | P50 | 1.244 | **1.236** | 1.256 | +0.7% | -1.0% |
|  | P95 | 1.535 | **1.497** | 1.528 | +2.5% | +0.4% |
|  | P99 | 1.626 | **1.567** | 1.665 | +3.6% | -2.4% |
| **512** | Mean | 2.173 | **1.977** | 2.090 | +9.0% | +3.8% |
|  | P50 | 2.344 | **2.171** | 2.240 | +7.4% | +4.4% |
|  | P95 | 2.933 | **2.684** | 2.810 | +8.5% | +4.2% |
|  | P99 | 3.123 | **2.721** | 2.960 | +12.9% | +5.2% |

**E2E Latency (s)** — lower is better

| Concurrency | Metric | HTTP | lite-grpc | SMG | lite-grpc vs HTTP | SMG vs HTTP |
|---:|:---|---:|---:|---:|---:|---:|
| **64** | Mean | 2.170 | **2.124** | 2.127 | +2.1% | +2.0% |
|  | P50 | 2.153 | **2.116** | 2.122 | +1.7% | +1.5% |
|  | P95 | 2.358 | 2.256 | **2.252** | +4.3% | +4.5% |
|  | P99 | 2.657 | 2.320 | **2.294** | +12.7% | +13.7% |
| **128** | Mean | 2.854 | **2.790** | 2.826 | +2.3% | +1.0% |
|  | P50 | 2.835 | **2.798** | 2.832 | +1.3% | +0.1% |
|  | P95 | 3.141 | **3.018** | 3.097 | +3.9% | +1.4% |
|  | P99 | 3.369 | **3.104** | 3.214 | +7.9% | +4.6% |
| **256** | Mean | 4.249 | **4.169** | 4.239 | +1.9% | +0.2% |
|  | P50 | 4.262 | **4.185** | 4.251 | +1.8% | +0.3% |
|  | P95 | 4.716 | **4.686** | 4.742 | +0.6% | -0.6% |
|  | P99 | 5.145 | **4.854** | 4.875 | +5.7% | +5.2% |
| **512** | Mean | 8.041 | **7.852** | 7.960 | +2.4% | +1.0% |
|  | P50 | 8.046 | **7.873** | 7.942 | +2.2% | +1.3% |
|  | P95 | 9.173 | **8.775** | 9.089 | +4.3% | +0.9% |
|  | P99 | 9.502 | **8.991** | 9.628 | +5.4% | -1.3% |

**Input Throughput (tok/s)** — higher is better

| Concurrency | Metric | HTTP | lite-grpc | SMG | lite-grpc vs HTTP | SMG vs HTTP |
|---:|:---|---:|---:|---:|---:|---:|
| **64** | Mean | **317.5** | 238.5 | 236.6 | -24.9% | -25.5% |
|  | P50 | **285.2** | 236.2 | 234.1 | -17.2% | -17.9% |
|  | P5 | 169.6 | 207.7 | **208.4** | +22.5% | +22.9% |
|  | P1 | 110.5 | 169.2 | **170.7** | +53.1% | +54.5% |
| **128** | Mean | **177.7** | 146.6 | 143.6 | -17.5% | -19.2% |
|  | P50 | **143.1** | 142.8 | 141.8 | -0.2% | -0.9% |
|  | P5 | 116.1 | **131.1** | 116.8 | +13.0% | +0.6% |
|  | P1 | 93.4 | **128.1** | 110.9 | +37.1% | +18.7% |
| **256** | Mean | **116.8** | 105.0 | 95.2 | -10.1% | -18.6% |
|  | P50 | 78.9 | **79.5** | 78.2 | +0.7% | -1.0% |
|  | P5 | 64.1 | **65.5** | 64.1 | +2.1% | +0.0% |
|  | P1 | 59.5 | **62.6** | 59.1 | +5.3% | -0.6% |
| **512** | Mean | 50.8 | **67.1** | 59.0 | +32.0% | +16.0% |
|  | P50 | 41.9 | **45.3** | 43.8 | +8.0% | +4.5% |
|  | P5 | 33.4 | **36.6** | 34.9 | +9.6% | +4.3% |
|  | P1 | 31.5 | **35.8** | 33.0 | +13.4% | +4.8% |

**Output Throughput (tok/s)** — higher is better

| Concurrency | Metric | HTTP | lite-grpc | SMG | lite-grpc vs HTTP | SMG vs HTTP |
|---:|:---|---:|---:|---:|---:|---:|
| **64** | Mean | 54.6 | **58.1** | 58.1 | +6.4% | +6.4% |
|  | P50 | 54.7 | **58.3** | 58.3 | +6.7% | +6.7% |
|  | P5 | 49.6 | **55.0** | 54.9 | +10.7% | +10.6% |
|  | P1 | 48.4 | **52.1** | 52.0 | +7.6% | +7.4% |
| **128** | Mean | 44.8 | **47.2** | 46.7 | +5.2% | +4.2% |
|  | P50 | 45.0 | **46.3** | 45.9 | +2.9% | +2.1% |
|  | P5 | 37.9 | **42.6** | 41.4 | +12.6% | +9.3% |
|  | P1 | 36.3 | **40.9** | 39.5 | +12.9% | +8.8% |
| **256** | Mean | 32.2 | **33.4** | 33.0 | +3.6% | +2.6% |
|  | P50 | 32.0 | 32.7 | **32.9** | +2.1% | +2.6% |
|  | P5 | 25.9 | 25.9 | **26.4** | +0.1% | +2.2% |
|  | P1 | 24.6 | 24.7 | **25.3** | +0.4% | +2.7% |
| **512** | Mean | 17.3 | **17.3** | 17.3 | +0.3% | +0.3% |
|  | P50 | 16.8 | **16.8** | 16.7 | +0.5% | -0.6% |
|  | P5 | 13.5 | **14.2** | 14.0 | +4.8% | +3.7% |
|  | P1 | 12.7 | **13.6** | 13.5 | +7.6% | +6.9% |

gRPC routers (lite-grpc and SMG) consistently outperform SGLang's native HTTP server on tail latency and worst-case throughput, with the gap widening at higher concurrency.

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
