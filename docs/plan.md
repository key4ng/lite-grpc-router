# lite-grpc Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone minimal Rust gRPC router that proxies OpenAI-compatible HTTP requests to SGLang backends, with DeepSeek reasoning + tool call parsing, for learning and benchmarking.

**Architecture:** Flat Rust binary. Axum HTTP server receives `/v1/chat/completions`, tokenizes input via a cached tokenizer (fetched from SGLang at startup), builds a proto `GenerateRequest`, sends it via tonic gRPC, detokenizes the response stream, pipes through DeepSeek reasoning + tool parsers, and returns OpenAI-format JSON or SSE.

**Tech Stack:** Rust, Axum, Tonic/Prost, HuggingFace `tokenizers`, minijinja (chat templates), clap, serde

**Spec:** `docs/superpowers/specs/2026-03-13-lite-grpc-router-design.md`

---

## File Structure

```
lite-grpc/                          # New standalone project (sibling to smg/ or wherever user prefers)
├── Cargo.toml                      # Single binary crate, all deps
├── build.rs                        # tonic-build proto compilation
├── proto/
│   ├── sglang_scheduler.proto      # Trimmed: Generate, GetModelInfo, HealthCheck RPCs only
│   └── common.proto                # GetTokenizer RPC + tokenizer chunk messages
└── src/
    ├── main.rs                     # CLI args (clap), startup flow, Axum server
    ├── client.rs                   # SglangClient: tonic gRPC wrapper
    ├── tokenizer.rs                # Tokenizer loading from GetTokenizer RPC + Jinja chat template rendering
    ├── worker.rs                   # Worker + WorkerPool with round-robin selection
    ├── types.rs                    # OpenAI-compatible request/response serde types
    ├── router.rs                   # Request handler: tokenize → select → build proto → dispatch → respond
    ├── streaming.rs                # SSE streaming response builder
    └── parsers/
        ├── mod.rs                  # ParsedChunk type, parser pipeline coordinator
        ├── reasoning.rs            # DeepSeek-R1 reasoning parser (<think>/</ think>)
        └── tool.rs                 # DeepSeek tool call parser (Unicode markers)
```

---

## Chunk 1: Project Scaffold, Proto, and gRPC Client

### Task 1: Create project and Cargo.toml

**Files:**
- Create: `lite-grpc/Cargo.toml`

- [ ] **Step 1: Create the project directory**

```bash
mkdir -p lite-grpc/src lite-grpc/proto
```

- [ ] **Step 2: Write Cargo.toml**

```toml
[package]
name = "lite-grpc"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "lite-grpc"
path = "src/main.rs"

[dependencies]
# HTTP server
axum = { version = "0.8", features = ["json"] }
tokio = { version = "1", features = ["full"] }
tokio-stream = "0.1"

# gRPC client
tonic = "0.13"
prost = "0.13"
prost-types = "0.13"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# CLI
clap = { version = "4", features = ["derive"] }

# Tokenizer
tokenizers = { version = "0.21", features = ["http"] }
zip = "2"
sha2 = "0.10"
tempfile = "3"

# Chat template rendering
minijinja = { version = "2", features = ["builtins"] }

# Utilities
uuid = { version = "1", features = ["v7"] }
anyhow = "1"
regex = "1"
futures = "0.3"

[build-dependencies]
tonic-build = "0.13"
```

Note: The `tokenizers` Rust crate does NOT have `apply_chat_template` (that's Python-only). We use `minijinja` to render the Jinja chat template from `tokenizer_config.json`. The `http` feature on `tokenizers` is only needed for compilation — we load tokenizers from local files, not HuggingFace Hub.

- [ ] **Step 3: Verify the project compiles (empty main)**

Create a minimal `src/main.rs`:
```rust
fn main() {
    println!("lite-grpc");
}
```

Run: `cd lite-grpc && cargo check`
Expected: Compiles successfully (deps may take a while to download on first run)

- [ ] **Step 4: Commit**

```bash
git add lite-grpc/Cargo.toml lite-grpc/src/main.rs
git commit -s -m "chore: scaffold lite-grpc project with dependencies"
```

---

### Task 2: Proto files and build.rs

**Files:**
- Create: `lite-grpc/proto/common.proto`
- Create: `lite-grpc/proto/sglang_scheduler.proto`
- Create: `lite-grpc/build.rs`

Reference: SMG's protos at `crates/grpc_client/proto/sglang_scheduler.proto` and `crates/grpc_client/proto/common.proto`. Strip everything except what we need.

- [ ] **Step 1: Write common.proto**

```protobuf
syntax = "proto3";
package smg.grpc.common;

message GetTokenizerRequest {}

message GetTokenizerChunk {
  bytes data = 1;       // Raw zip bytes (concatenate all chunks)
  string sha256 = 2;    // SHA-256 fingerprint; set on final chunk only, empty on previous
}
```

- [ ] **Step 2: Write sglang_scheduler.proto**

Trim from SMG's `crates/grpc_client/proto/sglang_scheduler.proto`. Keep only:
- `SglangScheduler` service with `Generate`, `GetTokenizer`, `GetModelInfo`, `HealthCheck`
- `GenerateRequest`, `GenerateResponse`, `GenerateStreamChunk`, `GenerateComplete`, `GenerateError`
- `SamplingParams`, `TokenizedInput`
- `GetModelInfoRequest`, `GetModelInfoResponse`
- `HealthCheckRequest`, `HealthCheckResponse`
- Import `common.proto` for tokenizer messages
- Import `google/protobuf/timestamp.proto` and `google/protobuf/struct.proto` as needed

Strip: `Embed*`, `SubscribeKvEvents*`, `GetLoads*`, `DisaggregatedParams`, `MultimodalInputs`, `LogProbs`, `HiddenStates`, `KvBlock*`, `KvCache*`, `KvEventBatch`.

For fields in GenerateRequest that reference stripped types (e.g., `mm_inputs`, `disaggregated_params`), remove those fields entirely. Keep the field numbers of remaining fields unchanged for wire compatibility.

```protobuf
syntax = "proto3";
package sglang.grpc.scheduler;

import "common.proto";
import "google/protobuf/timestamp.proto";
import "google/protobuf/struct.proto";

service SglangScheduler {
  rpc Generate(GenerateRequest) returns (stream GenerateResponse);
  rpc GetTokenizer(smg.grpc.common.GetTokenizerRequest) returns (stream smg.grpc.common.GetTokenizerChunk);
  rpc GetModelInfo(GetModelInfoRequest) returns (GetModelInfoResponse);
  rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
}

message TokenizedInput {
  string original_text = 1;  // For reference
  repeated uint32 input_ids = 2;
}

message SamplingParams {
  float temperature = 1;
  float top_p = 2;
  int32 top_k = 3;
  float min_p = 4;
  float frequency_penalty = 5;
  float presence_penalty = 6;
  float repetition_penalty = 7;
  optional uint32 max_new_tokens = 8;
  repeated string stop = 9;
  repeated uint32 stop_token_ids = 10;
  bool skip_special_tokens = 11;
  bool spaces_between_special_tokens = 12;
  oneof constraint {
    string regex = 13;
    string json_schema = 14;
    string ebnf_grammar = 15;
    string structural_tag = 16;
  }
  uint32 n = 17;
  uint32 min_new_tokens = 18;
  bool ignore_eos = 19;
  bool no_stop_trim = 20;
  optional int32 stream_interval = 21;
  map<string, float> logit_bias = 22;
  google.protobuf.Struct custom_params = 23;
}

message GenerateRequest {
  string request_id = 1;
  TokenizedInput tokenized = 2;
  // field 3 (mm_inputs) removed
  SamplingParams sampling_params = 4;
  bool return_logprob = 5;
  int32 logprob_start_len = 6;
  int32 top_logprobs_num = 7;
  repeated uint32 token_ids_logprob = 8;
  bool return_hidden_states = 9;
  // field 10 (disaggregated_params) removed
  string custom_logit_processor = 11;
  google.protobuf.Timestamp timestamp = 12;
  bool log_metrics = 13;
  repeated float input_embeds = 14;
  string lora_id = 15;
  int32 data_parallel_rank = 16;
  bool stream = 17;
}

message GenerateResponse {
  string request_id = 1;
  oneof response {
    GenerateStreamChunk chunk = 2;
    GenerateComplete complete = 3;
    GenerateError error = 4;
  }
}

message GenerateStreamChunk {
  repeated uint32 token_ids = 1;
  uint32 prompt_tokens = 2;
  uint32 completion_tokens = 3;
  uint32 cached_tokens = 4;
  // field 5 (output_logprobs) removed
  // field 6 (hidden_states) removed
  // field 7 (input_logprobs) removed
  uint32 index = 8;
}

message GenerateComplete {
  repeated uint32 output_ids = 1;
  string finish_reason = 2;
  uint32 prompt_tokens = 3;
  uint32 completion_tokens = 4;
  uint32 cached_tokens = 5;
  // field 6 (output_logprobs) removed
  // field 7 (all_hidden_states) removed
  oneof matched_stop {
    uint32 matched_token_id = 8;
    string matched_stop_str = 9;
  }
  // field 10 (input_logprobs) removed
  uint32 index = 11;
}

message GenerateError {
  string message = 1;
}

message GetModelInfoRequest {}

message GetModelInfoResponse {
  string model_path = 1;
  // field 2 (tokenizer_path) removed
  bool is_generation = 3;
  // fields 4-5 removed
  string served_model_name = 6;
}

message HealthCheckRequest {}

message HealthCheckResponse {
  bool healthy = 1;
}
```

**Important:** Field numbers MUST match the original proto for wire compatibility with existing SGLang servers. Removed fields leave gaps in numbering — this is correct and intentional.

- [ ] **Step 3: Write build.rs**

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(false) // client only
        .compile_protos(
            &["proto/sglang_scheduler.proto", "proto/common.proto"],
            &["proto/"],
        )?;
    Ok(())
}
```

- [ ] **Step 4: Verify proto compilation**

Run: `cd lite-grpc && cargo check`
Expected: Compiles. tonic-build generates Rust code from protos.

- [ ] **Step 5: Commit**

```bash
git add lite-grpc/proto/ lite-grpc/build.rs
git commit -s -m "feat: add trimmed SGLang protos and build.rs"
```

---

### Task 3: gRPC Client wrapper

**Files:**
- Create: `lite-grpc/src/client.rs`
- Modify: `lite-grpc/src/main.rs` (add module declaration)

Reference: SMG's `crates/grpc_client/src/sglang_scheduler.rs` for connection setup and request building.

- [ ] **Step 1: Write the SglangClient**

`src/client.rs`:
```rust
use anyhow::{Context, Result};
use tonic::transport::Channel;

// Import generated proto types
pub mod proto {
    pub mod common {
        tonic::include_proto!("smg.grpc.common");
    }
    pub mod sglang {
        tonic::include_proto!("sglang.grpc.scheduler");
    }
}

use proto::sglang::sglang_scheduler_client::SglangSchedulerClient;

pub struct SglangClient {
    client: SglangSchedulerClient<Channel>,
    url: String,
}

impl SglangClient {
    pub async fn connect(url: &str) -> Result<Self> {
        // tonic needs http:// prefix
        let endpoint = if url.starts_with("http") {
            url.to_string()
        } else {
            format!("http://{url}")
        };

        let channel = Channel::from_shared(endpoint.clone())
            .context("invalid endpoint URL")?
            .connect()
            .await
            .with_context(|| format!("failed to connect to {endpoint}"))?;

        Ok(Self {
            client: SglangSchedulerClient::new(channel),
            url: url.to_string(),
        })
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub async fn generate(
        &self,
        req: proto::sglang::GenerateRequest,
    ) -> Result<tonic::Streaming<proto::sglang::GenerateResponse>> {
        let mut client = self.client.clone();
        let response = client
            .generate(tonic::Request::new(req))
            .await
            .context("generate RPC failed")?;
        Ok(response.into_inner())
    }

    pub async fn get_tokenizer(
        &self,
    ) -> Result<tonic::Streaming<proto::common::GetTokenizerChunk>> {
        let mut client = self.client.clone();
        let response = client
            .get_tokenizer(tonic::Request::new(proto::common::GetTokenizerRequest {}))
            .await
            .context("get_tokenizer RPC failed")?;
        Ok(response.into_inner())
    }

    pub async fn get_model_info(&self) -> Result<proto::sglang::GetModelInfoResponse> {
        let mut client = self.client.clone();
        let response = client
            .get_model_info(tonic::Request::new(proto::sglang::GetModelInfoRequest {}))
            .await
            .context("get_model_info RPC failed")?;
        Ok(response.into_inner())
    }

    pub async fn health_check(&self) -> Result<bool> {
        let mut client = self.client.clone();
        let response = client
            .health_check(tonic::Request::new(proto::sglang::HealthCheckRequest {}))
            .await
            .context("health_check RPC failed")?;
        Ok(response.into_inner().healthy)
    }
}
```

- [ ] **Step 2: Add module to main.rs**

```rust
mod client;

fn main() {
    println!("lite-grpc");
}
```

- [ ] **Step 3: Verify compilation**

Run: `cd lite-grpc && cargo check`
Expected: Compiles

- [ ] **Step 4: Commit**

```bash
git add lite-grpc/src/client.rs lite-grpc/src/main.rs
git commit -s -m "feat: add SglangClient gRPC wrapper"
```

---

## Chunk 2: Tokenizer, Types, and Worker Pool

### Task 4: Tokenizer loading from GetTokenizer RPC

**Files:**
- Create: `lite-grpc/src/tokenizer.rs`
- Modify: `lite-grpc/src/main.rs` (add module)

Reference: SMG's `crates/grpc_client/src/tokenizer_bundle.rs` for stream collection, SHA-256 validation, and zip extraction logic.

- [ ] **Step 1: Write tokenizer.rs**

```rust
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;
use tonic::Streaming;

use crate::client::proto::common::GetTokenizerChunk;

const MAX_BUNDLE_SIZE: usize = 200 * 1024 * 1024; // 200 MB

/// Collect streaming GetTokenizer chunks into a single byte buffer,
/// validate SHA-256, extract zip, and load the HuggingFace tokenizer.
pub async fn load_tokenizer_from_rpc(
    mut stream: Streaming<GetTokenizerChunk>,
) -> Result<LoadedTokenizer> {
    // Step 1: Collect chunks
    let mut data = Vec::new();
    let mut sha256 = String::new();

    while let Some(chunk) = stream.message().await.context("stream error")? {
        if data.len() + chunk.data.len() > MAX_BUNDLE_SIZE {
            bail!("tokenizer bundle exceeds {MAX_BUNDLE_SIZE} bytes");
        }
        data.extend_from_slice(&chunk.data);
        if !chunk.sha256.is_empty() {
            sha256 = chunk.sha256;
        }
    }

    if data.is_empty() {
        bail!("empty tokenizer stream");
    }

    // Step 2: Validate SHA-256
    if !sha256.is_empty() {
        let computed = format!("{:x}", Sha256::digest(&data));
        if !computed.eq_ignore_ascii_case(&sha256) {
            bail!(
                "tokenizer SHA-256 mismatch: expected {sha256}, got {computed}"
            );
        }
    }

    // Step 3: Extract zip to temp directory
    let cursor = Cursor::new(&data);
    let mut archive =
        zip::ZipArchive::new(cursor).context("failed to open tokenizer zip")?;
    let temp_dir = tempfile::tempdir().context("failed to create temp dir")?;
    archive
        .extract(temp_dir.path())
        .context("failed to extract tokenizer zip")?;

    // Step 4: Load tokenizer from extracted files
    load_from_dir(temp_dir.path())
}

/// Load a HuggingFace tokenizer from a directory containing tokenizer.json
/// and optionally tokenizer_config.json.
fn load_from_dir(dir: &Path) -> Result<Arc<Tokenizer>> {
    let tokenizer_json = dir.join("tokenizer.json");
    if !tokenizer_json.exists() {
        bail!(
            "tokenizer.json not found in extracted archive (contents: {:?})",
            std::fs::read_dir(dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect::<Vec<_>>()
        );
    }

    let tokenizer = Tokenizer::from_file(&tokenizer_json)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    // Load chat template from tokenizer_config.json (if present)
    let chat_template = load_chat_template(dir);

    Ok(LoadedTokenizer {
        tokenizer: Arc::new(tokenizer),
        chat_template,
    })
}

/// Loaded tokenizer with optional chat template.
pub struct LoadedTokenizer {
    pub tokenizer: Arc<Tokenizer>,
    pub chat_template: Option<String>,
}

/// Extract the Jinja chat_template string from tokenizer_config.json.
fn load_chat_template(dir: &Path) -> Option<String> {
    let config_path = dir.join("tokenizer_config.json");
    let data = std::fs::read_to_string(&config_path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&data).ok()?;
    config.get("chat_template")?.as_str().map(|s| s.to_string())
}

/// Render messages using a Jinja chat template via minijinja.
pub fn apply_chat_template(
    template: &str,
    messages: &[serde_json::Value],
    add_generation_prompt: bool,
) -> Result<String> {
    let mut env = minijinja::Environment::new();
    env.add_template("chat", template)
        .context("invalid chat template")?;
    let tmpl = env.get_template("chat")
        .context("template not found")?;
    let result = tmpl.render(minijinja::context! {
        messages => messages,
        add_generation_prompt => add_generation_prompt,
    }).context("failed to render chat template")?;
    Ok(result)
}
```

- [ ] **Step 2: Add module to main.rs**

Add `mod tokenizer;` to main.rs.

- [ ] **Step 3: Verify compilation**

Run: `cd lite-grpc && cargo check`
Expected: Compiles

- [ ] **Step 4: Commit**

```bash
git add lite-grpc/src/tokenizer.rs lite-grpc/src/main.rs
git commit -s -m "feat: add tokenizer loading from GetTokenizer RPC"
```

---

### Task 5: OpenAI-compatible types

**Files:**
- Create: `lite-grpc/src/types.rs`
- Modify: `lite-grpc/src/main.rs` (add module)

- [ ] **Step 1: Write types.rs**

```rust
use serde::{Deserialize, Serialize};

// ── Request Types ───────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Tool {
    pub r#type: String,
    pub function: FunctionDef,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FunctionDef {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

// ── Response Types ──────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Streaming Types ─────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Default)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChunkToolCall {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    pub function: ChunkFunctionCall,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChunkFunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// ── Models Endpoint ─────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub owned_by: String,
}
```

- [ ] **Step 2: Add module, verify compilation**

Add `mod types;` to main.rs.
Run: `cd lite-grpc && cargo check`
Expected: Compiles

- [ ] **Step 3: Commit**

```bash
git add lite-grpc/src/types.rs lite-grpc/src/main.rs
git commit -s -m "feat: add OpenAI-compatible request/response types"
```

---

### Task 6: Worker pool with round-robin selection

**Files:**
- Create: `lite-grpc/src/worker.rs`
- Modify: `lite-grpc/src/main.rs` (add module)

- [ ] **Step 1: Write worker.rs**

```rust
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{bail, Result};
use tokio::sync::RwLock;

use crate::client::SglangClient;

pub struct Worker {
    pub client: SglangClient,
    pub model_name: String,
    healthy: AtomicBool,
}

impl Worker {
    pub fn new(client: SglangClient, model_name: String) -> Self {
        Self {
            client,
            model_name,
            healthy: AtomicBool::new(true),
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    pub fn mark_unhealthy(&self) {
        self.healthy.store(false, Ordering::Relaxed);
    }

    pub fn mark_healthy(&self) {
        self.healthy.store(true, Ordering::Relaxed);
    }
}

pub struct WorkerPool {
    workers: Vec<Arc<Worker>>,
    next: AtomicUsize,
}

impl WorkerPool {
    pub fn new(workers: Vec<Arc<Worker>>) -> Self {
        Self {
            workers,
            next: AtomicUsize::new(0),
        }
    }

    /// Round-robin select a healthy worker.
    /// If all workers are unhealthy, attempt health checks on all before failing.
    pub async fn select(&self) -> Result<Arc<Worker>> {
        let len = self.workers.len();

        // Try round-robin, skip unhealthy
        let start = self.next.fetch_add(1, Ordering::Relaxed);
        for i in 0..len {
            let idx = (start + i) % len;
            if self.workers[idx].is_healthy() {
                return Ok(Arc::clone(&self.workers[idx]));
            }
        }

        // All unhealthy — attempt recovery
        for worker in &self.workers {
            if let Ok(true) = worker.client.health_check().await {
                worker.mark_healthy();
                return Ok(Arc::clone(worker));
            }
        }

        bail!("no healthy workers available")
    }

    pub fn workers(&self) -> &[Arc<Worker>] {
        &self.workers
    }

    pub fn model_name(&self) -> &str {
        &self.workers[0].model_name
    }
}
```

- [ ] **Step 2: Add module, verify compilation**

Add `mod worker;` to main.rs.
Run: `cd lite-grpc && cargo check`
Expected: Compiles (may warn about unused imports — that's fine for now)

- [ ] **Step 3: Commit**

```bash
git add lite-grpc/src/worker.rs lite-grpc/src/main.rs
git commit -s -m "feat: add WorkerPool with round-robin selection"
```

---

## Chunk 3: Parsers

### Task 7: Reasoning parser

**Files:**
- Create: `lite-grpc/src/parsers/mod.rs`
- Create: `lite-grpc/src/parsers/reasoning.rs`
- Modify: `lite-grpc/src/main.rs` (add module)

Reference: SMG's `crates/reasoning_parser/src/parsers/base.rs` for the state machine logic.

- [ ] **Step 1: Write parsers/mod.rs**

```rust
pub mod reasoning;
pub mod tool;

/// Output from the parser pipeline for a single chunk of decoded text.
#[derive(Debug, Default)]
pub struct ParsedChunk {
    /// Normal (non-reasoning) content to send to the client
    pub content: String,
    /// Reasoning/thinking content
    pub reasoning_content: String,
    /// Completed tool calls (non-streaming: full calls; streaming: incremental)
    pub tool_calls: Vec<ParsedToolCall>,
}

#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}
```

- [ ] **Step 2: Write parsers/reasoning.rs**

```rust
/// DeepSeek-R1 reasoning parser.
///
/// State machine: starts in `in_reasoning = true` (DeepSeek-R1 emits reasoning
/// from the very first token, before any explicit `<think>` tag).
/// `</think>` transitions to normal content mode.
/// Optional `<think>` can re-enter reasoning mode.

const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";

#[derive(Debug)]
pub struct ReasoningParser {
    in_reasoning: bool,
    buffer: String,
    stripped_think_start: bool,
}

/// Result of parsing one chunk.
#[derive(Debug, Default)]
pub struct ReasoningResult {
    pub reasoning_text: String,
    pub normal_text: String,
}

impl ReasoningParser {
    pub fn new() -> Self {
        Self {
            in_reasoning: true, // DeepSeek-R1 starts in reasoning mode
            buffer: String::new(),
            stripped_think_start: false,
        }
    }

    /// Check if the buffer is a prefix of a start or end token.
    fn is_partial_token(text: &str) -> bool {
        (THINK_START.starts_with(text) && THINK_START != text)
            || (THINK_END.starts_with(text) && THINK_END != text)
    }

    /// Parse one chunk of streaming text. Returns reasoning and/or normal text.
    pub fn parse_chunk(&mut self, text: &str) -> ReasoningResult {
        self.buffer.push_str(text);
        let current = self.buffer.clone();

        // If the buffer is a prefix of a token, keep buffering
        if Self::is_partial_token(&current) {
            return ReasoningResult::default();
        }

        // Strip <think> start token if present
        let current = if !self.stripped_think_start && current.contains(THINK_START) {
            self.stripped_think_start = true;
            self.in_reasoning = true;
            let replaced = current.replace(THINK_START, "");
            self.buffer.clone_from(&replaced);
            replaced
        } else {
            current
        };

        // Look for </think> end token
        if self.in_reasoning {
            if let Some(end_idx) = current.find(THINK_END) {
                // Found end of reasoning
                let reasoning = current[..end_idx].trim().to_string();
                let normal_start = end_idx + THINK_END.len();
                let normal = if normal_start < current.len() {
                    current[normal_start..].to_string()
                } else {
                    String::new()
                };
                self.buffer.clear();
                self.in_reasoning = false;
                return ReasoningResult {
                    reasoning_text: reasoning,
                    normal_text: normal,
                };
            }

            // Still in reasoning, stream it out
            self.buffer.clear();
            ReasoningResult {
                reasoning_text: current,
                normal_text: String::new(),
            }
        } else {
            // Normal text mode
            self.buffer.clear();
            ReasoningResult {
                reasoning_text: String::new(),
                normal_text: current,
            }
        }
    }

    /// Parse complete text (non-streaming).
    pub fn parse_complete(&mut self, text: &str) -> ReasoningResult {
        let text = if self.in_reasoning || text.contains(THINK_START) {
            text.replace(THINK_START, "")
        } else {
            return ReasoningResult {
                reasoning_text: String::new(),
                normal_text: text.to_string(),
            };
        };

        if let Some(end_idx) = text.find(THINK_END) {
            let reasoning = text[..end_idx].trim().to_string();
            let normal = text[end_idx + THINK_END.len()..].trim().to_string();
            ReasoningResult {
                reasoning_text: reasoning,
                normal_text: normal,
            }
        } else {
            // No end token — all reasoning
            ReasoningResult {
                reasoning_text: text.trim().to_string(),
                normal_text: String::new(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starts_in_reasoning_mode() {
        let mut parser = ReasoningParser::new();
        let result = parser.parse_complete("I am thinking about this");
        assert_eq!(result.reasoning_text, "I am thinking about this");
        assert_eq!(result.normal_text, "");
    }

    #[test]
    fn test_think_end_splits_content() {
        let mut parser = ReasoningParser::new();
        let result = parser.parse_complete("reasoning here</think>final answer");
        assert_eq!(result.reasoning_text, "reasoning here");
        assert_eq!(result.normal_text, "final answer");
    }

    #[test]
    fn test_streaming_reasoning_then_content() {
        let mut parser = ReasoningParser::new();

        let r1 = parser.parse_chunk("thinking ");
        assert_eq!(r1.reasoning_text, "thinking ");
        assert_eq!(r1.normal_text, "");

        let r2 = parser.parse_chunk("more ");
        assert_eq!(r2.reasoning_text, "more ");
        assert_eq!(r2.normal_text, "");

        let r3 = parser.parse_chunk("done</think>answer");
        assert_eq!(r3.reasoning_text, "done");
        assert_eq!(r3.normal_text, "answer");

        let r4 = parser.parse_chunk(" continues");
        assert_eq!(r4.reasoning_text, "");
        assert_eq!(r4.normal_text, " continues");
    }

    #[test]
    fn test_streaming_partial_end_token() {
        let mut parser = ReasoningParser::new();

        let r1 = parser.parse_chunk("text</");
        // "</thi" is a prefix of "</think>", so buffer it
        // Actually "</thi" is not — "</think>" starts with "</", but "text</" is not a prefix.
        // The buffer is "text</" — is_partial_token checks if the ENTIRE buffer is a prefix.
        // "text</" is NOT a prefix of "</think>", so it should emit.
        assert_eq!(r1.reasoning_text, "text</");

        // Next chunk completes or doesn't
        let r2 = parser.parse_chunk("think>answer");
        assert_eq!(r2.reasoning_text, "");
        assert_eq!(r2.normal_text, "answer");
    }

    #[test]
    fn test_explicit_think_start_stripped() {
        let mut parser = ReasoningParser::new();
        let result = parser.parse_complete("<think>reasoning</think>answer");
        assert_eq!(result.reasoning_text, "reasoning");
        assert_eq!(result.normal_text, "answer");
    }
}
```

- [ ] **Step 3: Add module, run tests**

Add `mod parsers;` to main.rs.
Run: `cd lite-grpc && cargo test parsers::reasoning`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add lite-grpc/src/parsers/
git commit -s -m "feat: add DeepSeek-R1 reasoning parser"
```

---

### Task 8: Tool call parser

**Files:**
- Create: `lite-grpc/src/parsers/tool.rs`

Reference: SMG's `crates/tool_parser/src/parsers/deepseek.rs` for marker tokens and parsing logic.

- [ ] **Step 1: Write parsers/tool.rs**

```rust
/// DeepSeek tool call parser.
///
/// Detects and extracts tool calls from DeepSeek's Unicode-marker format:
/// ```text
/// <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>func_name
/// ```json
/// {"arg": "value"}
/// ```
/// <｜tool▁call▁end｜><｜tool▁calls▁end｜>
/// ```

use regex::Regex;

use super::ParsedToolCall;

// DeepSeek Unicode markers
const CALLS_BEGIN: &str = "<｜tool\u{2581}calls\u{2581}begin｜>";
const CALL_BEGIN: &str = "<｜tool\u{2581}call\u{2581}begin｜>";
const CALL_END: &str = "<｜tool\u{2581}call\u{2581}end｜>";
const CALLS_END: &str = "<｜tool\u{2581}calls\u{2581}end｜>";
const TOOL_SEP: &str = "<｜tool\u{2581}sep｜>";

#[derive(Debug)]
pub struct ToolCallParser {
    buffer: String,
    /// Regex to extract complete tool calls
    complete_re: Regex,
    /// Regex to extract function details from a single tool call block
    detail_re: Regex,
    tool_id_counter: u32,
}

impl ToolCallParser {
    pub fn new() -> Self {
        // Match a complete tool call block (DOTALL mode for newlines in JSON)
        let complete_pattern = format!(
            r"(?s){CALL_BEGIN}.*?{CALL_END}",
        );
        // Extract: function type, function name, JSON arguments
        let detail_pattern = format!(
            r"(?s){CALL_BEGIN}(.*?){TOOL_SEP}(.*?)\n```json\n(.*?)\n```{CALL_END}",
        );

        Self {
            buffer: String::new(),
            complete_re: Regex::new(&complete_pattern).expect("valid regex"),
            detail_re: Regex::new(&detail_pattern).expect("valid regex"),
            tool_id_counter: 0,
        }
    }

    /// Parse complete text (non-streaming). Returns (normal_text, tool_calls).
    pub fn parse_complete(&mut self, text: &str) -> (String, Vec<ParsedToolCall>) {
        // Check if there are any tool calls
        let Some(calls_begin_idx) = text.find(CALLS_BEGIN) else {
            return (text.to_string(), vec![]);
        };

        // Text before tool calls is normal content
        let normal_text = text[..calls_begin_idx].to_string();

        // Extract all tool calls
        let tool_section = &text[calls_begin_idx..];
        let tool_calls = self.extract_tool_calls(tool_section);

        (normal_text, tool_calls)
    }

    /// Extract tool calls from a text section containing tool markers.
    fn extract_tool_calls(&mut self, text: &str) -> Vec<ParsedToolCall> {
        let mut calls = Vec::new();

        for cap in self.detail_re.captures_iter(text) {
            let name = cap[2].trim().to_string();
            let arguments = cap[3].trim().to_string();
            let id = format!("call_{}", self.tool_id_counter);
            self.tool_id_counter += 1;

            calls.push(ParsedToolCall {
                id,
                name,
                arguments,
            });
        }

        calls
    }

    /// Streaming: accumulate text and detect tool call boundaries.
    /// Returns (normal_text_to_emit, completed_tool_calls).
    pub fn parse_chunk(&mut self, text: &str) -> (String, Vec<ParsedToolCall>) {
        self.buffer.push_str(text);

        // If we haven't seen the calls begin marker yet, check for it
        let Some(calls_begin_idx) = self.buffer.find(CALLS_BEGIN) else {
            // No tool calls started — emit all buffered text as normal
            // But keep a small tail in case a partial marker spans chunks
            if self.buffer.len() > CALLS_BEGIN.len() {
                let emit_end = self.buffer.len() - CALLS_BEGIN.len();
                let emit = self.buffer[..emit_end].to_string();
                self.buffer = self.buffer[emit_end..].to_string();
                return (emit, vec![]);
            }
            return (String::new(), vec![]);
        };

        // Emit any normal text before the tool calls marker
        let normal = self.buffer[..calls_begin_idx].to_string();

        // Check if we have the full tool calls section (ends with CALLS_END)
        let tool_section = &self.buffer[calls_begin_idx..];
        if let Some(calls_end_idx) = tool_section.find(CALLS_END) {
            let full_section = &tool_section[..calls_end_idx + CALLS_END.len()];
            let calls = self.extract_tool_calls(full_section);

            // Keep anything after CALLS_END
            let after = &tool_section[calls_end_idx + CALLS_END.len()..];
            self.buffer = after.to_string();

            return (normal, calls);
        }

        // Tool calls started but not complete — buffer and wait
        // Only emit the normal text prefix
        if !normal.is_empty() {
            self.buffer = self.buffer[calls_begin_idx..].to_string();
            return (normal, vec![]);
        }

        (String::new(), vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool_call(name: &str, args: &str) -> String {
        format!(
            "{CALL_BEGIN}function{TOOL_SEP}{name}\n```json\n{args}\n```{CALL_END}"
        )
    }

    fn wrap_calls(inner: &str) -> String {
        format!("{CALLS_BEGIN}{inner}{CALLS_END}")
    }

    #[test]
    fn test_no_tool_calls() {
        let mut parser = ToolCallParser::new();
        let (normal, calls) = parser.parse_complete("just regular text");
        assert_eq!(normal, "just regular text");
        assert!(calls.is_empty());
    }

    #[test]
    fn test_single_tool_call() {
        let mut parser = ToolCallParser::new();
        let tool = make_tool_call("get_weather", r#"{"city": "NYC"}"#);
        let text = format!("Here's the weather: {}", wrap_calls(&tool));
        let (normal, calls) = parser.parse_complete(&text);
        assert_eq!(normal, "Here's the weather: ");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city": "NYC"}"#);
    }

    #[test]
    fn test_multiple_tool_calls() {
        let mut parser = ToolCallParser::new();
        let tool1 = make_tool_call("get_weather", r#"{"city": "NYC"}"#);
        let tool2 = make_tool_call("search", r#"{"query": "weather"}"#);
        let text = wrap_calls(&format!("{tool1}{tool2}"));
        let (_, calls) = parser.parse_complete(&text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[1].name, "search");
    }

    #[test]
    fn test_streaming_complete_in_one_chunk() {
        let mut parser = ToolCallParser::new();
        let tool = make_tool_call("func", r#"{"a": 1}"#);
        let text = format!("normal text{}", wrap_calls(&tool));
        let (normal, calls) = parser.parse_chunk(&text);
        assert_eq!(normal, "normal text");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "func");
    }

    #[test]
    fn test_streaming_split_across_chunks() {
        let mut parser = ToolCallParser::new();
        let tool = make_tool_call("func", r#"{"a": 1}"#);
        let full = format!("hello {}", wrap_calls(&tool));
        let mid = full.len() / 2;

        let (n1, c1) = parser.parse_chunk(&full[..mid]);
        // May or may not emit normal text depending on where the split lands
        assert!(c1.is_empty()); // Tool call not complete yet

        let (n2, c2) = parser.parse_chunk(&full[mid..]);
        let combined_normal = format!("{n1}{n2}");
        assert!(combined_normal.contains("hello"));
        assert_eq!(c2.len(), 1);
        assert_eq!(c2[0].name, "func");
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cd lite-grpc && cargo test parsers::tool`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add lite-grpc/src/parsers/tool.rs
git commit -s -m "feat: add DeepSeek tool call parser"
```

---

## Chunk 4: Router, Streaming, and Main

### Task 9: Streaming SSE response builder

**Files:**
- Create: `lite-grpc/src/streaming.rs`
- Modify: `lite-grpc/src/main.rs` (add module)

- [ ] **Step 1: Write streaming.rs**

```rust
use std::sync::Arc;

use axum::response::sse::{Event, Sse};
use futures::stream::{self, Stream, StreamExt};
use tokio_stream::StreamExt as TokioStreamExt;
use tonic::Streaming;

use crate::client::proto::sglang::{GenerateResponse, generate_response};
use crate::parsers::reasoning::ReasoningParser;
use crate::parsers::tool::ToolCallParser;
use crate::types::*;

/// Build an SSE stream from a gRPC GenerateResponse stream.
pub fn build_sse_stream(
    grpc_stream: Streaming<GenerateResponse>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    request_id: String,
    model: String,
    created: i64,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let mut reasoning_parser = ReasoningParser::new();
    let mut tool_parser = ToolCallParser::new();
    let mut sent_role = false;

    let stream = TokioStreamExt::map(grpc_stream, move |result| {
        match result {
            Ok(response) => {
                match response.response {
                    Some(generate_response::Response::Chunk(chunk)) => {
                        // Detokenize
                        let text = tokenizer
                            .decode(&chunk.token_ids, true)
                            .unwrap_or_default();

                        if text.is_empty() {
                            return None;
                        }

                        // Run through reasoning parser
                        let reasoning_result = reasoning_parser.parse_chunk(&text);

                        // Run normal text through tool parser
                        let (content, tool_calls) = if !reasoning_result.normal_text.is_empty() {
                            tool_parser.parse_chunk(&reasoning_result.normal_text)
                        } else {
                            (String::new(), vec![])
                        };

                        // Build delta
                        let mut delta = Delta::default();

                        if !sent_role {
                            delta.role = Some("assistant".to_string());
                        }

                        if !reasoning_result.reasoning_text.is_empty() {
                            delta.reasoning_content = Some(reasoning_result.reasoning_text);
                        }
                        if !content.is_empty() {
                            delta.content = Some(content);
                        }
                        if !tool_calls.is_empty() {
                            delta.tool_calls = Some(
                                tool_calls.iter().enumerate().map(|(i, tc)| {
                                    ChunkToolCall {
                                        index: i as u32,
                                        id: Some(tc.id.clone()),
                                        r#type: Some("function".to_string()),
                                        function: ChunkFunctionCall {
                                            name: Some(tc.name.clone()),
                                            arguments: Some(tc.arguments.clone()),
                                        },
                                    }
                                }).collect(),
                            );
                        }

                        // Skip empty deltas
                        if delta.role.is_none()
                            && delta.content.is_none()
                            && delta.reasoning_content.is_none()
                            && delta.tool_calls.is_none()
                        {
                            return None;
                        }

                        sent_role = true;

                        let chunk = ChatCompletionChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta,
                                finish_reason: None,
                            }],
                        };

                        let json = serde_json::to_string(&chunk).unwrap_or_default();
                        Some(Event::default().data(json))
                    }
                    Some(generate_response::Response::Complete(complete)) => {
                        // Send final chunk with finish_reason
                        let chunk = ChatCompletionChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta::default(),
                                finish_reason: Some(complete.finish_reason.clone()),
                            }],
                        };
                        let json = serde_json::to_string(&chunk).unwrap_or_default();
                        Some(Event::default().data(json))
                    }
                    Some(generate_response::Response::Error(err)) => {
                        let error_json = serde_json::json!({"error": err.message});
                        Some(Event::default().data(error_json.to_string()))
                    }
                    None => None,
                }
            }
            Err(_) => None,
        }
    })
    .filter_map(|opt| async { opt })
    .chain(stream::once(async {
        Event::default().data("[DONE]")
    }))
    .map(Ok);

    Sse::new(stream)
}
```

- [ ] **Step 2: Add module, verify compilation**

Add `mod streaming;` to main.rs.
Run: `cd lite-grpc && cargo check`
Expected: Compiles

- [ ] **Step 3: Commit**

```bash
git add lite-grpc/src/streaming.rs lite-grpc/src/main.rs
git commit -s -m "feat: add SSE streaming response builder"
```

---

### Task 10: Router (request handler)

**Files:**
- Create: `lite-grpc/src/router.rs`
- Modify: `lite-grpc/src/main.rs` (add module)

Reference: SMG's `crates/grpc_client/src/sglang_scheduler.rs` for building `GenerateRequest` and `SamplingParams`.

- [ ] **Step 1: Write router.rs**

```rust
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use uuid::Uuid;

use crate::client::proto::sglang::{self, generate_response};
use crate::parsers::reasoning::ReasoningParser;
use crate::parsers::tool::ToolCallParser;
use crate::streaming::build_sse_stream;
use crate::types::*;
use crate::worker::WorkerPool;

/// Shared application state
pub struct AppState {
    pub pool: WorkerPool,
    pub tokenizer: Arc<tokenizers::Tokenizer>,
    pub chat_template: Option<String>,
    pub model_name: String,
}

/// POST /v1/chat/completions handler
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, String)> {
    let is_stream = req.stream.unwrap_or(false);
    let request_id = format!("chatcmpl-{}", Uuid::now_v7());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    // 1. Select worker
    let worker = state.pool.select().await.map_err(|e| {
        (StatusCode::SERVICE_UNAVAILABLE, e.to_string())
    })?;

    // 2. Tokenize: apply chat template + encode
    let (text, token_ids) = tokenize_messages(
        &state.tokenizer,
        state.chat_template.as_deref(),
        &req.messages,
    ).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    // 3. Build proto GenerateRequest
    let proto_req = build_generate_request(
        &request_id,
        text,
        token_ids,
        &req,
        is_stream,
    );

    // 4. Send gRPC request
    let grpc_stream = worker.client.generate(proto_req).await.map_err(|e| {
        worker.mark_unhealthy();
        (StatusCode::BAD_GATEWAY, format!("gRPC error: {e}"))
    })?;

    // 5. Build response
    if is_stream {
        let sse = build_sse_stream(
            grpc_stream,
            Arc::clone(&state.tokenizer),
            request_id,
            state.model_name.clone(),
            created,
        );
        Ok(sse.into_response())
    } else {
        let response = collect_response(
            grpc_stream,
            &state.tokenizer,
            request_id,
            state.model_name.clone(),
            created,
        )
        .await
        .map_err(|e| {
            worker.mark_unhealthy();
            (StatusCode::BAD_GATEWAY, format!("response error: {e}"))
        })?;
        Ok(Json(response).into_response())
    }
}

/// Apply chat template and tokenize messages.
fn tokenize_messages(
    tokenizer: &tokenizers::Tokenizer,
    chat_template: Option<&str>,
    messages: &[Message],
) -> Result<(String, Vec<u32>)> {
    // Build messages as JSON values for the Jinja template
    let msg_values: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| serde_json::json!({
            "role": m.role,
            "content": m.content.clone().unwrap_or_default(),
        }))
        .collect();

    // Apply chat template if available, otherwise concatenate messages
    let text = if let Some(template) = chat_template {
        crate::tokenizer::apply_chat_template(template, &msg_values, true)?
    } else {
        // Fallback: simple concatenation (not ideal, but functional)
        messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content.as_deref().unwrap_or("")))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let encoding = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenization error: {e}"))?;

    Ok((text, encoding.get_ids().to_vec()))
}

/// Build a proto GenerateRequest from the OpenAI request.
fn build_generate_request(
    request_id: &str,
    text: String,
    token_ids: Vec<u32>,
    req: &ChatCompletionRequest,
    stream: bool,
) -> sglang::GenerateRequest {
    let sampling_params = sglang::SamplingParams {
        temperature: req.temperature.unwrap_or(1.0) as f32,
        top_p: req.top_p.unwrap_or(1.0) as f32,
        top_k: -1, // Always disabled
        max_new_tokens: req.max_tokens,
        stop: req.stop.clone().unwrap_or_default(),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0) as f32,
        presence_penalty: req.presence_penalty.unwrap_or(0.0) as f32,
        skip_special_tokens: true,
        spaces_between_special_tokens: true,
        ..Default::default()
    };

    sglang::GenerateRequest {
        request_id: request_id.to_string(),
        tokenized: Some(sglang::TokenizedInput {
            original_text: text,
            input_ids: token_ids,
        }),
        sampling_params: Some(sampling_params),
        stream,
        ..Default::default()
    }
}

/// Collect a non-streaming response from the gRPC stream.
async fn collect_response(
    mut grpc_stream: tonic::Streaming<sglang::GenerateResponse>,
    tokenizer: &tokenizers::Tokenizer,
    request_id: String,
    model: String,
    created: i64,
) -> Result<ChatCompletionResponse> {
    let mut all_token_ids: Vec<u32> = Vec::new();
    let mut prompt_tokens = 0u32;
    let mut completion_tokens = 0u32;
    let mut finish_reason = String::new();

    while let Some(response) = grpc_stream.message().await.context("stream error")? {
        match response.response {
            Some(generate_response::Response::Chunk(chunk)) => {
                all_token_ids.extend_from_slice(&chunk.token_ids);
                prompt_tokens = chunk.prompt_tokens;
                completion_tokens = chunk.completion_tokens;
            }
            Some(generate_response::Response::Complete(complete)) => {
                finish_reason = complete.finish_reason;
                prompt_tokens = complete.prompt_tokens;
                completion_tokens = complete.completion_tokens;
                if !complete.output_ids.is_empty() {
                    all_token_ids = complete.output_ids;
                }
            }
            Some(generate_response::Response::Error(err)) => {
                bail!("backend error: {}", err.message);
            }
            None => {}
        }
    }

    // Detokenize
    let full_text = tokenizer
        .decode(&all_token_ids, true)
        .map_err(|e| anyhow::anyhow!("detokenize error: {e}"))?;

    // Parse reasoning
    let mut reasoning_parser = ReasoningParser::new();
    let reasoning_result = reasoning_parser.parse_complete(&full_text);

    // Parse tool calls from normal text
    let mut tool_parser = ToolCallParser::new();
    let (content, tool_calls) = tool_parser.parse_complete(&reasoning_result.normal_text);

    // Build response
    let message = Message {
        role: "assistant".to_string(),
        content: if content.is_empty() { None } else { Some(content) },
        reasoning_content: if reasoning_result.reasoning_text.is_empty() {
            None
        } else {
            Some(reasoning_result.reasoning_text)
        },
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls.into_iter().map(|tc| ToolCall {
                id: tc.id,
                r#type: "function".to_string(),
                function: FunctionCall {
                    name: tc.name,
                    arguments: tc.arguments,
                },
            }).collect())
        },
        tool_call_id: None,
    };

    let total_tokens = prompt_tokens + completion_tokens;

    Ok(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion",
        created,
        model,
        choices: vec![Choice {
            index: 0,
            message,
            finish_reason: Some(if finish_reason.is_empty() {
                "stop".to_string()
            } else {
                finish_reason
            }),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        },
    })
}

/// GET /health handler
pub async fn health(State(state): State<Arc<AppState>>) -> StatusCode {
    for worker in state.pool.workers() {
        if worker.is_healthy() {
            return StatusCode::OK;
        }
    }
    StatusCode::SERVICE_UNAVAILABLE
}

/// GET /v1/models handler
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelObject {
            id: state.model_name.clone(),
            object: "model",
            owned_by: "sglang".to_string(),
        }],
    })
}
```

- [ ] **Step 2: Add module, verify compilation**

Add `mod router;` to main.rs.
Run: `cd lite-grpc && cargo check`
Expected: Compiles

- [ ] **Step 3: Commit**

```bash
git add lite-grpc/src/router.rs lite-grpc/src/main.rs
git commit -s -m "feat: add request router with tokenization and proto building"
```

---

### Task 11: Main entry point with CLI and Axum server

**Files:**
- Modify: `lite-grpc/src/main.rs` (complete rewrite)

- [ ] **Step 1: Write the full main.rs**

```rust
mod client;
mod parsers;
mod router;
mod streaming;
mod tokenizer;
mod types;
mod worker;

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use axum::routing::{get, post};
use axum::Router;
use clap::Parser;

use crate::client::SglangClient;
use crate::router::AppState;
use crate::worker::{Worker, WorkerPool};

#[derive(Parser)]
#[command(name = "lite-grpc", about = "Minimal gRPC router for SGLang")]
struct Cli {
    /// SGLang backend gRPC URLs (can specify multiple)
    #[arg(long = "backend", required = true)]
    backends: Vec<String>,

    /// Port to listen on
    #[arg(long, default_value = "8080")]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.backends.is_empty() {
        bail!("at least one --backend URL is required");
    }

    eprintln!("lite-grpc starting...");
    eprintln!("Backends: {:?}", cli.backends);

    // Connect to all backends
    let mut workers = Vec::new();
    for url in &cli.backends {
        eprint!("Connecting to {url}... ");
        let client = SglangClient::connect(url)
            .await
            .with_context(|| format!("failed to connect to {url}"))?;

        let model_info = client.get_model_info().await
            .with_context(|| format!("failed to get model info from {url}"))?;
        eprintln!("OK (model: {})", model_info.served_model_name);

        workers.push(Arc::new(Worker::new(client, model_info.served_model_name)));
    }

    let model_name = workers[0].model_name.clone();

    // Load tokenizer from first backend
    eprint!("Loading tokenizer from {}... ", cli.backends[0]);
    let tokenizer_stream = workers[0].client.get_tokenizer().await
        .context("failed to start tokenizer download")?;
    let loaded = tokenizer::load_tokenizer_from_rpc(tokenizer_stream).await
        .context("failed to load tokenizer")?;
    eprintln!("OK (chat template: {})", if loaded.chat_template.is_some() { "found" } else { "not found" });

    // Build app state
    let pool = WorkerPool::new(workers);
    let state = Arc::new(AppState {
        pool,
        tokenizer: loaded.tokenizer,
        chat_template: loaded.chat_template,
        model_name,
    });

    // Build Axum router
    let app = Router::new()
        .route("/v1/chat/completions", post(router::chat_completions))
        .route("/v1/models", get(router::list_models))
        .route("/health", get(router::health))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", cli.port);
    eprintln!("Listening on {addr}");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
```

- [ ] **Step 2: Verify full compilation**

Run: `cd lite-grpc && cargo build`
Expected: Compiles and produces `target/debug/lite-grpc` binary

- [ ] **Step 3: Test CLI help**

Run: `cd lite-grpc && cargo run -- --help`
Expected: Shows usage with `--backend` and `--port` options

- [ ] **Step 4: Commit**

```bash
git add lite-grpc/src/main.rs
git commit -s -m "feat: add main entry point with CLI and Axum server"
```

---

## Chunk 5: Integration Testing and Polish

### Task 12: Build verification and dry-run test

- [ ] **Step 1: Full build**

Run: `cd lite-grpc && cargo build`
Expected: Clean compilation with no errors

- [ ] **Step 2: Run all unit tests**

Run: `cd lite-grpc && cargo test`
Expected: All reasoning and tool parser tests pass

- [ ] **Step 3: Clippy lint check**

Run: `cd lite-grpc && cargo clippy --all-targets -- -D warnings`
Expected: No warnings (fix any that appear)

- [ ] **Step 4: Test binary starts and fails gracefully without backend**

Run: `cd lite-grpc && cargo run -- --backend http://localhost:9999`
Expected: Connection error message (not a panic)

- [ ] **Step 5: Commit any fixes**

```bash
git add -A lite-grpc/
git commit -s -m "fix: address clippy warnings and build issues"
```

---

### Task 13: Manual integration test (requires running SGLang)

This task is for when you have an SGLang backend running. Skip if no backend is available.

- [ ] **Step 1: Start lite-grpc**

```bash
cd lite-grpc && cargo run -- --backend http://<sglang-host>:30000 --port 8080
```

Expected: Connects, loads tokenizer, prints "Listening on 0.0.0.0:8080"

- [ ] **Step 2: Test health endpoint**

```bash
curl http://localhost:8080/health
```

Expected: 200 OK

- [ ] **Step 3: Test models endpoint**

```bash
curl http://localhost:8080/v1/models | jq
```

Expected: JSON with model name from SGLang

- [ ] **Step 4: Test non-streaming chat completion**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [{"role": "user", "content": "What is 2+2? Think step by step."}],
    "stream": false,
    "max_tokens": 200
  }' | jq
```

Expected: JSON response with `reasoning_content` and `content` fields

- [ ] **Step 5: Test streaming chat completion**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": true,
    "max_tokens": 200
  }'
```

Expected: SSE stream with `data:` lines containing chunks, ending with `data: [DONE]`

- [ ] **Step 6: Test tool calling (if model supports it)**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [{"role": "user", "content": "What is the weather in NYC?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
    "stream": false,
    "max_tokens": 500
  }' | jq
```

Expected: Response with `tool_calls` array if model invokes the tool

---

### Task 14: Final commit and summary

- [ ] **Step 1: Final commit with all files**

Ensure all files are committed:
```bash
cd lite-grpc && git status
```

If any uncommitted files:
```bash
git add -A lite-grpc/
git commit -s -m "feat: complete lite-grpc minimal gRPC router for SGLang"
```
