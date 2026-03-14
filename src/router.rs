use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use uuid::Uuid;

use crate::proto::sglang::{self, generate_response};
use crate::parsers::reasoning::ReasoningParser;
use crate::parsers::tool::ToolCallParser;
use crate::streaming::build_sse_stream;
use crate::types::*;
use crate::worker::WorkerPool;

pub struct AppState {
    pub pool: WorkerPool,
    pub tokenizer: Arc<tokenizers::Tokenizer>,
    pub chat_template: Option<String>,
    pub model_name: String,
}

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

    let worker = state.pool.select().await.map_err(|e| {
        (StatusCode::SERVICE_UNAVAILABLE, e.to_string())
    })?;

    let (text, token_ids) = tokenize_messages(
        &state.tokenizer,
        state.chat_template.as_deref(),
        &req.messages,
    ).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let proto_req = build_generate_request(
        &request_id,
        text,
        token_ids,
        &req,
        is_stream,
    );

    let grpc_stream = worker.client.generate(proto_req).await.map_err(|e| {
        worker.mark_unhealthy();
        (StatusCode::BAD_GATEWAY, format!("gRPC error: {e}"))
    })?;

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

fn tokenize_messages(
    tokenizer: &tokenizers::Tokenizer,
    chat_template: Option<&str>,
    messages: &[Message],
) -> Result<(String, Vec<u32>)> {
    let msg_values: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| serde_json::json!({
            "role": m.role,
            "content": m.content.clone().unwrap_or_default(),
        }))
        .collect();

    let text = if let Some(template) = chat_template {
        crate::tokenizer::apply_chat_template(template, &msg_values, true)?
    } else {
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
        top_k: -1,
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

    let full_text = tokenizer
        .decode(&all_token_ids, true)
        .map_err(|e| anyhow::anyhow!("detokenize error: {e}"))?;

    let mut reasoning_parser = ReasoningParser::new();
    let reasoning_result = reasoning_parser.parse_complete(&full_text);

    let mut tool_parser = ToolCallParser::new();
    let (content, tool_calls) = tool_parser.parse_complete(&reasoning_result.normal_text);

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

pub async fn health(State(state): State<Arc<AppState>>) -> StatusCode {
    for worker in state.pool.workers() {
        if worker.is_healthy() {
            return StatusCode::OK;
        }
    }
    StatusCode::SERVICE_UNAVAILABLE
}

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
