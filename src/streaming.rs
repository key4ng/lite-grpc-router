use std::sync::Arc;

use axum::response::sse::{Event, Sse};
use futures::stream::{self, Stream, StreamExt};
use tonic::Streaming;

use crate::proto::sglang::{GenerateResponse, generate_response};
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

    // Wrap grpc_stream in futures::StreamExt-compatible stream
    let grpc_stream = futures::stream::unfold(grpc_stream, |mut s| async {
        match s.message().await {
            Ok(Some(msg)) => Some((Ok(msg), s)),
            _ => None,
        }
    });

    let stream = grpc_stream.map(move |result: Result<GenerateResponse, tonic::Status>| {
        match result {
            Ok(response) => {
                match response.response {
                    Some(generate_response::Response::Chunk(chunk)) => {
                        let text = tokenizer
                            .decode(&chunk.token_ids, true)
                            .unwrap_or_default();

                        if text.is_empty() {
                            return None;
                        }

                        let reasoning_result = reasoning_parser.parse_chunk(&text);

                        let (content, tool_calls) = if !reasoning_result.normal_text.is_empty() {
                            tool_parser.parse_chunk(&reasoning_result.normal_text)
                        } else {
                            (String::new(), vec![])
                        };

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
    .filter_map(|opt: Option<Event>| async { opt })
    .chain(stream::once(async {
        Event::default().data("[DONE]")
    }))
    .map(Ok);

    Sse::new(stream)
}
