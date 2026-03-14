pub mod reasoning;
pub mod tool;

#[derive(Debug, Default)]
pub struct ParsedChunk {
    pub content: String,
    pub reasoning_content: String,
    pub tool_calls: Vec<ParsedToolCall>,
}

#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}
