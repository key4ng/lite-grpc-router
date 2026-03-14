use regex::Regex;

use super::ParsedToolCall;

const CALLS_BEGIN: &str = "<\u{ff5c}tool\u{2581}calls\u{2581}begin\u{ff5c}>";
const CALL_BEGIN: &str = "<\u{ff5c}tool\u{2581}call\u{2581}begin\u{ff5c}>";
const CALL_END: &str = "<\u{ff5c}tool\u{2581}call\u{2581}end\u{ff5c}>";
const CALLS_END: &str = "<\u{ff5c}tool\u{2581}calls\u{2581}end\u{ff5c}>";
const TOOL_SEP: &str = "<\u{ff5c}tool\u{2581}sep\u{ff5c}>";

#[derive(Debug)]
pub struct ToolCallParser {
    buffer: String,
    complete_re: Regex,
    detail_re: Regex,
    tool_id_counter: u32,
}

impl ToolCallParser {
    pub fn new() -> Self {
        let complete_pattern = format!(r"(?s){CALL_BEGIN}.*?{CALL_END}");
        let detail_pattern =
            format!(r"(?s){CALL_BEGIN}(.*?){TOOL_SEP}(.*?)\n```json\n(.*?)\n```{CALL_END}");

        Self {
            buffer: String::new(),
            complete_re: Regex::new(&complete_pattern).expect("valid regex"),
            detail_re: Regex::new(&detail_pattern).expect("valid regex"),
            tool_id_counter: 0,
        }
    }

    pub fn parse_complete(&mut self, text: &str) -> (String, Vec<ParsedToolCall>) {
        let Some(calls_begin_idx) = text.find(CALLS_BEGIN) else {
            return (text.to_string(), vec![]);
        };

        let normal_text = text[..calls_begin_idx].to_string();
        let tool_section = &text[calls_begin_idx..];
        let tool_calls = self.extract_tool_calls(tool_section);

        (normal_text, tool_calls)
    }

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

    pub fn parse_chunk(&mut self, text: &str) -> (String, Vec<ParsedToolCall>) {
        self.buffer.push_str(text);

        let Some(calls_begin_idx) = self.buffer.find(CALLS_BEGIN) else {
            if self.buffer.len() > CALLS_BEGIN.len() {
                let emit_end = self.buffer.len() - CALLS_BEGIN.len();
                let emit = self.buffer[..emit_end].to_string();
                self.buffer = self.buffer[emit_end..].to_string();
                return (emit, vec![]);
            }
            return (String::new(), vec![]);
        };

        let normal = self.buffer[..calls_begin_idx].to_string();

        let tool_section = self.buffer[calls_begin_idx..].to_string();
        if let Some(calls_end_idx) = tool_section.find(CALLS_END) {
            let full_section = &tool_section[..calls_end_idx + CALLS_END.len()];
            let calls = self.extract_tool_calls(full_section);

            let after = &tool_section[calls_end_idx + CALLS_END.len()..];
            self.buffer = after.to_string();

            return (normal, calls);
        }

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
        format!("{CALL_BEGIN}function{TOOL_SEP}{name}\n```json\n{args}\n```{CALL_END}")
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
        assert!(c1.is_empty());

        let (n2, c2) = parser.parse_chunk(&full[mid..]);
        let combined_normal = format!("{n1}{n2}");
        assert!(combined_normal.contains("hello"));
        assert_eq!(c2.len(), 1);
        assert_eq!(c2[0].name, "func");
    }
}
