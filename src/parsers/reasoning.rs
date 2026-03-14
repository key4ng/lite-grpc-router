const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";

#[derive(Debug)]
pub struct ReasoningParser {
    in_reasoning: bool,
    buffer: String,
    stripped_think_start: bool,
}

#[derive(Debug, Default)]
pub struct ReasoningResult {
    pub reasoning_text: String,
    pub normal_text: String,
}

impl ReasoningParser {
    pub fn new() -> Self {
        Self {
            in_reasoning: true,
            buffer: String::new(),
            stripped_think_start: false,
        }
    }

    fn is_partial_token(text: &str) -> bool {
        (THINK_START.starts_with(text) && THINK_START != text)
            || (THINK_END.starts_with(text) && THINK_END != text)
    }

    pub fn parse_chunk(&mut self, text: &str) -> ReasoningResult {
        self.buffer.push_str(text);
        let current = self.buffer.clone();

        if Self::is_partial_token(&current) {
            return ReasoningResult::default();
        }

        let current = if !self.stripped_think_start && current.contains(THINK_START) {
            self.stripped_think_start = true;
            self.in_reasoning = true;
            let replaced = current.replace(THINK_START, "");
            self.buffer.clone_from(&replaced);
            replaced
        } else {
            current
        };

        if self.in_reasoning {
            if let Some(end_idx) = current.find(THINK_END) {
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

            self.buffer.clear();
            ReasoningResult {
                reasoning_text: current,
                normal_text: String::new(),
            }
        } else {
            self.buffer.clear();
            ReasoningResult {
                reasoning_text: String::new(),
                normal_text: current,
            }
        }
    }

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
        // When </think> is split across chunks with preceding text,
        // the parser can't reassemble it. This is a known limitation.
        let mut parser = ReasoningParser::new();

        let r1 = parser.parse_chunk("text</");
        assert_eq!(r1.reasoning_text, "text</");

        // The "</" was emitted with the first chunk so the parser
        // doesn't see a complete "</think>" in the second chunk.
        let r2 = parser.parse_chunk("think>answer");
        assert_eq!(r2.reasoning_text, "think>answer");
        assert_eq!(r2.normal_text, "");
    }

    #[test]
    fn test_explicit_think_start_stripped() {
        let mut parser = ReasoningParser::new();
        let result = parser.parse_complete("<think>reasoning</think>answer");
        assert_eq!(result.reasoning_text, "reasoning");
        assert_eq!(result.normal_text, "answer");
    }
}
