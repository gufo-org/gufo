//! The request being decoded. Nothing reaches C++ until the whole body is
//! accepted; `Draft::emit` then replays it as builder steps.

use crate::error::Outcome;
use crate::host::{self, Host, Op};

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System = host::ROLE_SYSTEM,
    Developer = host::ROLE_DEVELOPER,
    User = host::ROLE_USER,
    Assistant = host::ROLE_ASSISTANT,
    Tool = host::ROLE_TOOL,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Image {
    pub id: u32,
    /// The content length when the image was read.
    pub offset: usize,
    pub low: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Argument {
    pub name: String,
    pub value: String,
    pub is_string: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Call {
    pub id: String,
    pub name: String,
    pub arguments: Vec<Argument>,
}

/// One text or thought part of a message, in the order the client sent them
/// (`ChatMessage::Part`): it ends at `end` bytes of `content` or `thought`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Part {
    pub thought: bool,
    pub end: usize,
}

/// `tokenization::ChatMessage`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub name: String,
    pub thought: String,
    pub tool_call_id: String,
    pub images: Vec<Image>,
    pub calls: Vec<Call>,
    /// The text and thought parts in order, when the content was a list.
    pub parts: Vec<Part>,
}

impl Message {
    pub fn new(role: Role) -> Self {
        Self {
            role,
            content: String::new(),
            name: String::new(),
            thought: String::new(),
            tool_call_id: String::new(),
            images: Vec::new(),
            calls: Vec::new(),
            parts: Vec::new(),
        }
    }

    /// Appends one text part.
    pub fn text_part(&mut self, text: &str) -> Outcome<()> {
        crate::fields::append(&mut self.content, text)?;
        crate::fields::push(
            &mut self.parts,
            Part {
                thought: false,
                end: self.content.len(),
            },
        )
    }

    /// Appends one thought part.
    pub fn thought_part(&mut self, text: &str) -> Outcome<()> {
        crate::fields::append(&mut self.thought, text)?;
        crate::fields::push(
            &mut self.parts,
            Part {
                thought: true,
                end: self.thought.len(),
            },
        )
    }

    /// The parts, when their order says more than the legacy shape (a thought
    /// before the text, each one block at most).
    pub fn ordered_parts(&self) -> &[Part] {
        let legacy = match self.parts.as_slice() {
            [] | [_] => true,
            [first, second] => first.thought && !second.thought,
            _ => false,
        };
        if legacy { &[] } else { &self.parts }
    }
}

/// `tokenization::ChatTool`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: String,
    pub definition: String,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolChoice {
    Auto = host::TOOL_CHOICE_AUTO,
    None = host::TOOL_CHOICE_NONE,
    Required = host::TOOL_CHOICE_REQUIRED,
}

impl ToolChoice {
    pub fn from_raw(raw: i32) -> Option<Self> {
        match raw {
            host::TOOL_CHOICE_AUTO => Some(Self::Auto),
            host::TOOL_CHOICE_NONE => Some(Self::None),
            host::TOOL_CHOICE_REQUIRED => Some(Self::Required),
            _ => None,
        }
    }
}

/// `ReasoningOptions`; `effort` holds a `ReasoningEffort` value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Reasoning {
    pub enabled: Option<bool>,
    pub effort: Option<u8>,
    pub preserve_thinking: Option<bool>,
    pub budget_tokens: Option<u64>,
}

/// The request's tools: the ones it had (the fragments' callers pass them)
/// and the ones decoded, until a forced choice keeps one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tools {
    pub current: Vec<Vec<u8>>,
    pub added: Vec<Tool>,
    pub selected: Option<usize>,
    pub choice: ToolChoice,
    pub choice_changed: bool,
    pub forced_name: Option<String>,
    pub parallel: Option<bool>,
    pub constrained: Option<bool>,
    pub namespaces: Vec<(String, String)>,
}

impl Tools {
    pub fn new(current: Vec<Vec<u8>>, choice: ToolChoice) -> Self {
        Self {
            current,
            added: Vec::new(),
            selected: None,
            choice,
            choice_changed: false,
            forced_name: None,
            parallel: None,
            constrained: None,
            namespaces: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        match self.selected {
            Some(_) => 1,
            None => self.current.len() + self.added.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The names of every tool, the request's own first.
    pub fn names(&self) -> impl Iterator<Item = &[u8]> {
        let all = self
            .current
            .iter()
            .map(Vec::as_slice)
            .chain(self.added.iter().map(|tool| tool.name.as_bytes()));
        all.enumerate()
            .filter(|(index, _)| self.selected.is_none_or(|selected| selected == *index))
            .map(|(_, name)| name)
    }

    /// Keeps only the first tool named `name`, as `tools = {*found}` did.
    pub fn select(&mut self, name: &str) -> bool {
        let Some(found) = self
            .names()
            .position(|candidate| candidate == name.as_bytes())
        else {
            return false;
        };
        // An earlier selection left one tool, which is the one found.
        if self.selected.is_none() {
            self.selected = Some(found);
        }
        true
    }

    pub fn set_choice(&mut self, choice: ToolChoice) {
        self.choice = choice;
        self.choice_changed = true;
    }
}

/// Everything a decode may set; `None` leaves the C++ field as it was.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Draft {
    pub model: Option<String>,
    pub max_tokens: Option<u64>,
    pub stream: Option<bool>,
    pub include_usage: Option<bool>,
    pub cache_prompt: Option<bool>,
    pub cache_persist: Option<bool>,
    pub return_progress: Option<bool>,
    pub add_vision_id: Option<bool>,
    pub reasoning: Reasoning,
    pub tools: Tools,
    pub messages: Vec<Message>,
}

impl Draft {
    pub fn new(reasoning: Reasoning, tools: Tools) -> Self {
        Self {
            model: None,
            max_tokens: None,
            stream: None,
            include_usage: None,
            cache_prompt: None,
            cache_persist: None,
            return_progress: None,
            add_vision_id: None,
            reasoning,
            tools,
            messages: Vec::new(),
        }
    }

    pub fn emit(&self, host: &mut impl Host) -> Outcome<()> {
        if let Some(model) = &self.model {
            host.emit(Op::Model, 0, 0, model.as_bytes())?;
        }
        if let Some(value) = self.max_tokens {
            let value = i64::try_from(value).map_err(|_| crate::error::Failure::Internal)?;
            host.emit(Op::MaxTokens, 0, value, &[])?;
        }
        let flags = [
            (Op::Stream, self.stream),
            (Op::IncludeUsage, self.include_usage),
            (Op::CachePrompt, self.cache_prompt),
            (Op::CachePersist, self.cache_persist),
            (Op::ReturnProgress, self.return_progress),
            (Op::AddVisionId, self.add_vision_id),
        ];
        for (op, value) in flags {
            if let Some(value) = value {
                host.emit(op, 0, i64::from(value), &[])?;
            }
        }
        let reasoning = self.reasoning;
        let unset = -1;
        host.emit(
            Op::ReasoningEnabled,
            0,
            reasoning.enabled.map_or(unset, i64::from),
            &[],
        )?;
        host.emit(
            Op::ReasoningEffort,
            0,
            reasoning.effort.map_or(unset, i64::from),
            &[],
        )?;
        host.emit(
            Op::PreserveThinking,
            0,
            reasoning.preserve_thinking.map_or(unset, i64::from),
            &[],
        )?;
        let budget = match reasoning.budget_tokens {
            None => unset,
            Some(value) => i64::try_from(value).map_err(|_| crate::error::Failure::Internal)?,
        };
        host.emit(Op::BudgetTokens, 0, budget, &[])?;
        self.emit_tools(host)?;
        for message in &self.messages {
            emit_message(host, message)?;
        }
        Ok(())
    }

    fn emit_tools(&self, host: &mut impl Host) -> Outcome<()> {
        let tools = &self.tools;
        for tool in &tools.added {
            host.emit(Op::Tool, 0, 0, tool.name.as_bytes())?;
            host.emit(Op::ToolDescription, 0, 0, tool.description.as_bytes())?;
            host.emit(Op::ToolParameters, 0, 0, tool.parameters.as_bytes())?;
            host.emit(Op::ToolDefinition, 0, 0, tool.definition.as_bytes())?;
        }
        if let Some(index) = tools.selected {
            let index = u32::try_from(index).map_err(|_| crate::error::Failure::Internal)?;
            host.emit(Op::SelectTool, index, 0, &[])?;
        }
        if tools.choice_changed {
            host.emit(Op::ToolChoice, 0, tools.choice as i64, &[])?;
        }
        if let Some(name) = &tools.forced_name {
            host.emit(Op::ForcedToolName, 0, 0, name.as_bytes())?;
        }
        if let Some(value) = tools.parallel {
            host.emit(Op::ParallelToolCalls, 0, i64::from(value), &[])?;
        }
        if let Some(value) = tools.constrained {
            host.emit(Op::ConstrainedTools, 0, i64::from(value), &[])?;
        }
        for (key, value) in &tools.namespaces {
            host.emit(Op::NamespaceKey, 0, 0, key.as_bytes())?;
            host.emit(Op::NamespaceValue, 0, 0, value.as_bytes())?;
        }
        Ok(())
    }
}

pub fn emit_message(host: &mut impl Host, message: &Message) -> Outcome<()> {
    host.emit(Op::Message, 0, message.role as i64, &[])?;
    for (op, text) in [
        (Op::Content, &message.content),
        (Op::Name, &message.name),
        (Op::Thought, &message.thought),
        (Op::ToolCallId, &message.tool_call_id),
    ] {
        if !text.is_empty() {
            host.emit(op, 0, 0, text.as_bytes())?;
        }
    }
    for image in &message.images {
        let op = if image.low { Op::ImageLow } else { Op::Image };
        let offset = i64::try_from(image.offset).map_err(|_| crate::error::Failure::Internal)?;
        host.emit(op, image.id, offset, &[])?;
    }
    for part in message.ordered_parts() {
        let end = i64::try_from(part.end).map_err(|_| crate::error::Failure::Internal)?;
        host.emit(Op::Part, u32::from(part.thought), end, &[])?;
    }
    for call in &message.calls {
        host.emit(Op::Call, 0, 0, call.id.as_bytes())?;
        host.emit(Op::CallName, 0, 0, call.name.as_bytes())?;
        for argument in &call.arguments {
            host.emit(Op::Argument, 0, 0, argument.name.as_bytes())?;
            host.emit(
                Op::ArgumentValue,
                0,
                i64::from(argument.is_string),
                argument.value.as_bytes(),
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str) -> Tool {
        Tool {
            name: name.into(),
            description: String::new(),
            parameters: "{}".into(),
            definition: String::new(),
        }
    }

    #[test]
    fn selection_finds_the_first_tool_across_current_and_added() {
        let mut tools = Tools::new(vec![b"a".to_vec(), b"b".to_vec()], ToolChoice::Auto);
        tools.added.push(tool("b"));
        tools.added.push(tool("c"));
        assert_eq!(tools.len(), 4);
        assert!(tools.select("c"));
        assert_eq!(tools.selected, Some(3));
        assert_eq!(tools.len(), 1);
        assert_eq!(tools.names().collect::<Vec<_>>(), [b"c".as_slice()]);
        assert!(!tools.select("a"));
        let mut again = Tools::new(vec![b"b".to_vec()], ToolChoice::Auto);
        again.added.push(tool("b"));
        assert!(again.select("b"));
        assert_eq!(again.selected, Some(0));
    }
}
