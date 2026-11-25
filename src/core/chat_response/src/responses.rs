//! OpenAI Responses API (`/v1/responses`) streaming and buffered output state machine.

use std::collections::BTreeMap;
use std::fmt::Write;

use gufo_json::{Object, Value};

use crate::chat::{FinishReason, ToolDelta};
use crate::format::{
    dump_json, named_sse_frame, num_i64, num_usize, parse_json, push_json_escaped, push_member,
    set_member,
};
use crate::metrics::Metrics;

pub struct ResponsesConfig {
    pub id: String,
    pub created_at: i64,
    pub model: String,
    pub parallel_tool_calls: bool,
    pub tool_choice: String,
    pub forced_tool_name: String,
    pub tool_definitions_json: Vec<String>,
    pub namespaces: BTreeMap<String, String>,
    pub cache_persist: bool,
    pub top_logprobs: Option<u32>,
}

pub struct ResponsesOutput {
    streaming: bool,
    namespaces: BTreeMap<String, String>,
    cache_persist: bool,
    response: Object,
    item: Object,
    tool: Option<Object>,
    tool_arguments: String,
    text: String,
    logprobs: Vec<Value>,
    stream_logprobs: Vec<Value>,
    sequence: usize,
    output_index: usize,
    active: bool,
    reasoning: bool,
    connected: bool,
    unfinished_tool: bool,
    scratch: String,
}

impl ResponsesOutput {
    pub fn new(config: ResponsesConfig, streaming: bool) -> Self {
        let mut response = Object::new();
        push_member(&mut response, "id", Value::String(config.id));
        push_member(
            &mut response,
            "object",
            Value::String("response".to_string()),
        );
        push_member(&mut response, "created_at", num_i64(config.created_at));
        push_member(&mut response, "model", Value::String(config.model));
        push_member(
            &mut response,
            "status",
            Value::String("in_progress".to_string()),
        );
        push_member(&mut response, "error", Value::Null);
        push_member(&mut response, "incomplete_details", Value::Null);
        push_member(&mut response, "usage", Value::Null);
        push_member(&mut response, "output", Value::Array(Vec::new()));
        push_member(&mut response, "store", Value::Bool(false));
        push_member(
            &mut response,
            "parallel_tool_calls",
            Value::Bool(config.parallel_tool_calls),
        );
        if !config.forced_tool_name.is_empty() {
            let mut tc = Object::new();
            push_member(&mut tc, "type", Value::String("function".to_string()));
            push_member(&mut tc, "name", Value::String(config.forced_tool_name));
            push_member(&mut response, "tool_choice", Value::Object(tc));
        } else {
            push_member(
                &mut response,
                "tool_choice",
                Value::String(config.tool_choice),
            );
        }
        let mut tools = Vec::with_capacity(config.tool_definitions_json.len());
        for def_json in &config.tool_definitions_json {
            if let Some(Value::Object(def)) = parse_json(def_json)
                && let Some(Value::Object(mut func)) = def.get("function").cloned()
            {
                set_member(&mut func, "type", Value::String("function".to_string()));
                tools.push(Value::Object(func));
            }
        }
        push_member(&mut response, "tools", Value::Array(tools));
        if let Some(count) = config.top_logprobs {
            push_member(&mut response, "top_logprobs", num_usize(count as usize));
        }

        Self {
            streaming,
            namespaces: config.namespaces,
            cache_persist: config.cache_persist,
            response,
            item: Object::new(),
            tool: None,
            tool_arguments: String::new(),
            text: String::new(),
            logprobs: Vec::new(),
            stream_logprobs: Vec::new(),
            sequence: 0,
            output_index: 0,
            active: false,
            reasoning: false,
            connected: true,
            unfinished_tool: false,
            scratch: String::with_capacity(256),
        }
    }

    pub fn begin(&mut self, write: &mut dyn FnMut(&str) -> bool) -> bool {
        self.lifecycle("response.created", write) && self.lifecycle("response.in_progress", write)
    }

    pub fn progress(
        &mut self,
        prompt_progress_json: &str,
        write: &mut dyn FnMut(&str) -> bool,
    ) -> bool {
        let mut event = Object::new();
        push_member(
            &mut event,
            "type",
            Value::String("response.in_progress".to_string()),
        );
        push_member(&mut event, "response", Value::Object(self.response.clone()));
        if let Some(progress) = parse_json(prompt_progress_json) {
            push_member(&mut event, "prompt_progress", progress);
        }
        self.emit(event, write)
    }

    pub fn append(
        &mut self,
        text: &str,
        reasoning: bool,
        item_id: &str,
        logprobs_json: &str,
        stream_logprobs_json: &str,
        write: &mut dyn FnMut(&str) -> bool,
    ) -> bool {
        if text.is_empty() {
            return true;
        }
        if !self.close_tool("incomplete", write) {
            return false;
        }
        if !self.active || self.reasoning != reasoning {
            if !self.close_item("completed", write) {
                return false;
            }
            self.reasoning = reasoning;
            self.active = true;
            let mut item = Object::new();
            push_member(&mut item, "id", Value::String(item_id.to_string()));
            push_member(
                &mut item,
                "type",
                Value::String(if reasoning { "reasoning" } else { "message" }.to_string()),
            );
            push_member(
                &mut item,
                "status",
                Value::String("in_progress".to_string()),
            );
            push_member(
                &mut item,
                if reasoning { "summary" } else { "content" },
                Value::Array(Vec::new()),
            );
            if !reasoning {
                push_member(&mut item, "role", Value::String("assistant".to_string()));
            }
            self.item = item;
            let mut added = self.indexed_event("response.output_item.added");
            push_member(&mut added, "item", Value::Object(self.item.clone()));
            if !self.emit(added, write) {
                return false;
            }
            self.text.clear();
            self.logprobs.clear();
            self.stream_logprobs.clear();
            let mut part = self.part_event(if reasoning {
                "response.reasoning_summary_part.added"
            } else {
                "response.content_part.added"
            });
            push_member(&mut part, "part", self.part());
            if !self.emit(part, write) {
                return false;
            }
        }
        self.text.push_str(text);
        if !reasoning && !logprobs_json.is_empty() {
            if let Some(Value::Array(entries)) = parse_json(logprobs_json) {
                self.logprobs.extend(entries);
            }
            if !stream_logprobs_json.is_empty()
                && stream_logprobs_json != "[]"
                && let Some(Value::Array(entries)) = parse_json(stream_logprobs_json)
            {
                self.stream_logprobs.extend(entries);
            }
        }
        if !self.streaming {
            return true;
        }
        if !self.connected {
            return false;
        }
        let seq = self.sequence;
        self.sequence += 1;
        let kind = if reasoning {
            "response.reasoning_summary_text.delta"
        } else {
            "response.output_text.delta"
        };
        let idx_field = if reasoning {
            "summary_index"
        } else {
            "content_index"
        };
        let item_id = member_str(&self.item, "id");
        self.scratch.clear();
        let _ = write!(
            self.scratch,
            "event: {kind}\ndata: {{\"type\":\"{kind}\",\"output_index\":{},\"item_id\":\"",
            self.output_index
        );
        push_json_escaped(&mut self.scratch, item_id);
        let _ = write!(self.scratch, "\",\"{idx_field}\":0,\"delta\":\"");
        push_json_escaped(&mut self.scratch, text);
        self.scratch.push('"');
        if !reasoning {
            self.scratch.push_str(",\"logprobs\":");
            if stream_logprobs_json.is_empty() {
                self.scratch.push_str("[]");
            } else {
                self.scratch.push_str(stream_logprobs_json);
            }
        }
        let _ = write!(self.scratch, ",\"sequence_number\":{seq}}}\n\n");
        self.connected = write(&self.scratch);
        self.connected
    }

    pub fn tool(
        &mut self,
        call: &ToolDelta<'_>,
        fc_id: &str,
        write: &mut dyn FnMut(&str) -> bool,
    ) -> bool {
        if !call.id.is_empty() {
            if !self.close_item("completed", write) {
                return false;
            }
            let mut tool = Object::new();
            push_member(&mut tool, "id", Value::String(fc_id.to_string()));
            push_member(
                &mut tool,
                "type",
                Value::String("function_call".to_string()),
            );
            push_member(&mut tool, "call_id", Value::String(call.id.to_string()));
            push_member(&mut tool, "name", Value::String(call.name.to_string()));
            if let Some(ns) = self.namespaces.get(call.name) {
                push_member(&mut tool, "namespace", Value::String(ns.clone()));
            }
            push_member(&mut tool, "arguments", Value::String(String::new()));
            push_member(
                &mut tool,
                "status",
                Value::String("in_progress".to_string()),
            );
            self.tool_arguments.clear();
            let mut added = self.indexed_event("response.output_item.added");
            push_member(&mut added, "item", Value::Object(tool.clone()));
            self.tool = Some(tool);
            if !self.emit(added, write) {
                return false;
            }
        }
        self.tool_arguments.push_str(call.arguments);
        if !call.arguments.is_empty()
            && let Some(tool) = &self.tool
            && self.streaming
        {
            if !self.connected {
                return false;
            }
            let seq = self.sequence;
            self.sequence += 1;
            let item_id = member_str(tool, "id");
            self.scratch.clear();
            let _ = write!(
                self.scratch,
                "event: response.function_call_arguments.delta\ndata: {{\"type\":\"response.function_call_arguments.delta\",\"output_index\":{},\"item_id\":\"",
                self.output_index
            );
            push_json_escaped(&mut self.scratch, item_id);
            self.scratch.push_str("\",\"delta\":\"");
            push_json_escaped(&mut self.scratch, call.arguments);
            let _ = write!(self.scratch, "\",\"sequence_number\":{seq}}}\n\n");
            self.connected = write(&self.scratch);
            if !self.connected {
                return false;
            }
        }
        !call.complete || self.close_tool("completed", write)
    }

    pub fn complete(
        &mut self,
        finish_reason: FinishReason,
        metrics: &Metrics,
        write: &mut dyn FnMut(&str) -> bool,
    ) -> (String, bool) {
        let limited = finish_reason == FinishReason::Length;
        self.close_tool("incomplete", write);
        self.close_item(if limited { "incomplete" } else { "completed" }, write);
        let status = if limited || self.unfinished_tool {
            "incomplete"
        } else {
            "completed"
        };
        set_member(
            &mut self.response,
            "status",
            Value::String(status.to_string()),
        );
        if limited {
            let mut details = Object::new();
            push_member(
                &mut details,
                "reason",
                Value::String("max_output_tokens".to_string()),
            );
            set_member(
                &mut self.response,
                "incomplete_details",
                Value::Object(details),
            );
        } else if self.unfinished_tool {
            set_member(
                &mut self.response,
                "incomplete_details",
                Value::Object(Object::new()),
            );
        }
        let mut in_details = Object::new();
        push_member(
            &mut in_details,
            "cached_tokens",
            num_usize(metrics.cached_prompt_tokens),
        );
        push_member(
            &mut in_details,
            "cache_write_tokens",
            num_usize(metrics.prefill_tokens),
        );
        let mut out_details = Object::new();
        push_member(
            &mut out_details,
            "reasoning_tokens",
            num_usize(metrics.reasoning_tokens),
        );
        let mut usage = Object::new();
        push_member(&mut usage, "input_tokens", num_usize(metrics.prompt_tokens));
        push_member(
            &mut usage,
            "input_tokens_details",
            Value::Object(in_details),
        );
        push_member(
            &mut usage,
            "output_tokens",
            num_usize(metrics.completion_tokens),
        );
        push_member(
            &mut usage,
            "output_tokens_details",
            Value::Object(out_details),
        );
        push_member(
            &mut usage,
            "total_tokens",
            num_usize(
                metrics
                    .prompt_tokens
                    .saturating_add(metrics.completion_tokens),
            ),
        );
        set_member(&mut self.response, "usage", Value::Object(usage));
        set_member(&mut self.response, "timings", metrics.timings_value());
        set_member(
            &mut self.response,
            "cache_persist",
            Value::Bool(self.cache_persist),
        );
        let ok = self.lifecycle(
            if limited || self.unfinished_tool {
                "response.incomplete"
            } else {
                "response.completed"
            },
            write,
        );
        let body = dump_json(&Value::Object(std::mem::take(&mut self.response)));
        (body, ok)
    }

    pub fn fail(&mut self, message: &str, write: &mut dyn FnMut(&str) -> bool) -> bool {
        set_member(
            &mut self.response,
            "status",
            Value::String("failed".to_string()),
        );
        let mut err = Object::new();
        push_member(&mut err, "code", Value::String("server_error".to_string()));
        push_member(&mut err, "message", Value::String(message.to_string()));
        set_member(&mut self.response, "error", Value::Object(err));
        self.lifecycle("response.failed", write)
    }

    fn close_tool(&mut self, status: &str, write: &mut dyn FnMut(&str) -> bool) -> bool {
        let Some(mut tool) = self.tool.take() else {
            return self.connected;
        };
        if status == "incomplete" {
            self.unfinished_tool = true;
        }
        let mut done = self.indexed_event("response.function_call_arguments.done");
        push_member(
            &mut done,
            "item_id",
            Value::String(member_str(&tool, "id").to_string()),
        );
        push_member(
            &mut done,
            "name",
            Value::String(member_str(&tool, "name").to_string()),
        );
        push_member(
            &mut done,
            "arguments",
            Value::String(self.tool_arguments.clone()),
        );
        if !self.emit(done, write) {
            return false;
        }
        set_member(
            &mut tool,
            "arguments",
            Value::String(self.tool_arguments.clone()),
        );
        set_member(&mut tool, "status", Value::String(status.to_string()));
        let mut closed = self.indexed_event("response.output_item.done");
        push_member(&mut closed, "item", Value::Object(tool.clone()));
        if let Some(Value::Array(output)) = self.response.get_mut("output") {
            output.push(Value::Object(tool));
        }
        self.output_index += 1;
        self.emit(closed, write)
    }

    fn close_item(&mut self, status: &str, write: &mut dyn FnMut(&str) -> bool) -> bool {
        if !self.active {
            return self.connected;
        }
        let mut done = self.part_event(if self.reasoning {
            "response.reasoning_summary_text.done"
        } else {
            "response.output_text.done"
        });
        push_member(&mut done, "text", Value::String(self.text.clone()));
        if !self.reasoning {
            push_member(
                &mut done,
                "logprobs",
                Value::Array(self.stream_logprobs.clone()),
            );
        }
        if !self.emit(done, write) {
            return false;
        }
        let part = self.part();
        let mut part_done = self.part_event(if self.reasoning {
            "response.reasoning_summary_part.done"
        } else {
            "response.content_part.done"
        });
        push_member(&mut part_done, "part", part.clone());
        if !self.emit(part_done, write) {
            return false;
        }
        let key = if self.reasoning { "summary" } else { "content" };
        if let Some(Value::Array(arr)) = self.item.get_mut(key) {
            arr.push(part);
        }
        set_member(&mut self.item, "status", Value::String(status.to_string()));
        let mut item_done = self.indexed_event("response.output_item.done");
        push_member(&mut item_done, "item", Value::Object(self.item.clone()));
        if let Some(Value::Array(output)) = self.response.get_mut("output") {
            output.push(Value::Object(self.item.clone()));
        }
        self.active = false;
        self.output_index += 1;
        self.emit(item_done, write)
    }

    fn indexed_event(&self, kind: &str) -> Object {
        let mut event = Object::new();
        push_member(&mut event, "type", Value::String(kind.to_string()));
        push_member(&mut event, "output_index", num_usize(self.output_index));
        event
    }

    fn part_event(&self, kind: &str) -> Object {
        let mut event = self.indexed_event(kind);
        push_member(
            &mut event,
            "item_id",
            Value::String(member_str(&self.item, "id").to_string()),
        );
        push_member(
            &mut event,
            if self.reasoning {
                "summary_index"
            } else {
                "content_index"
            },
            num_usize(0),
        );
        event
    }

    fn part(&self) -> Value {
        let mut part = Object::new();
        push_member(
            &mut part,
            "type",
            Value::String(
                if self.reasoning {
                    "summary_text"
                } else {
                    "output_text"
                }
                .to_string(),
            ),
        );
        push_member(&mut part, "text", Value::String(self.text.clone()));
        if !self.reasoning {
            push_member(&mut part, "annotations", Value::Array(Vec::new()));
            push_member(&mut part, "logprobs", Value::Array(self.logprobs.clone()));
        }
        Value::Object(part)
    }

    fn lifecycle(&mut self, kind: &str, write: &mut dyn FnMut(&str) -> bool) -> bool {
        let mut event = Object::new();
        push_member(&mut event, "type", Value::String(kind.to_string()));
        push_member(&mut event, "response", Value::Object(self.response.clone()));
        self.emit(event, write)
    }

    fn emit(&mut self, mut event: Object, write: &mut dyn FnMut(&str) -> bool) -> bool {
        if !self.streaming {
            return true;
        }
        if !self.connected {
            return false;
        }
        let seq = self.sequence;
        self.sequence += 1;
        push_member(&mut event, "sequence_number", num_usize(seq));
        let kind = member_str(&event, "type").to_string();
        let frame = named_sse_frame(&kind, &Value::Object(event));
        self.connected = write(&frame);
        self.connected
    }
}

fn member_str<'a>(obj: &'a Object, key: &str) -> &'a str {
    obj.get(key).and_then(Value::as_str).unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses_stream_and_complete() {
        let cfg = ResponsesConfig {
            id: "resp_1".into(),
            created_at: 100,
            model: "qwen".into(),
            parallel_tool_calls: false,
            tool_choice: "none".into(),
            forced_tool_name: String::new(),
            tool_definitions_json: Vec::new(),
            namespaces: BTreeMap::new(),
            cache_persist: true,
            top_logprobs: None,
        };
        let mut out = ResponsesOutput::new(cfg, true);
        let mut frames = Vec::new();
        let mut collect = |s: &str| {
            frames.push(s.to_string());
            true
        };
        assert!(out.begin(&mut collect));
        assert!(out.append("Hello", false, "msg_1", "", "", &mut collect));
        let (body, ok) = out.complete(FinishReason::Stop, &Metrics::default(), &mut collect);
        assert!(ok);
        assert!(body.contains(r#""status":"completed""#));
        assert!(frames[0].starts_with("event: response.created\n"));
        assert!(
            frames
                .last()
                .unwrap()
                .starts_with("event: response.completed\n")
        );
    }
}
