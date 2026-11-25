//! Decoder tests through a recording host. tests/cli/chat_request_test.cpp
//! compares every API with the frozen C++ decoders.

use super::*;
use crate::draft::{Message, Role};
use crate::error::{Outcome, Rejection};
use crate::host::{HostField, Op};

#[derive(Default)]
struct Fake {
    images: Vec<Vec<u8>>,
    fail_url: Option<&'static str>,
    reject_field: Option<HostField>,
    fields: Vec<(HostField, i32, Option<String>)>,
    schemas: Vec<String>,
    events: Vec<(Op, u32, i64, Vec<u8>)>,
}

impl Host for Fake {
    fn serves_model(&mut self, name: &str) -> Outcome<bool> {
        Ok(name == "m")
    }
    fn read_image(&mut self, url: &[u8]) -> Outcome<u32> {
        if self.fail_url.is_some_and(|fail| fail.as_bytes() == url) {
            return Err(Failure::Invalid("bad image".into()));
        }
        self.images.push(url.to_vec());
        Ok(u32::try_from(self.images.len() - 1).unwrap())
    }
    fn compile_schema(&mut self, json: &[u8]) -> Outcome<()> {
        self.schemas.push(String::from_utf8(json.to_vec()).unwrap());
        Ok(())
    }
    fn field(&mut self, field: HostField, argument: i32, json: Option<&[u8]>) -> Outcome<()> {
        let json = json.map(|json| String::from_utf8(json.to_vec()).unwrap());
        self.fields.push((field, argument, json));
        if self.reject_field == Some(field) {
            return Err(Failure::HostRejected);
        }
        Ok(())
    }
    fn emit(&mut self, op: Op, index: u32, integer: i64, text: &[u8]) -> Outcome<()> {
        self.events.push((op, index, integer, text.to_vec()));
        Ok(())
    }
}

fn run_api(api: Api, body: &str, host: &mut Fake) -> Result<Draft, Failure> {
    decode(api, body.as_bytes(), &Defaults::default(), host)
}

fn rejected(status: i32, code: &'static str, message: &str) -> Failure {
    Failure::Rejected(Rejection {
        status,
        code,
        message: message.into(),
    })
}

#[test]
fn chat_decodes_in_order_and_calls_the_helpers() {
    let mut host = Fake::default();
    let draft = run_api(
        Api::Chat,
        r#"{"model":"m","messages":[{"role":"user","content":[
            {"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"u","detail":"low"}},
            {"type":"text","text":"b"}]}],
            "tools":[{"type":"function","function":{"name":"f","strict":true}}],
            "tool_choice":{"type":"function","function":{"name":"f"}},"max_tokens":2.0,
            "temperature":1.50,"input":[1]}"#,
        &mut host,
    )
    .unwrap();
    assert_eq!(draft.model.as_deref(), Some("m"));
    assert_eq!(draft.max_tokens, Some(2));
    let message = &draft.messages[0];
    assert_eq!(message.content, "ab");
    assert_eq!(message.images[0].offset, 1);
    assert!(message.images[0].low);
    assert_eq!(draft.tools.selected, Some(0));
    assert_eq!(draft.tools.parallel, Some(false));
    assert_eq!(
        draft.tools.added[0].parameters,
        r#"{"type":"object","properties":{},"required":[],"additionalProperties":false}"#
    );
    assert_eq!(
        draft.tools.added[0].definition,
        r#"{"type":"function","function":{"name":"f","strict":true,"parameters":{"type":"object","properties":{},"required":[],"additionalProperties":false}}}"#
    );
    assert_eq!(host.schemas.len(), 1);
    let fields: Vec<_> = host.fields.iter().map(|(field, _, _)| *field).collect();
    assert_eq!(
        fields,
        [
            HostField::ResponseFormat,
            HostField::Stop,
            HostField::Sampling,
            HostField::Logprobs
        ]
    );
    assert_eq!(host.fields[0].2, None);
    assert_eq!(
        host.fields[2].2.as_deref(),
        Some(
            r#"{"model":"m","messages":null,"tools":null,"tool_choice":{"type":"function","function":{"name":"f"}},"max_tokens":2.0,"temperature":1.50,"input":null}"#
        )
    );
}

#[test]
fn chat_rejections_keep_their_messages() {
    let mut host = Fake::default();
    let cases = [
        (
            "[",
            rejected(
                400,
                "parse_error",
                "JSON parse error: unexpected end of input",
            ),
        ),
        (
            "[]",
            rejected(400, "invalid_body", "request body must be a JSON object"),
        ),
        (
            r#"{"model":"x","messages":[]}"#,
            rejected(
                404,
                "model_not_found",
                "model 'x' is not served by this process",
            ),
        ),
        (
            r#"{"model":"m","messages":[]}"#,
            rejected(
                400,
                "missing_messages",
                "'messages' must be a non-empty array",
            ),
        ),
        (
            r#"{"model":"m","messages":[{"role":"assistant","content":[{"type":"image_url","image_url":{"url":"u"}}]}]}"#,
            rejected(
                400,
                "invalid_messages",
                "image_url requires a user or tool message and a string URL",
            ),
        ),
        (
            r#"{"model":"m","messages":[{"role":"user"}],"max_tokens":0}"#,
            rejected(
                400,
                "invalid_max_tokens",
                "'max_tokens' must be a positive integer",
            ),
        ),
        (
            r#"{"model":"m","messages":[{"role":"user"}],"tool_choice":"required"}"#,
            rejected(
                400,
                "invalid_tool_choice",
                "'tool_choice' cannot be required without tools",
            ),
        ),
    ];
    for (body, expected) in cases {
        match run_api(Api::Chat, body, &mut host) {
            Err(Failure::Rejected(rejection)) if matches!(&expected, Failure::Rejected(e) if e.code == "parse_error") =>
            {
                assert_eq!(rejection.code, "parse_error");
                assert!(rejection.message.starts_with("JSON parse error: "));
            }
            other => assert_eq!(other.err(), Some(expected), "{body}"),
        }
    }
    host.fail_url = Some("u");
    let failed = run_api(
        Api::Chat,
        r#"{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"u"}}]}]}"#,
        &mut host,
    );
    assert_eq!(
        failed.err(),
        Some(rejected(400, "invalid_messages", "bad image"))
    );
}

#[test]
fn responses_defer_system_items_and_merge_turns() {
    let mut host = Fake::default();
    let draft = run_api(
        Api::Responses,
        r#"{"instructions":"sys","input":[
            {"type":"reasoning","summary":[{"type":"summary_text","text":"t"}]},
            {"role":"developer","content":"d"},
            {"role":"assistant","content":[{"type":"output_text","text":"answer"}]},
            {"type":"function_call","call_id":"c","name":"f","arguments":"{}"},
            {"type":"function_call_output","call_id":"c","output":[{"type":"input_image","image_url":"u"}]}],
            "stream":true,"model":"m"}"#,
        &mut host,
    )
    .unwrap();
    let roles: Vec<Role> = draft.messages.iter().map(|m: &Message| m.role).collect();
    assert_eq!(
        roles,
        [Role::System, Role::Assistant, Role::Developer, Role::Tool]
    );
    assert_eq!(draft.messages[1].content, "answer");
    assert_eq!(draft.messages[1].calls[0].name, "f");
    assert_eq!(draft.messages[3].images.len(), 1);
    assert_eq!(draft.stream, Some(true));
    assert_eq!(
        run_api(Api::Responses, r#"{"input":""}"#, &mut host).err(),
        Some(rejected(
            400,
            "invalid_request",
            "'input' must contain text, message items with text/images, reasoning items, function calls or function outputs"
        ))
    );
    host.reject_field = Some(HostField::ResponsesOptions);
    assert_eq!(
        run_api(Api::Responses, r#"{"input":"x"}"#, &mut host).err(),
        Some(Failure::HostRejected)
    );
}

#[test]
fn messages_split_tool_turns() {
    let mut host = Fake::default();
    let draft = run_api(
        Api::Messages,
        r#"{"system":[{"type":"text","text":"s"}],"thinking":{"type":"enabled","budget_tokens":9},
            "tools":[{"name":"f","input_schema":{"type":"object"}}],
            "tool_choice":{"type":"any","disable_parallel_tool_use":true},
            "messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"t"},
            {"type":"tool_use","id":"c","name":"f","input":{"x":1.0}}]},
            {"role":"user","content":[{"type":"text","text":"before"},
            {"type":"tool_result","tool_use_id":"c","content":"r"},{"type":"text","text":"after"}]}]}"#,
        &mut host,
    )
    .unwrap();
    let roles: Vec<Role> = draft.messages.iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        [
            Role::System,
            Role::Assistant,
            Role::User,
            Role::Tool,
            Role::User
        ]
    );
    // Inputs are written canonically before they are read back.
    assert_eq!(draft.messages[1].calls[0].arguments[0].value, "1");
    assert_eq!(draft.reasoning.enabled, Some(true));
    assert_eq!(draft.reasoning.budget_tokens, Some(9));
    assert_eq!(draft.tools.parallel, Some(false));
    assert_eq!(
        run_api(
            Api::Messages,
            r#"{"output_config":{"x\u0000y":1}}"#,
            &mut host
        )
        .err(),
        Some(rejected(
            400,
            "invalid_request",
            "unsupported output_config member: x"
        ))
    );
}

#[test]
fn fragments_start_from_the_request() {
    let mut host = Fake::default();
    let defaults = Defaults {
        tool_names: vec![b"a".to_vec()],
        tool_choice: ToolChoice::Required,
        ..Defaults::default()
    };
    let draft = decode(
        Api::ResponseControls,
        br#"{"tool_choice":{"type":"function","name":"a"}}"#,
        &defaults,
        &mut host,
    )
    .unwrap();
    assert_eq!(draft.tools.selected, Some(0));
    assert_eq!(draft.tools.constrained, Some(true));
    let empty = Defaults {
        tool_choice: ToolChoice::Required,
        ..Defaults::default()
    };
    assert_eq!(
        decode(
            Api::AnthropicTools,
            br#"{"tool_choice":{"type":"auto"}}"#,
            &empty,
            &mut host
        )
        .err(),
        None
    );
    assert!(decode(Api::ResponseControls, b"{}", &empty, &mut host).is_err());
}

#[test]
fn accepted_drafts_emit_their_steps() {
    let mut host = Fake::default();
    let draft = run_api(
        Api::ResponseItem,
        r#"{"role":"user","content":"hi","name":"n"}"#,
        &mut host,
    )
    .unwrap();
    draft.emit(&mut host).unwrap();
    let ops: Vec<Op> = host.events.iter().map(|event| event.0).collect();
    assert_eq!(
        ops,
        [
            Op::ReasoningEnabled,
            Op::ReasoningEffort,
            Op::PreserveThinking,
            Op::BudgetTokens,
            Op::Message,
            Op::Content,
            Op::Name
        ]
    );
    assert_eq!(host.events[4].2, Role::User as i64);
}

#[test]
#[allow(unsafe_code)]
fn caller_errors_are_internal() {
    let defaults = GufoChatDefaults {
        model_id: GufoChatBytes {
            data: std::ptr::null(),
            len: 0,
        },
        require_model: 1,
        reasoning: GufoChatReasoning {
            enabled: -1,
            effort: -1,
            preserve_thinking: -1,
            has_budget: 0,
            budget_tokens: 0,
        },
        tool_names: std::ptr::null(),
        tool_name_count: 0,
        tool_choice: 0,
    };
    let empty = GufoChatHost {
        serves_model: None,
        read_image: None,
        compile_schema: None,
        host_field: None,
        emit: None,
        reject: None,
    };
    let body = b"{}";
    // SAFETY: every pointer is valid or null; no callback runs.
    unsafe {
        let host: *const GufoChatHost = &empty;
        let context = std::ptr::null_mut();
        let call = |api, data: *const u8, len, defaults: *const GufoChatDefaults, host| {
            gufo_chat_decode(api, data, len, defaults, host, context)
        };
        assert_eq!(
            call(0, body.as_ptr(), 2, &defaults, host),
            GufoChatDecode::Internal
        );
        assert_eq!(
            call(9, body.as_ptr(), 2, &defaults, host),
            GufoChatDecode::Internal
        );
        assert_eq!(
            call(0, std::ptr::null(), 2, &defaults, host),
            GufoChatDecode::Internal
        );
        assert_eq!(
            call(0, body.as_ptr(), 2, std::ptr::null(), host),
            GufoChatDecode::Internal
        );
        let bad = GufoChatDefaults {
            tool_choice: 7,
            ..defaults
        };
        assert_eq!(
            call(0, body.as_ptr(), 2, &bad, host),
            GufoChatDecode::Internal
        );
    }
}
