#include "staging/models/glm5_next/chat_template.hpp"

#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/families/qwen/text/chat_template.hpp"

namespace glm = gufo::models::glm5_next;
using gufo::tokenization::ChatMessage;
using gufo::tokenization::ChatRole;
using gufo::tokenization::ChatTool;

namespace {

void Require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

// Expected strings are assembled left-to-right from std::string pieces and
// the string_view tag constants; libstdc++ lacks plain string + string_view.
std::string operator+(std::string left, std::string_view right) {
  left.append(right);
  return left;
}

std::string Role(std::string_view name) {
  return std::string(glm::kCtrlOpen) + std::string(name) +
         std::string(glm::kCtrlClose);
}

// The framed header pair: the masked-start token followed by the short
// opener, spelled from raw bytes so no literal special token forms.
const std::string kHeader = std::string("\x5B") + "gMASK" "\x5D" +  // NOLINT
                            std::string("\x3C") + "sop" "\x3E";     // NOLINT
const std::string kEffortHeader =
    kHeader + Role("system") + "Reasoning Effort: Max";
const std::string kGenPrompt = Role("assistant") + std::string(glm::kThinkOpenTag);

ChatMessage Message(ChatRole role, std::string content) {
  ChatMessage message;
  message.role = role;
  message.content = std::move(content);
  return message;
}

void ExpectPrompt(std::span<const ChatMessage> messages,
                  const std::string& expected,
                  const glm::ChatRenderOptions& options = {},
                  bool add_generation_prompt = true) {
  const auto rendered =
      glm::RenderChat(messages, add_generation_prompt, {}, options);
  if (!rendered.has_value()) {
    throw std::runtime_error("RenderChat unexpectedly nullopt");
  }
  if (*rendered != expected) {
    const auto escape = [](std::string_view text) {
      std::string out;
      for (const char c : text) {
        if (c == '\n') {
          out += "\\n";
        } else {
          out += c;
        }
      }
      return out;
    };
    throw std::runtime_error("RenderChat mismatch\n expected: [" +
                             escape(expected) + "\n   got: [" +
                             escape(*rendered) + "]");
  }
}

void TestBasicTurns() {
  const std::vector<ChatMessage> messages{Message(ChatRole::kUser, "Hi")};
  ExpectPrompt(messages,
               kEffortHeader + Role("user") + "Hi" + kGenPrompt);
  ExpectPrompt(messages, kEffortHeader + Role("user") + "Hi", {}, false);
}

void TestReasoningEffort() {
  const std::vector<ChatMessage> messages{Message(ChatRole::kUser, "Hi")};
  glm::ChatRenderOptions low;
  low.reasoning_effort = "low";
  ExpectPrompt(messages,
               kHeader + Role("system") + "Reasoning Effort: Low" +
                   Role("user") + "Hi" + kGenPrompt,
               low);
  glm::ChatRenderOptions bogus;
  bogus.reasoning_effort = "bogus";
  ExpectPrompt(messages, kEffortHeader + Role("user") + "Hi" + kGenPrompt,
               bogus);
}

void TestAssistantReasoning() {
  ChatMessage answered = Message(ChatRole::kAssistant, "A");
  answered.thought = "R";
  const std::vector<ChatMessage> with_thought{
      Message(ChatRole::kUser, "Q"), answered};
  ExpectPrompt(with_thought,
               kEffortHeader + Role("user") + "Q" + Role("assistant") +
                   glm::kThinkOpenTag + "R" + glm::kThinkCloseTag + "A" +
                   kGenPrompt);

  // A literal close-think boundary inside content splits out reasoning.
  ChatMessage split =
      Message(ChatRole::kAssistant,
              std::string(glm::kThinkOpenTag) + "hid" +
                  glm::kThinkCloseTag + " vis ");
  const std::vector<ChatMessage> split_turn{split};
  ExpectPrompt(split_turn,
               kEffortHeader + Role("assistant") + glm::kThinkOpenTag +
                   "hid" + glm::kThinkCloseTag + "vis" + kGenPrompt);

  // Whitespace-only content contributes nothing after the think pair.
  const std::vector<ChatMessage> blank_turn{
      Message(ChatRole::kAssistant, "  \n")};
  ExpectPrompt(blank_turn,
               kEffortHeader + Role("assistant") + glm::kThinkOpenTag +
                   glm::kThinkCloseTag + kGenPrompt);
}

void TestClearThinking() {
  ChatMessage before = Message(ChatRole::kAssistant, "one");
  before.thought = "r1";
  ChatMessage after = Message(ChatRole::kAssistant, "two");
  after.thought = "r2";
  const std::vector<ChatMessage> messages{
      before, Message(ChatRole::kUser, "U"), after};
  glm::ChatRenderOptions clear;
  clear.clear_thinking = true;
  ExpectPrompt(messages,
               kEffortHeader + Role("assistant") + glm::kThinkOpenTag +
                   glm::kThinkCloseTag + "one" + Role("user") + "U" +
                   Role("assistant") + glm::kThinkOpenTag + "r2" +
                   glm::kThinkCloseTag + "two" + kGenPrompt,
               clear);

  // Without any user turn the reference sentinel (-1) clears nothing.
  const std::vector<ChatMessage> no_user{before};
  ExpectPrompt(no_user,
               kEffortHeader + Role("assistant") + glm::kThinkOpenTag + "r1" +
                   glm::kThinkCloseTag + "one" + kGenPrompt,
               clear);
}

ChatMessage::ToolCall MakeCall(std::string id, std::string name) {
  ChatMessage::ToolCall call;
  call.id = std::move(id);
  call.name = std::move(name);
  return call;
}

void TestToolCallSection() {
  ChatMessage::ToolCall call = MakeCall("c1", "f");
  call.arguments.push_back({"k", "v", true});
  ChatMessage asked = Message(ChatRole::kAssistant, "");
  asked.tool_calls.push_back(call);
  const std::vector<ChatMessage> messages{Message(ChatRole::kUser, "Q"),
                                          asked};
  ExpectPrompt(messages,
               kEffortHeader + Role("user") + "Q" + Role("assistant") +
                   glm::kThinkOpenTag + glm::kThinkCloseTag +
                   // One newline opens the section, one stands between the
                   // name and its arguments, one closes the section.
                   "\n" + glm::kToolCallOpenTag + "f" + "\n" +
                   glm::kArgKeyOpenTag + "k" + glm::kArgKeyCloseTag +
                   glm::kArgValueOpenTag + "v" + glm::kArgValueCloseTag +
                   glm::kToolCallCloseTag + "\n" + kGenPrompt);
}

void TestToolResponseRun() {
  ChatMessage asked = Message(ChatRole::kAssistant, "");
  asked.tool_calls.push_back(MakeCall("a", "fa"));
  asked.tool_calls.push_back(MakeCall("b", "fb"));
  ChatMessage answer_b = Message(ChatRole::kTool, "B");
  answer_b.tool_call_id = "b";
  ChatMessage answer_a = Message(ChatRole::kTool, "A");
  answer_a.tool_call_id = "a";

  // Matched ids reorder the run to the call order, with no newlines.
  const std::vector<ChatMessage> sorted{asked, answer_b, answer_a};
  ExpectPrompt(sorted,
               kEffortHeader + Role("assistant") + glm::kThinkOpenTag +
                   glm::kThinkCloseTag + "\n" + glm::kToolCallOpenTag + "fa" +
                   "\n" + glm::kToolCallCloseTag +
                   glm::kToolCallOpenTag + "fb" + "\n" +
                   glm::kToolCallCloseTag + "\n" + Role("observation") +
                   glm::kToolResponseOpenTag + "A" +
                   glm::kToolResponseCloseTag + glm::kToolResponseOpenTag +
                   "B" + glm::kToolResponseCloseTag + kGenPrompt);

  // An unmatched answer keeps the original tool order.
  ChatMessage answer_x = Message(ChatRole::kTool, "X");
  answer_x.tool_call_id = "zz";
  const std::vector<ChatMessage> unsorted{asked, answer_b, answer_x};
  const std::string calls =
      "\n" + glm::kToolCallOpenTag + "fa" + "\n" + glm::kToolCallCloseTag +
      glm::kToolCallOpenTag + "fb" + "\n" + glm::kToolCallCloseTag +
      "\n";
  ExpectPrompt(unsorted,
               kEffortHeader + Role("assistant") + glm::kThinkOpenTag +
                   glm::kThinkCloseTag + calls + Role("observation") +
                   glm::kToolResponseOpenTag + "B" +
                   glm::kToolResponseCloseTag + glm::kToolResponseOpenTag +
                   "X" + glm::kToolResponseCloseTag + kGenPrompt);
}

void TestToolSchemasFailClosed() {
  const std::vector<ChatMessage> messages{Message(ChatRole::kUser, "Hi")};
  ChatTool tool;
  tool.name = "lookup";
  const std::vector<ChatTool> tools{tool};
  if (glm::RenderChat(messages, true, tools).has_value()) {
    throw std::runtime_error("tool schemas must render nullopt");
  }
}

void TestEosHidden() {
  const std::string with_eos =
      std::string("a") + std::string(glm::kGlmEosSpelling) + "b";
  ChatMessage thought = Message(ChatRole::kAssistant, "tail");
  thought.thought = with_eos;
  const std::vector<ChatMessage> messages{
      Message(ChatRole::kUser, with_eos), thought};
  const auto rendered = glm::RenderChat(messages);
  Require(rendered.has_value(), "render failed");
  if (rendered->find(glm::kGlmEosSpelling) != std::string::npos) {
    throw std::runtime_error("stop spelling leaked into the prompt");
  }
  const std::string expected_user =
      Role("user") + "a" + std::string(glm::kEosPlaceholder) + "b";
  if (rendered->find(expected_user) == std::string::npos) {
    throw std::runtime_error("user placeholder missing");
  }
  if (rendered->find(std::string(glm::kThinkOpenTag) + "a" +
                     std::string(glm::kEosPlaceholder) + "b" +
                     std::string(glm::kThinkCloseTag)) ==
      std::string::npos) {
    throw std::runtime_error("reasoning placeholder missing");
  }
}

void TestSystemAndDeveloper() {
  const std::vector<ChatMessage> messages{
      Message(ChatRole::kSystem, "S"), Message(ChatRole::kDeveloper, "D"),
      Message(ChatRole::kUser, "U")};
  ExpectPrompt(messages,
               kEffortHeader + Role("system") + "S" + Role("user") + "U" +
                   kGenPrompt);
}

void Run(const char* name, void (*test)()) {
  try {
    test();
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string(name) + ": " + error.what());
  }
}

}  // namespace

int main() {
  try {
    Run("TestBasicTurns", TestBasicTurns);
    Run("TestReasoningEffort", TestReasoningEffort);
    Run("TestAssistantReasoning", TestAssistantReasoning);
    Run("TestClearThinking", TestClearThinking);
    Run("TestToolCallSection", TestToolCallSection);
    Run("TestToolResponseRun", TestToolResponseRun);
    Run("TestToolSchemasFailClosed", TestToolSchemasFailClosed);
    Run("TestEosHidden", TestEosHidden);
    Run("TestSystemAndDeveloper", TestSystemAndDeveloper);
    std::cout << "glm5_next chat_template: OK" << std::endl;
  } catch (const std::exception& error) {
    std::cerr << "glm5_next chat_template FAILED: " << error.what()
              << std::endl;
    return 1;
  }
  return 0;
}
