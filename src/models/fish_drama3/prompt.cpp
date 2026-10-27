#include "src/models/fish_drama3/prompt.hpp"

#include <regex>
#include <stdexcept>

namespace gufo::models::fish_drama3 {
namespace {

const std::regex kSpeakerPattern(R"(<\|speaker:([0-9]+)\|>)");

std::size_t LeadingSpaceBytes(const std::string& text) {
  std::size_t offset = 0;
  while (offset < text.size()) {
    const auto first = static_cast<unsigned char>(text[offset]);
    std::uint32_t code = first;
    std::size_t width = 1;
    if ((first & 0xe0U) == 0xc0U) {
      code = first & 0x1fU;
      width = 2;
    } else if ((first & 0xf0U) == 0xe0U) {
      code = first & 0x0fU;
      width = 3;
    } else if ((first & 0xf8U) == 0xf0U) {
      code = first & 0x07U;
      width = 4;
    }
    if (offset + width > text.size()) {
      break;
    }
    for (std::size_t part = 1; part < width; ++part) {
      code = (code << 6U) |
             (static_cast<unsigned char>(text[offset + part]) & 0x3fU);
    }
    const bool whitespace = (code >= 9 && code <= 13) || code == 0x20 ||
                            code == 0x85 || code == 0xa0 || code == 0x1680 ||
                            (code >= 0x2000 && code <= 0x200a) ||
                            code == 0x2028 || code == 0x2029 ||
                            code == 0x202f || code == 0x205f || code == 0x3000;
    if (!whitespace) {
      break;
    }
    offset += width;
  }
  return offset;
}

std::string TaggedText(const std::string& text, bool reference) {
  const std::size_t leading = LeadingSpaceBytes(text);
  const std::string body = text.substr(leading);
  std::smatch match;
  if (!std::regex_search(body, match, kSpeakerPattern,
                         std::regex_constants::match_continuous)) {
    return "<|speaker:0|> " + text;
  }
  if (!reference) {
    return text;
  }
  return match.str() + text.substr(0, leading) + body.substr(match.length());
}

}  // namespace

void ValidateDrama3Text(const std::string& text, const char* location) {
  if (LeadingSpaceBytes(text) == text.size()) {
    throw std::invalid_argument(std::string(location) + ": text is empty");
  }
  for (const auto* token : {"<|im_start|>", "<|im_end|>", "<|voice|>",
                            "<|text|>", "<|interleave|>", "<|semantic:"}) {
    if (text.find(token) != std::string::npos) {
      throw std::invalid_argument(
          std::string(location) +
          ": structural tokens are reserved for the prompt builder");
    }
  }
  std::size_t offset = 0;
  while ((offset = text.find("<|speaker", offset)) != std::string::npos) {
    const auto body = text.substr(offset);
    std::smatch match;
    if (!std::regex_search(body, match, kSpeakerPattern,
                           std::regex_constants::match_continuous) ||
        match[1].length() > 2 ||
        (match[1].length() > 1 && match[1].str()[0] == 0) ||
        std::stoi(match[1].str()) > 15) {
      throw std::invalid_argument(
          std::string(location) +
          ": speaker tags require N in 0..15 without leading zeroes");
    }
    offset += static_cast<std::size_t>(match.length());
  }
}

void ValidateDrama3Codes(const FishDacCodes& codes,
                         std::int64_t codebook_size) {
  if (codes.codebooks != 10 || codes.frames <= 0 ||
      codes.codes.size() !=
          static_cast<std::size_t>(codes.frames * codes.codebooks)) {
    throw std::runtime_error(
        "Drama 3 codes require a nonempty [10, frames] grid");
  }
  for (const auto code : codes.codes) {
    if (code < 0 || code >= codebook_size) {
      throw std::runtime_error("Drama 3 audio code is outside the codebook");
    }
  }
}

std::vector<Drama3Frame> BuildDrama3Prompt(
    const ModelConfig& config, const Drama3Tokenizer& tokenizer,
    const std::string& text, const std::vector<Drama3Take>& references,
    const std::vector<Drama3Take>& history) {
  std::vector<Drama3Frame> frames;
  const auto append_text = [&](const std::string& segment) {
    for (const auto token : tokenizer.Encode(segment)) {
      Drama3Frame row{};
      row[0] = token;
      frames.push_back(row);
    }
  };
  const auto append_codes = [&](const FishDacCodes& codes) {
    ValidateDrama3Codes(codes, config.tower.codebook_size);
    for (std::int64_t frame = 0; frame < codes.frames; ++frame) {
      Drama3Frame row{};
      for (std::int64_t book = 0; book < 10; ++book) {
        row[static_cast<std::size_t>(1 + book)] =
            codes.codes[static_cast<std::size_t>(book * codes.frames + frame)];
      }
      row[0] = row[1] + config.tower.semantic_begin;
      frames.push_back(row);
    }
  };

  append_text("<|im_start|>system\n");
  append_text(
      references.empty()
          ? "convert the provided text to speech"
          : "convert the provided text to speech reference to the following:");
  for (std::size_t index = 0; index < references.size(); ++index) {
    ValidateDrama3Text(references[index].text, "reference");
    append_text("\n\n[Reference " + std::to_string(index + 1) + "]\nText: " +
                TaggedText(references[index].text, true) + "\nSpeech: ");
    append_codes(references[index].codes);
  }
  append_text("<|im_end|>\n");

  const auto user_turn = [&](const std::string& value) {
    ValidateDrama3Text(value, "input");
    append_text("<|im_start|>user\n");
    append_text(TaggedText(value, false));
    append_text("<|im_end|>\n");
    append_text("<|im_start|>assistant\n<|voice|>");
  };
  for (const auto& take : history) {
    user_turn(take.text);
    append_codes(take.codes);
    append_text("<|im_end|>\n");
  }
  user_turn(text);
  return frames;
}

}  // namespace gufo::models::fish_drama3
