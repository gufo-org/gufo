#include "src/models/embedding_gemma2/audio_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "src/core/json.hpp"

namespace model = gufo::models::embedding_gemma2;
namespace {
void Check(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}
std::vector<std::uint8_t> Bytes(const std::string& hex) {
  std::vector<std::uint8_t> result;
  const auto nibble = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
  for (std::size_t i = 0; i < hex.size(); i += 2)
    result.push_back(
        static_cast<std::uint8_t>(nibble(hex[i]) * 16 + nibble(hex[i + 1])));
  return result;
}
}  // namespace
int main() {
  try {
    std::ifstream file(GUFO_EMBEDDING_GEMMA2_AUDIO_DECODER_GOLDEN);
    Check(file.good(), "independent decoder fixture must exist");
    std::ostringstream content;
    content << file.rdbuf();
    const auto fixture = gufo::json::parse(content.str());
    double worst = 0;
    for (const auto& probe : fixture.find("cases")->items()) {
      const auto bytes = Bytes(probe.member_str("encoded_hex"));
      const auto actual = model::DecodeAudio(bytes);
      const auto& expected = probe.find("waveform")->items();
      Check(actual.size() == expected.size(),
            probe.member_str("name") + " decoded length");
      for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(actual[i] - expected[i].as_double());
        worst = std::max(worst, error);
        Check(std::isfinite(actual[i]) && error < 2e-6,
              probe.member_str("name") +
                  " decoded/resampled waveform mismatch: " +
                  std::to_string(error));
      }
      bool rejected = false;
      try {
        (void)model::DecodeAudio(std::span(bytes).first(20));
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      Check(rejected, "truncated container must fail");
      rejected = false;
      try {
        (void)model::DecodeAudio(bytes, [] { return true; });
      } catch (const std::exception&) {
        rejected = true;
      }
      Check(rejected, "cancelled decode must fail");
      Check(model::DecodeAudio(bytes).size() == actual.size(),
            "valid decode recovers after errors");
    }
    // A CBR MP3 with an ID3v2 tag and no Xing/Info header: libsndfile
    // estimates 9456 frames from the file size, but 9216 decode. soundfile
    // 0.14.0 (libsndfile 1.2.2), which librosa reads through, returns 9216.
    const auto mp3 = Bytes(
        "494433040000000000325449543200000006000003636c697000545353450000"
        "000e0000034c61766636312e372e3130330000000000000000000000fff348c4"
        "001c788e6c15583000255af54ea9d53aa754ea9d77b136beb088985c84e868e0"
        "139b627389b4a68118005982f02823107221c961326030180c2c9a60e0200802"
        "0e89c1fe51d39d3e7397f39cbfbba7dd9404c1f3eb0f82018940183ef93040e6"
        "5dfddcb83e0f83e1f04010040100183e0f83e1f04010041c91383e2009208100"
        "681060581dbe7f9832150101fff348c41120a1f66c2d9da0009d0c803e83a623"
        "13e6bb25460d1546c41ea6118bc70c8c2062e081a01e06185818a620a0b2700c"
        "96201a0c5d0d982d182d08523fe25110547285cc2e6ffc991cd1cd2688b116ff"
        "f2648a9013222c458c7fff2e974c8bc5e44ba5d4bffff2f17912e9741a0a82ac"
        "ffe582a1a2dffffff55aaaa000009208a10fe7fff40ce94ca0b6f0c0fff348c4"
        "111998762637de40003c050c094190c204058c208334c5b8ef4e5d0c18fd94ff"
        "0caecb88c7585acc4d091cc00c3f4c1c808cc0380053b958a1e8042ca1adb733"
        "39bbf62ff5bffef27757affde5efe4d6eecfb9bb7f65c43eea8feeeefd3d55ff"
        "ffd4a9414c2418c806ccd11ccddb8d0aacc0650470c0bc08b4c1d31950ca74fe"
        "b8d64b2128c1840784c0d501fff348c42d23e3ba0400dfc470c0c28c0d1cc10f"
        "04dcc0ef0120c0500054c01f00003000d4ce6f24929a6dfeb5d5a19a170f8f41"
        "7bff5cf409bd4d47ab547ab5f4df4f8f57ab57264c52746dfd5a1726d8debab5"
        "7e0f5daa4a0d07a3d9e0b26e9ef46dabfebb63faebf0d3a8c56f76f5800011fb"
        "ffe4ad610ba63c0028a002e60060088601d80fe603901b8606b83626fff348c4"
        "202103be0de8ff44a80ed099464b5de3469f10cd8613e846a6acac2707614669"
        "9f465285e61700c100da0315db107f23785fff7aebab53264a8afff9687db1b7"
        "db1ebaebb7f57abd75cdae37a7fae4cfbe7daad5d5eb93e7da9ae982cf9f6fbd"
        "1aadffa7afd75f8396b53a7615a0000c0667f2eff6858c160030b80002078685"
        "50c27cc86020cf9084e34138fff348c41f1f83be15f4efca88cbb305f0d8c802"
        "a8c28b03c4c0b90504c0460ef8c00905305004b1d005080001634f5c30081a3b"
        "bd75d77cfae3bd35d31946df3fdb7affbefebabd7eb933fdfff334764cdb55a3"
        "747f7fa7a3d74ddad933ff468dd3fd7e3bd7fc4256fde9ea80047e3dffaaed21"
        "24c18031e50c92933f10c03601f4c06f0408c11a0c44c66d7ddccdbcfff348c4"
        "2418e0761948d7f866100cc1050318c8459397ff8ce8b233407cc32000c00aba"
        "7568ebd60d6dbd9baa3bfbd3726e429e9e8f713ead5a7aad4e8d0bb33bb937b7"
        "6aaffdbfeefcfeee95ffffe5235f4b840e1081c090ccc1a180c432b4c84550d2"
        "ff64c60e98b4ccd1149cc1ac06b00df8ee03cffc40d4895033f0400388605810"
        "1bf8a5066c8b9713faf53d5afff348c4431a18de1400efea80b7db3ffda259f4"
        "5d53d9b53ee277d3affce15b37a1766cdf5bdab706d777aadff7fee47d3d0a80"
        "00091427f9fff71eb5b0bc4bf23c2303150ca61334f0d0eeaa531f506333428c"
        "101306d00b63208d4314e7d3040d50a04c4201a8e3d71095dbbcfead7726cfd8"
        "5acf53d9fee26ad796d1f3caffede85dabb117faa95fbf7fe793f4f5fff348c4"
        "5d17487621f4e7fa805547ffffd6705138c20cd054d790e6f4c04301c0c08403"
        "acc14b0b50c7527538cf9f0ec8c1440274c02101f8c112092cc0690334c06100"
        "98c0160014b7cae5faa20c57ed5d35db7db1bf4dabbd391b98bfb16f66cddbbb"
        "576eeaa4d794f925daa9e45df7fffebb91efffffe5771138c380104026220d8c"
        "0f194c302c0c69584ceefe8cfff348c4821a00da1808cfc46e4b99e50c9fa153"
        "8c16d069c0de6d103d56080d526a034381c068900881e1c609cc6608a17d06da"
        "bedeadb5d49fffd752b56df79cb7a3de4d4ea6f4ff9c55dd97b77ef4dcdbdabb"
        "ff23feefd89bf727a1ffffecdaf8439a3b0b01c06120c4d06cc84108cf5234e2"
        "a640cb693ba0d82608f0c29104fcc0e10534c0be0ea0c03f0500c01bfff348c4"
        "9d1a71561400efea8001142800920daf074dfc02038a7d756a6bbe6c5bffe99f"
        "6d76de32d5ff793bbb8a69ea9e2f7a74dfceeedcdbc994bbef67fbbf3e8fa580"
        "0002200408fd7ffd57d94c4b946070418382261d1118d88a66f4d1c63e6631ca"
        "93266a30512608f80982875359fc0c80a831d074040e4c16bd0eda1319ebf722"
        "edec5de8d3d5ffbc9af56bfffff348c4b61a31561400efca886a3e9bfb3d56aa"
        "c4dff7b3fdffbbff5a4c414d45332e3130308000044acb3fbfff943d69ce0a00"
        "0cc0040050c0120050c02a0010c06b0334c0d904bcc2510c7cc2fc5adcc15908"
        "d4c01100b8c0620098c093005cc0540114c05c0110c01800300c00c3d6d6dc71"
        "1b6fdabb77e77f693b91d4e67febd7716ff329fa6feddf55aab0addffff348c4"
        "d017286e221ce7f8807b3fdff9e45fd3ff7e98218730571822888f9813847573"
        "027031311508d312b19b302842933a653a31302223a8436f2a2c61a6595581b8"
        "ce201551058060604ee818ec26173807a1360195db406b61c793c440be5a032c"
        "1480a13c0ca61b030e8afe4fa8ae5c580911018b81806040b008068186c0bf9a"
        "27374202416024000041c028fff348c4ed1c387626155f4800020b4100a07ff3"
        "44e6eaa016dc32d00a0003bc1ea8363e1f8ffd374d4ccb74c4e4173027a13e86"
        "de2911841f20b3bffd542efb5dc62094c56a30c5902e61a02e71991a62e419af"
        "fffb6fb6fc64073c708e3200388701171da3364c0ee1cc26c79ffffffffff1cf"
        "301e8730e90839e7097200892040d224c801a91021e7d5019c80404bfff348c4"
        "ff3e54163c019ea8000c90c914baa36862603998ac78bda020a6abe0a04d3990"
        "a2ac2d394b925e14c171648c3953019501204812264c2a08935a114a16688854"
        "cd2c8914ad5427815054b032581a3c0abc4a1a51e834784af586a7a223c57057"
        "8883bc1576a0ef1286b50347b2a0abb11077582a751110344c414d45332e3130"
        "30aaaaaaaaaaaaaaaaaaaaaafff348c4881cf8da5c17d84800aaaaaaaaaaaaaa"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    const auto decoded = model::DecodeAudio(mp3);
    Check(decoded.size() == 9216, "estimated MP3 length keeps decoded samples");
    double energy = 0;
    for (const float sample : decoded)
      energy += static_cast<double>(sample) * sample;
    Check(std::abs(energy - 56.380433648018936) < 1e-4,
          "estimated MP3 length waveform energy");
    constexpr std::pair<std::size_t, double> kSamples[] = {
        {0, 0.0},
        {1500, -0.09022247046232224},
        {4600, 0.0770723968744278},
        {9215, -0.0002711241249926388}};
    for (const auto& [index, value] : kSamples)
      Check(std::abs(decoded[index] - value) < 2e-6,
            "estimated MP3 length sample " + std::to_string(index));
    std::cout << "official librosa decoder waveforms pass; maximum error "
              << worst << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
