#pragma once
#include <string>
#include <vector>

namespace trading_monitor::detect {

struct GlyphOCRResult {
    std::string text;
    float score = -1.0f;
};

GlyphOCRResult ocrTickerFromRowGray(const std::vector<uint8_t>& rowGray, int w, int h);

} // namespace trading_monitor::detect
