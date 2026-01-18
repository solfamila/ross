#pragma once

#include <string>
#include <vector>

#include "types.h"

namespace trading_monitor::detect {

struct SymbolTemplate {
    std::vector<uint8_t> gray;
    int w = 0;
    int h = 0;
};

class SymbolMatcher {
public:
    bool loadSymbolTemplates(const std::string& dir, const std::string& symbol, std::string& err);

    // Returns best NCC score in ROI across loaded templates (>= -1).
    float matchInGrayROI(const std::vector<uint8_t>& frameGray, int frameW, int frameH,
                         const ROI& roi) const;

private:
    std::vector<SymbolTemplate> m_templates;
};

} // namespace trading_monitor::detect
