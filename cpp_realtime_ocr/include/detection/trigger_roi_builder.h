#pragma once

#include <cstdint>
#include <vector>

#include "types.h"

namespace trading_monitor::detect {

struct TriggerRoiConfig {
    int topRows = 8;
    float headerSearchFracH = 0.22f;
    int bodyPadX = 6;
    int bodyPadTop = 4;
    int bodyPadBottom = 6;
    int rowMin = 10;
    int rowMax = 28;
    float leftStripFrac = 0.55f;
    float minSeparatorStrength = 1.6f;
    int minSymbolColW = 40;
    int maxSymbolColW = 220;
    int rebuildEveryNFrames = 15;
};

struct TriggerRoiResult {
    bool ok = false;
    ROI tableBody{};
    ROI symbolCol{};
    ROI triggerRoi{};
    int rowHeight = 18;
    float headerBottomScore = 0.f;
    float rowPeriodScore = 0.f;
    float separatorScore = 0.f;
};

class TriggerRoiBuilder {
public:
    TriggerRoiResult build(const std::vector<uint8_t>& gray, int W, int H,
                           const ROI& positionsPanel,
                           const TriggerRoiConfig& cfg) const;
};

} // namespace trading_monitor::detect