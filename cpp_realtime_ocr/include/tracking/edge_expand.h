#pragma once

#include <vector>
#include <cstdint>

#include "types.h"

namespace trading_monitor::track {

struct EdgeExpandConfig {
    int maxExpandLeft = 40;
    int maxExpandRight = 200;
    int maxExpandUp = 20;
    int maxExpandDown = 260;
    float edgeThresh = 20.0f;
    int smoothWindow = 7;
};

// Refine a panel rectangle using simple edge profiles around a header.
ROI expandPanelByEdges(const std::vector<uint8_t>& gray, int W, int H,
                       const ROI& headerRect, const ROI& panelGuess,
                       const EdgeExpandConfig& cfg);

} // namespace trading_monitor::track
