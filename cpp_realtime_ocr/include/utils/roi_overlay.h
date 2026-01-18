#pragma once

#include "types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trading_monitor {

struct ROIOverlayHandle {
    void* statePtr = nullptr;
    void* threadHandle = nullptr;
    uint32_t threadId = 0;
};

// Creates a click-through overlay over the target window's client area and draws ROI rectangles.
// ROIs are interpreted in CLIENT coordinates (same space as the interactive ROI selector).
bool startROIOverlay(void* targetHwnd, const std::vector<ROI>& rois, ROIOverlayHandle& outHandle, std::string& errorMessage);

void stopROIOverlay(ROIOverlayHandle& handle);

} // namespace trading_monitor
