#pragma once

#include "types.h"

#include <string>

namespace trading_monitor {

// Shows a temporary overlay over the given HWND's client area.
// User click-drags to select a rectangle. ESC cancels.
// Returns true on success and fills outRoi.{x,y,w,h} in client coordinates.
bool selectROIInteractive(void* hwnd, ROI& outRoi, std::string& errorMessage);

} // namespace trading_monitor
