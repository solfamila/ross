#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "types.h"

namespace trading_monitor::detect {

struct PanelTemplates {
    // Grayscale templates (8-bit)
    std::vector<uint8_t> positionsHdr;
    int positionsW = 0;
    int positionsH = 0;

    std::vector<uint8_t> orderHdr;
    int orderW = 0;
    int orderH = 0;

    std::vector<uint8_t> quoteHdr;
    int quoteW = 0;
    int quoteH = 0;
};

struct FoundPanels {
    bool hasPositions = false;
    bool hasOrder = false;
    bool hasQuote = false;

    ROI positionsPanel{};
    ROI orderPanel{};
    ROI quotePanel{};

    ROI positionsHeader{};
    ROI orderHeader{};
    ROI quoteHeader{};

    float scorePositions = 0.0f;
    float scoreOrder = 0.0f;
    float scoreQuote = 0.0f;
};

struct PanelFinderConfig {
    float hdrThreshold = 0.55f;

    // Downscale the search image to speed up matching (0 = disable)
    float maxSearchW = 640.0f;

    // Multi-scale template matching factors
    std::vector<float> scales = {0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};

    // Initial panel size guess (pixels at full-res)
    int defaultPanelW = 520;
    int defaultPanelH = 360;

    // Padding around the header when expanding
    int panelPadX = 6;
    int panelPadY = 4;

    // Edge-based refinement (heuristic)
    int edgeSearchPad = 24;
    float edgeMinStrength = 12.0f; // row/col gradient energy threshold (tune)
};

class PanelFinder {
public:
    bool loadTemplates(const std::string& positionsHdrPath,
                       const std::string& orderHdrPath,
                       const std::string& quoteHdrPath,
                       std::string& err);

    FoundPanels findPanelsFromBGRA(const uint8_t* bgra, int w, int h, int strideBytes,
                                   const PanelFinderConfig& cfg) const;

    // Use precomputed grayscale image (size = w*h)
    FoundPanels findPanelsFromGray(const uint8_t* gray, int w, int h,
                                   const PanelFinderConfig& cfg) const;

private:
    PanelTemplates m_tpl;
};

} // namespace trading_monitor::detect
