#pragma once

#include <cstdint>
#include <vector>

#include "types.h"

namespace trading_monitor::track {

struct TrackerConfig {
    int searchRadiusPx = 40;
    float minTrackScore = 0.60f;
    int maxSearchW = 480;          // downscale search width (0=disable)
    int reexpandEveryNFrames = 15; // edge refine cadence (0=disable)
    std::vector<float> scaleMultipliers = {0.97f, 1.00f, 1.03f};
    float minScale = 0.60f;
    float maxScale = 1.40f;
};

struct HeaderTemplate {
    std::vector<uint8_t> gray;
    int w = 0;
    int h = 0;
};

struct TrackResult {
    bool ok = false;
    float score = 0.0f;
    ROI headerRect{};
    ROI panelRect{};
};

class PanelTracker {
public:
    void init(const HeaderTemplate& headerTpl, const ROI& initialHeaderRect, const ROI& initialPanelRect);
    bool isInitialized() const { return m_inited; }

    TrackResult update(const std::vector<uint8_t>& frameGray, int frameW, int frameH,
                       const TrackerConfig& cfg, uint64_t frameIndex, bool enableEdgeExpansion);

    const ROI& lastPanel() const { return m_lastPanel; }
    const ROI& lastHeader() const { return m_lastHeader; }

private:
    bool m_inited = false;
    HeaderTemplate m_tpl{};
    ROI m_lastHeader{};
    ROI m_lastPanel{};
    float m_lastScore = 0.0f;
    uint64_t m_lastExpandedFrame = 0;
    float m_lastScale = 1.0f;
};

} // namespace trading_monitor::track
