#include "tracking/edge_expand.h"

#include <algorithm>
#include <cmath>

namespace trading_monitor::track {

static inline int clampi(int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); }

static ROI clampROI(const ROI& r, int W, int H) {
    ROI out = r;
    out.x = clampi(out.x, 0, W - 1);
    out.y = clampi(out.y, 0, H - 1);
    out.w = std::max(1, std::min(out.w, W - out.x));
    out.h = std::max(1, std::min(out.h, H - out.y));
    return out;
}

static void smooth1D(std::vector<float>& v, int win) {
    if (win <= 1 || (int)v.size() < win) return;
    std::vector<float> out(v.size(), 0.0f);
    int r = win / 2;
    for (int i = 0; i < (int)v.size(); ++i) {
        float s = 0.0f;
        int c = 0;
        for (int k = -r; k <= r; ++k) {
            int j = i + k;
            if (j < 0 || j >= (int)v.size()) continue;
            s += v[(size_t)j];
            c++;
        }
        out[(size_t)i] = (c ? s / (float)c : 0.0f);
    }
    v.swap(out);
}

static float avg(const std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    double s = 0.0;
    for (float x : v) s += x;
    return (float)(s / (double)v.size());
}

static std::vector<float> verticalEdgeProfile(const std::vector<uint8_t>& g, int W, const ROI& r) {
    std::vector<float> prof((size_t)r.w, 0.0f);
    for (int y = r.y + 1; y < r.y + r.h - 1; ++y) {
        const uint8_t* row = g.data() + (size_t)y * (size_t)W;
        for (int x = r.x + 1; x < r.x + r.w - 1; ++x) {
            int gx = (int)row[x + 1] - (int)row[x - 1];
            prof[(size_t)(x - r.x)] += (float)std::abs(gx);
        }
    }
    return prof;
}

static std::vector<float> horizontalEdgeProfile(const std::vector<uint8_t>& g, int W, const ROI& r) {
    std::vector<float> prof((size_t)r.h, 0.0f);
    for (int y = r.y + 1; y < r.y + r.h - 1; ++y) {
        const uint8_t* up = g.data() + (size_t)(y - 1) * (size_t)W;
        const uint8_t* dn = g.data() + (size_t)(y + 1) * (size_t)W;
        float acc = 0.0f;
        for (int x = r.x + 1; x < r.x + r.w - 1; ++x) {
            int gy = (int)dn[x] - (int)up[x];
            acc += (float)std::abs(gy);
        }
        prof[(size_t)(y - r.y)] = acc;
    }
    return prof;
}

static int findEdgePeak(const std::vector<float>& prof, int startIdx, int dir, float thresh) {
    int n = (int)prof.size();
    int i = clampi(startIdx, 0, n - 1);
    for (; i >= 0 && i < n; i += dir) {
        if (prof[(size_t)i] >= thresh) return i;
    }
    return -1;
}

ROI expandPanelByEdges(const std::vector<uint8_t>& gray, int W, int H,
                       const ROI& headerRect, const ROI& panelGuess,
                       const EdgeExpandConfig& cfg) {
    if ((int)gray.size() < W * H) return panelGuess;
    ROI guess = clampROI(panelGuess, W, H);

    ROI search;
    search.name = "panel_edge_search";
    search.x = guess.x - cfg.maxExpandLeft;
    search.y = guess.y - cfg.maxExpandUp;
    search.w = guess.w + cfg.maxExpandLeft + cfg.maxExpandRight;
    search.h = guess.h + cfg.maxExpandUp + cfg.maxExpandDown;
    search = clampROI(search, W, H);

    auto vprof = verticalEdgeProfile(gray, W, search);
    auto hprof = horizontalEdgeProfile(gray, W, search);
    smooth1D(vprof, cfg.smoothWindow);
    smooth1D(hprof, cfg.smoothWindow);

    float vthr = std::max(cfg.edgeThresh, avg(vprof) * 1.8f);
    float hthr = std::max(cfg.edgeThresh, avg(hprof) * 1.8f);

    int guessLeft = guess.x - search.x;
    int guessRight = (guess.x + guess.w - 1) - search.x;
    int guessTop = guess.y - search.y;
    int guessBottom = (guess.y + guess.h - 1) - search.y;

    int leftIdx = findEdgePeak(vprof, guessLeft, -1, vthr);
    int rightIdx = findEdgePeak(vprof, guessRight, +1, vthr);
    int topIdx = findEdgePeak(hprof, guessTop, -1, hthr);
    int botIdx = findEdgePeak(hprof, guessBottom, +1, hthr);

    if (leftIdx < 0) leftIdx = guessLeft;
    if (rightIdx < 0) rightIdx = guessRight;
    if (topIdx < 0) topIdx = guessTop;
    if (botIdx < 0) botIdx = guessBottom;

    ROI refined;
    refined.name = guess.name;
    refined.x = search.x + leftIdx;
    refined.y = search.y + topIdx;
    refined.w = (search.x + rightIdx) - refined.x + 1;
    refined.h = (search.y + botIdx) - refined.y + 1;
    refined = clampROI(refined, W, H);

    // Sanity: header must stay inside refined panel
    bool headerInside = headerRect.x >= refined.x && headerRect.y >= refined.y &&
                        (headerRect.x + headerRect.w) <= (refined.x + refined.w) &&
                        (headerRect.y + headerRect.h) <= (refined.y + refined.h);
    if (!headerInside) return guess;
    if (refined.w < 120 || refined.h < 120) return guess;
    return refined;
}

} // namespace trading_monitor::track
