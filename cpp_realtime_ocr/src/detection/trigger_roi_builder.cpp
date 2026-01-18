#include "detection/trigger_roi_builder.h"

#include <algorithm>
#include <cmath>

namespace trading_monitor::detect {

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
    std::vector<float> out(v.size(), 0.f);
    int r = win / 2;
    for (int i = 0; i < (int)v.size(); ++i) {
        float s = 0.f; int c = 0;
        for (int k=-r; k<=r; ++k) {
            int j=i+k;
            if (j<0||j>=(int)v.size()) continue;
            s += v[(size_t)j]; c++;
        }
        out[(size_t)i] = c? s/(float)c : 0.f;
    }
    v.swap(out);
}

static float avg(const std::vector<float>& v) {
    if (v.empty()) return 0.f;
    double s=0;
    for (float x : v) s += x;
    return (float)(s/(double)v.size());
}

static std::vector<float> rowEdgeEnergy(const std::vector<uint8_t>& g, int W, const ROI& r) {
    std::vector<float> prof((size_t)r.h, 0.f);
    for (int y=r.y+1; y<r.y+r.h-1; ++y) {
        const uint8_t* up = g.data() + (size_t)(y-1)*(size_t)W;
        const uint8_t* dn = g.data() + (size_t)(y+1)*(size_t)W;
        float acc=0.f;
        for (int x=r.x+1; x<r.x+r.w-1; ++x) {
            int gy = (int)dn[x] - (int)up[x];
            acc += (float)std::abs(gy);
        }
        prof[(size_t)(y-r.y)] = acc;
    }
    return prof;
}

static std::vector<float> colEdgeEnergy(const std::vector<uint8_t>& g, int W, const ROI& r) {
    std::vector<float> prof((size_t)r.w, 0.f);
    for (int y=r.y+1; y<r.y+r.h-1; ++y) {
        const uint8_t* row = g.data() + (size_t)y*(size_t)W;
        for (int x=r.x+1; x<r.x+r.w-1; ++x) {
            int gx = (int)row[x+1] - (int)row[x-1];
            prof[(size_t)(x-r.x)] += (float)std::abs(gx);
        }
    }
    return prof;
}

static int argmaxRange(const std::vector<float>& v, int a, int b) {
    a = std::max(0, a);
    b = std::min((int)v.size()-1, b);
    int best = a;
    float bestVal = v[(size_t)a];
    for (int i=a+1; i<=b; ++i) {
        float val = v[(size_t)i];
        if (val > bestVal) { bestVal = val; best = i; }
    }
    return best;
}

static std::pair<int,float> estimateRowHeight(const std::vector<float>& prof, int minP, int maxP) {
    if ((int)prof.size() < maxP*2) maxP = std::max(minP, (int)prof.size()/2);
    int bestP = minP;
    float bestScore = -1.f;
    for (int p=minP; p<=maxP; ++p) {
        double s=0; int n=0;
        for (int i=0; i+p<(int)prof.size(); ++i) {
            s += (double)prof[(size_t)i] * (double)prof[(size_t)(i+p)];
            n++;
        }
        float sc = (n? (float)(s/(double)n) : -1.f);
        if (sc > bestScore) { bestScore = sc; bestP = p; }
    }
    return {bestP, bestScore};
}

TriggerRoiResult TriggerRoiBuilder::build(const std::vector<uint8_t>& gray, int W, int H,
                                         const ROI& positionsPanel,
                                         const TriggerRoiConfig& cfg) const {
    TriggerRoiResult out;
    if ((int)gray.size() < W*H) return out;

    ROI panel = clampROI(positionsPanel, W, H);

    // 1) header bottom via row-edge peak in upper panel region
    ROI hdrSearch;
    hdrSearch.x = panel.x + cfg.bodyPadX;
    hdrSearch.y = panel.y;
    hdrSearch.w = std::max(1, panel.w - 2*cfg.bodyPadX);
    hdrSearch.h = std::max(20, (int)std::lround(panel.h * cfg.headerSearchFracH));
    hdrSearch = clampROI(hdrSearch, W, H);

    auto hdrEdges = rowEdgeEnergy(gray, W, hdrSearch);
    smooth1D(hdrEdges, 7);
    if (hdrEdges.size() < 10) return out;
    int hdrBottomLocal = argmaxRange(hdrEdges, 4, (int)hdrEdges.size()-5);
    out.headerBottomScore = hdrEdges[(size_t)hdrBottomLocal];
    int tableTop = hdrSearch.y + hdrBottomLocal + cfg.bodyPadTop;

    // 2) table body
    ROI body;
    body.name = "positions_body";
    body.x = panel.x + cfg.bodyPadX;
    body.y = tableTop;
    body.w = std::max(1, panel.w - 2*cfg.bodyPadX);
    body.h = (panel.y + panel.h) - body.y - cfg.bodyPadBottom;
    body = clampROI(body, W, H);
    if (body.w < 120 || body.h < 60) return out;
    out.tableBody = body;

    // 3) estimate row height from periodicity of row-edge energy on left strip
    ROI bodyLeft = body;
    bodyLeft.w = std::max(60, (int)std::lround(body.w * cfg.leftStripFrac));
    bodyLeft = clampROI(bodyLeft, W, H);
    auto rowEdges = rowEdgeEnergy(gray, W, bodyLeft);
    smooth1D(rowEdges, 5);
    auto [rowH, rowScore] = estimateRowHeight(rowEdges, cfg.rowMin, cfg.rowMax);
    out.rowHeight = rowH;
    out.rowPeriodScore = rowScore;
    if (rowScore < 1e3f) out.rowHeight = 18; // safe fallback

    // 4) symbol col boundary via vertical edge profile
    ROI topBand = bodyLeft;
    topBand.h = std::min(topBand.h, std::max(80, out.rowHeight * std::min(cfg.topRows + 2, 12)));
    topBand = clampROI(topBand, W, H);
    auto colEdges = colEdgeEnergy(gray, W, topBand);
    smooth1D(colEdges, 9);
    float need = std::max(1.0f, avg(colEdges) * cfg.minSeparatorStrength);
    int bestSep = -1;
    float bestVal = 0.f;
    for (int i=25; i<(int)colEdges.size(); ++i) {
        float v = colEdges[(size_t)i];
        if (v >= need && v > bestVal) { bestVal = v; bestSep = i; }
    }
    out.separatorScore = bestVal;
    int symW = (bestSep >= 0)
        ? clampi(bestSep, cfg.minSymbolColW, cfg.maxSymbolColW)
        : clampi((int)std::lround(body.w * 0.28), cfg.minSymbolColW, cfg.maxSymbolColW);

    ROI sym;
    sym.name = "symbol_col";
    sym.x = body.x;
    sym.y = body.y;
    sym.w = symW;
    sym.h = body.h;
    sym = clampROI(sym, W, H);
    out.symbolCol = sym;

    // 5) trigger ROI = symbol col x top rows
    ROI trig;
    trig.name = "trigger_roi";
    trig.x = sym.x;
    trig.y = sym.y;
    trig.w = sym.w;
    trig.h = std::min(sym.h, out.rowHeight * cfg.topRows);
    trig = clampROI(trig, W, H);
    if (trig.h < 30) return out;

    out.triggerRoi = trig;
    out.ok = true;
    return out;
}

} // namespace trading_monitor::detect
