#include "detection/panel_finder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "utils/stb_image.h" // stbi_load via WIC-backed implementation

namespace trading_monitor::detect {

namespace {

static inline int clampi(int v, int lo, int hi) { return (std::max)(lo, (std::min)(v, hi)); }

static ROI clampROI(const ROI& r, int W, int H) {
    ROI out = r;
    out.x = clampi(out.x, 0, (std::max)(0, W - 1));
    out.y = clampi(out.y, 0, (std::max)(0, H - 1));
    out.w = (std::max)(1, (std::min)(out.w, W - out.x));
    out.h = (std::max)(1, (std::min)(out.h, H - out.y));
    return out;
}

static void bgraToGray(const uint8_t* bgra, int w, int h, int strideBytes, std::vector<uint8_t>& outGray) {
    outGray.resize(static_cast<size_t>(w) * static_cast<size_t>(h));
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = bgra + static_cast<size_t>(y) * static_cast<size_t>(strideBytes);
        uint8_t* dst = outGray.data() + static_cast<size_t>(y) * static_cast<size_t>(w);
        for (int x = 0; x < w; ++x) {
            const uint8_t b = row[x * 4 + 0];
            const uint8_t g = row[x * 4 + 1];
            const uint8_t r = row[x * 4 + 2];
            // ITU-R BT.601-ish luma
            dst[x] = static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
        }
    }
}


static std::vector<uint8_t> resizeGrayNearest(const std::vector<uint8_t>& src, int srcW, int srcH, int dstW, int dstH) {
    std::vector<uint8_t> dst(static_cast<size_t>(dstW) * static_cast<size_t>(dstH));
    for (int y = 0; y < dstH; ++y) {
        const int sy = (srcH == 1) ? 0 : (y * (srcH - 1)) / (dstH - 1);
        for (int x = 0; x < dstW; ++x) {
            const int sx = (srcW == 1) ? 0 : (x * (srcW - 1)) / (dstW - 1);
            dst[static_cast<size_t>(y) * static_cast<size_t>(dstW) + static_cast<size_t>(x)] =
                src[static_cast<size_t>(sy) * static_cast<size_t>(srcW) + static_cast<size_t>(sx)];
        }
    }
    return dst;
}

static bool loadTemplateGray(const std::string& path, std::vector<uint8_t>& tplGray, int& tplW, int& tplH) {
    int w = 0, h = 0, c = 0;
    stbi_uc* data = stbi_load(path.c_str(), &w, &h, &c, 4);
    if (!data || w <= 0 || h <= 0) {
        if (data) stbi_image_free(data);
        return false;
    }

    tplW = w;
    tplH = h;
    tplGray.resize(static_cast<size_t>(w) * static_cast<size_t>(h));

    // stbi_load returns BGRA (via WIC impl in this repo)
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = data + static_cast<size_t>(y) * static_cast<size_t>(w) * 4;
        for (int x = 0; x < w; ++x) {
            const uint8_t b = row[x * 4 + 0];
            const uint8_t g = row[x * 4 + 1];
            const uint8_t r = row[x * 4 + 2];
            tplGray[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] =
                static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
        }
    }

    stbi_image_free(data);
    return true;
}

struct AnchorMatchResult {
    bool found = false;
    float score = 0.0f;
    float tplScale = 1.0f;
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

// Naive NCC (CPU). Intended for downscaled search images and small templates.
static AnchorMatchResult matchTemplateNCC(const std::vector<uint8_t>& img, int imgW, int imgH,
                                         const std::vector<uint8_t>& tpl, int tplW, int tplH,
                                         float minScore) {
    AnchorMatchResult best;
    if (tplW <= 0 || tplH <= 0 || imgW < tplW || imgH < tplH) return best;

    // Precompute template mean and variance.
    const int tN = tplW * tplH;
    double tSum = 0.0;
    double tSq = 0.0;
    for (int i = 0; i < tN; ++i) {
        const double v = tpl[static_cast<size_t>(i)];
        tSum += v;
        tSq += v * v;
    }
    const double tMean = tSum / (double)tN;
    const double tVar = tSq - (tSum * tSum) / (double)tN;
    const double tDen = (tVar > 1e-6) ? std::sqrt(tVar) : 0.0;
    if (tDen == 0.0) return best;

    best.found = false;
    best.score = -1.0f;

    // Sliding window NCC.
    for (int y = 0; y <= imgH - tplH; ++y) {
        for (int x = 0; x <= imgW - tplW; ++x) {
            double iSum = 0.0;
            double iSq = 0.0;
            double dot = 0.0;

            for (int ty = 0; ty < tplH; ++ty) {
                const uint8_t* irow = img.data() + static_cast<size_t>(y + ty) * static_cast<size_t>(imgW) + static_cast<size_t>(x);
                const uint8_t* trow = tpl.data() + static_cast<size_t>(ty) * static_cast<size_t>(tplW);
                for (int tx = 0; tx < tplW; ++tx) {
                    const double iv = irow[tx];
                    const double tv = trow[tx];
                    iSum += iv;
                    iSq += iv * iv;
                    dot += (iv - 0.0) * (tv - tMean); // center template only
                }
            }

            const double iVar = iSq - (iSum * iSum) / (double)tN;
            const double iDen = (iVar > 1e-6) ? std::sqrt(iVar) : 0.0;
            if (iDen == 0.0) continue;

            const double ncc = dot / (iDen * tDen);
            const float score = static_cast<float>(ncc);
            if ((!best.found || score > best.score) && score >= minScore) {
                best.found = true;
                best.score = score;
                best.x = x;
                best.y = y;
                best.w = tplW;
                best.h = tplH;
            }
        }
    }

    return best;
}

static AnchorMatchResult findAnchorByTemplate(const std::vector<uint8_t>& imgGray, int imgW, int imgH,
                                             const std::vector<uint8_t>& tplGray, int tplW, int tplH,
                                             const std::vector<float>& scales, float minScore, float maxSearchW) {
    AnchorMatchResult best;

    // Downscale input image if requested.
    float imgScale = 1.0f;
    std::vector<uint8_t> searchImg = imgGray;
    int sW = imgW;
    int sH = imgH;

    if (maxSearchW > 0.0f && imgW > static_cast<int>(maxSearchW)) {
        imgScale = maxSearchW / static_cast<float>(imgW);
        sW = static_cast<int>(std::round(imgW * imgScale));
        sH = static_cast<int>(std::round(imgH * imgScale));
        sW = (std::max)(16, sW);
        sH = (std::max)(16, sH);
        searchImg = resizeGrayNearest(imgGray, imgW, imgH, sW, sH);
    }

    for (float sc : scales) {
        if (sc <= 0.0f) continue;
        const int tW = (std::max)(4, static_cast<int>(std::round(tplW * sc)));
        const int tH = (std::max)(4, static_cast<int>(std::round(tplH * sc)));
        if (tW > sW || tH > sH) continue;

        auto tplScaled = resizeGrayNearest(tplGray, tplW, tplH, tW, tH);
        AnchorMatchResult m = matchTemplateNCC(searchImg, sW, sH, tplScaled, tW, tH, minScore);
        if (m.found && (!best.found || m.score > best.score)) {
            best = m;
            best.tplScale = sc;
        }
    }

    if (!best.found) return best;

    // Map coordinates back to full-res.
    best.x = static_cast<int>(std::round(best.x / imgScale));
    best.y = static_cast<int>(std::round(best.y / imgScale));
    best.w = static_cast<int>(std::round(best.w / imgScale));
    best.h = static_cast<int>(std::round(best.h / imgScale));

    return best;
}

static std::vector<float> rowEdgeEnergy(const std::vector<uint8_t>& g, int W, int H, const ROI& r) {
    std::vector<float> prof(static_cast<size_t>(r.h), 0.0f);
    for (int y = r.y + 1; y < r.y + r.h - 1; ++y) {
        const uint8_t* up = g.data() + static_cast<size_t>(y - 1) * static_cast<size_t>(W);
        const uint8_t* dn = g.data() + static_cast<size_t>(y + 1) * static_cast<size_t>(W);
        float acc = 0.0f;
        for (int x = r.x + 1; x < r.x + r.w - 1; ++x) {
            acc += static_cast<float>(std::abs((int)dn[x] - (int)up[x]));
        }
        prof[static_cast<size_t>(y - r.y)] = acc;
    }
    return prof;
}

static std::vector<float> colEdgeEnergy(const std::vector<uint8_t>& g, int W, int H, const ROI& r) {
    std::vector<float> prof(static_cast<size_t>(r.w), 0.0f);
    for (int y = r.y + 1; y < r.y + r.h - 1; ++y) {
        const uint8_t* row = g.data() + static_cast<size_t>(y) * static_cast<size_t>(W);
        for (int x = r.x + 1; x < r.x + r.w - 1; ++x) {
            prof[static_cast<size_t>(x - r.x)] += static_cast<float>(std::abs((int)row[x + 1] - (int)row[x - 1]));
        }
    }
    return prof;
}

static int argmaxRange(const std::vector<float>& v, int a, int b) {
    if (v.empty()) return 0;
    a = (std::max)(0, a);
    b = (std::min)(static_cast<int>(v.size()) - 1, b);
    int best = a;
    float bestVal = v[static_cast<size_t>(a)];
    for (int i = a + 1; i <= b; ++i) {
        float val = v[static_cast<size_t>(i)];
        if (val > bestVal) {
            bestVal = val;
            best = i;
        }
    }
    return best;
}

static ROI expandHeaderToPanelEdges(const ROI& header, const std::vector<uint8_t>& gray, int W, int H,
                                   const PanelFinderConfig& cfg, const std::string& name) {
    // Start with a conservative guess anchored at header top-left.
    ROI guess;
    guess.name = name;
    guess.x = header.x - cfg.panelPadX;
    guess.y = header.y - cfg.panelPadY;
    guess.w = cfg.defaultPanelW;
    guess.h = cfg.defaultPanelH;
    guess = clampROI(guess, W, H);

    // Define a search region around guess.
    ROI search = guess;
    search.x -= cfg.edgeSearchPad;
    search.y -= cfg.edgeSearchPad;
    search.w += 2 * cfg.edgeSearchPad;
    search.h += 2 * cfg.edgeSearchPad;
    search = clampROI(search, W, H);

    // Find strong vertical edges near left/right boundaries.
    auto colE = colEdgeEnergy(gray, W, H, search);
    auto rowE = rowEdgeEnergy(gray, W, H, search);

    // Left boundary: search in first ~25% of search ROI.
    const int leftMax = (std::max)(8, static_cast<int>(std::round(search.w * 0.25)));
    int leftIdx = argmaxRange(colE, 0, leftMax);

    // Right boundary: search in last ~35% of search ROI.
    const int rightMin = (std::max)(0, static_cast<int>(std::round(search.w * 0.65)));
    int rightIdx = argmaxRange(colE, rightMin, search.w - 1);

    // Top boundary: search above header within search ROI.
    const int topMax = (std::min)(search.h - 1, (std::max)(8, header.y - search.y + 6));
    int topIdx = argmaxRange(rowE, 0, topMax);

    // Bottom boundary: search near bottom.
    const int bottomMin = (std::max)(0, static_cast<int>(std::round(search.h * 0.55)));
    int bottomIdx = argmaxRange(rowE, bottomMin, search.h - 1);

    // Convert to absolute coords.
    int x0 = search.x + leftIdx;
    int x1 = search.x + rightIdx;
    int y0 = search.y + topIdx;
    int y1 = search.y + bottomIdx;

    // Sanity checks + fallbacks.
    if (x1 <= x0 + 40) {
        x0 = guess.x;
        x1 = guess.x + guess.w - 1;
    }
    if (y1 <= y0 + 40) {
        y0 = guess.y;
        y1 = guess.y + guess.h - 1;
    }

    ROI out;
    out.name = name;
    out.x = clampi(x0, 0, W - 1);
    out.y = clampi(y0, 0, H - 1);
    out.w = clampi(x1 - out.x + 1, 1, W - out.x);
    out.h = clampi(y1 - out.y + 1, 1, H - out.y);
    return out;
}

static FoundPanels findPanelsFromGrayImpl(const uint8_t* grayPtr, int w, int h,
                                         const PanelFinderConfig& cfg, const PanelTemplates& tpl) {
    FoundPanels out;
    if (!grayPtr || w <= 0 || h <= 0) return out;

    std::vector<uint8_t> gray(grayPtr, grayPtr + (static_cast<size_t>(w) * static_cast<size_t>(h)));

    if (!tpl.positionsHdr.empty()) {
        auto m = findAnchorByTemplate(gray, w, h, tpl.positionsHdr, tpl.positionsW, tpl.positionsH,
                                      cfg.scales, cfg.hdrThreshold, cfg.maxSearchW);
        if (m.found) {
            out.hasPositions = true;
            out.scorePositions = m.score;
            out.positionsHeader = clampROI(ROI{"positions_header", m.x, m.y, m.w, m.h}, w, h);
            out.positionsPanel = expandHeaderToPanelEdges(out.positionsHeader, gray, w, h, cfg, "positions_panel");
        }
    }

    if (!tpl.orderHdr.empty()) {
        auto m = findAnchorByTemplate(gray, w, h, tpl.orderHdr, tpl.orderW, tpl.orderH,
                                      cfg.scales, cfg.hdrThreshold, cfg.maxSearchW);
        if (m.found) {
            out.hasOrder = true;
            out.scoreOrder = m.score;
            out.orderHeader = clampROI(ROI{"order_header", m.x, m.y, m.w, m.h}, w, h);
            out.orderPanel = expandHeaderToPanelEdges(out.orderHeader, gray, w, h, cfg, "order_panel");
        }
    }

    if (!tpl.quoteHdr.empty()) {
        auto m = findAnchorByTemplate(gray, w, h, tpl.quoteHdr, tpl.quoteW, tpl.quoteH,
                                      cfg.scales, cfg.hdrThreshold, cfg.maxSearchW);
        if (m.found) {
            out.hasQuote = true;
            out.scoreQuote = m.score;
            out.quoteHeader = clampROI(ROI{"quote_header", m.x, m.y, m.w, m.h}, w, h);
            out.quotePanel = expandHeaderToPanelEdges(out.quoteHeader, gray, w, h, cfg, "quote_panel");
        }
    }

    return out;
}

} // namespace

bool PanelFinder::loadTemplates(const std::string& positionsHdrPath,
                               const std::string& orderHdrPath,
                               const std::string& quoteHdrPath,
                               std::string& err) {
    m_tpl = PanelTemplates{};

    if (!loadTemplateGray(positionsHdrPath, m_tpl.positionsHdr, m_tpl.positionsW, m_tpl.positionsH)) {
        err = "Failed to load positions header template: " + positionsHdrPath;
        return false;
    }

    if (!orderHdrPath.empty()) {
        if (!loadTemplateGray(orderHdrPath, m_tpl.orderHdr, m_tpl.orderW, m_tpl.orderH)) {
            err = "Failed to load order entry header template: " + orderHdrPath;
            return false;
        }
    }

    if (!quoteHdrPath.empty()) {
        if (!loadTemplateGray(quoteHdrPath, m_tpl.quoteHdr, m_tpl.quoteW, m_tpl.quoteH)) {
            err = "Failed to load stock quote header template: " + quoteHdrPath;
            return false;
        }
    }

    return true;
}

FoundPanels PanelFinder::findPanelsFromBGRA(const uint8_t* bgra, int w, int h, int strideBytes,
                                           const PanelFinderConfig& cfg) const {
    std::vector<uint8_t> gray;
    if (!bgra || w <= 0 || h <= 0 || strideBytes <= 0) return {};
    bgraToGray(bgra, w, h, strideBytes, gray);
    return findPanelsFromGrayImpl(gray.data(), w, h, cfg, m_tpl);
}

FoundPanels PanelFinder::findPanelsFromGray(const uint8_t* gray, int w, int h,
                                           const PanelFinderConfig& cfg) const {
    return findPanelsFromGrayImpl(gray, w, h, cfg, m_tpl);
}

} // namespace trading_monitor::detect
