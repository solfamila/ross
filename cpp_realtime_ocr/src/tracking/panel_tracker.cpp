#include "tracking/panel_tracker.h"
#include "tracking/edge_expand.h"

#include <algorithm>

#if defined(TM_USE_OPENCV)
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#endif
#include <cmath>
#include <cstring>

namespace {
static trading_monitor::ROI clampROI(const trading_monitor::ROI& r, int W, int H) {
    trading_monitor::ROI out = r;
    out.x = std::max(0, std::min(out.x, W - 1));
    out.y = std::max(0, std::min(out.y, H - 1));
    out.w = std::max(1, std::min(out.w, W - out.x));
    out.h = std::max(1, std::min(out.h, H - out.y));
    return out;
}

static std::vector<uint8_t> cropGrayRegion(const std::vector<uint8_t>& src, int srcW, int srcH, int x, int y, int w, int h) {
    x = std::max(0, std::min(x, srcW - 1));
    y = std::max(0, std::min(y, srcH - 1));
    w = std::max(1, std::min(w, srcW - x));
    h = std::max(1, std::min(h, srcH - y));
    std::vector<uint8_t> out((size_t)w * (size_t)h);
    for (int yy = 0; yy < h; ++yy) {
        const uint8_t* row = src.data() + (size_t)(y + yy) * (size_t)srcW + (size_t)x;
        uint8_t* dst = out.data() + (size_t)yy * (size_t)w;
        std::memcpy(dst, row, (size_t)w);
    }
    return out;
}

static std::vector<uint8_t> resizeGrayNearest(const std::vector<uint8_t>& src, int srcW, int srcH, int dstW, int dstH) {
#if defined(TM_USE_OPENCV)
    cv::Mat srcMat(srcH, srcW, CV_8UC1, (void*)src.data());
    cv::Mat dstMat;
    cv::resize(srcMat, dstMat, cv::Size(dstW, dstH), 0.0, 0.0, cv::INTER_AREA);
    std::vector<uint8_t> dst((size_t)dstW * (size_t)dstH);
    for (int y = 0; y < dstH; ++y) {
        const uint8_t* row = dstMat.ptr<uint8_t>(y);
        std::memcpy(dst.data() + (size_t)y * (size_t)dstW, row, (size_t)dstW);
    }
    return dst;
#else
    std::vector<uint8_t> dst((size_t)dstW * (size_t)dstH);
    for (int y = 0; y < dstH; ++y) {
        int sy = (y * srcH) / dstH;
        const uint8_t* srow = src.data() + (size_t)sy * (size_t)srcW;
        uint8_t* drow = dst.data() + (size_t)y * (size_t)dstW;
        for (int x = 0; x < dstW; ++x) {
            int sx = (x * srcW) / dstW;
            drow[x] = srow[sx];
        }
    }
    return dst;
#endif
}

struct MatchResult { bool found=false; float score=0; int x=0,y=0; };

// Minimal NCC match: reuse your existing GPU NCC later if desired.
static MatchResult matchTemplateNCC(const std::vector<uint8_t>& img, int iW, int iH,
                                    const std::vector<uint8_t>& tpl, int tW, int tH) {
#if defined(TM_USE_OPENCV)
    MatchResult best;
    if (tW <= 0 || tH <= 0 || iW < tW || iH < tH) return best;
    cv::Mat imgMat(iH, iW, CV_8UC1, (void*)img.data());
    cv::Mat tplMat(tH, tW, CV_8UC1, (void*)tpl.data());
    cv::Mat res;
    cv::matchTemplate(imgMat, tplMat, res, cv::TM_CCOEFF_NORMED);
    double minv, maxv; cv::Point minp, maxp;
    cv::minMaxLoc(res, &minv, &maxv, &minp, &maxp);
    best.found = true;
    best.score = static_cast<float>(maxv);
    best.x = maxp.x;
    best.y = maxp.y;
    return best;
#else
    MatchResult best;
    if (tW <= 0 || tH <= 0 || iW < tW || iH < tH) return best;

    const int n = tW * tH;
    double sumT=0, sumT2=0;
    for (int i=0;i<n;++i){ double v=tpl[(size_t)i]; sumT+=v; sumT2+=v*v; }
    double meanT = sumT / n;
    double varT = sumT2 - sumT * meanT;
    if (varT <= 1e-6) return best;

    best.score = -1.0f;
    for (int y=0; y<=iH-tH; ++y){
        for (int x=0; x<=iW-tW; ++x){
            double sumI=0,sumI2=0,sumIT=0;
            for (int ty=0; ty<tH; ++ty){
                const uint8_t* ir = img.data() + (size_t)(y+ty)*(size_t)iW + (size_t)x;
                const uint8_t* tr = tpl.data() + (size_t)ty*(size_t)tW;
                for (int tx=0; tx<tW; ++tx){
                    double iv = ir[tx];
                    double tv = tr[tx];
                    sumI += iv; sumI2 += iv*iv; sumIT += iv*tv;
                }
            }
            double meanI = sumI / n;
            double varI = sumI2 - sumI * meanI;
            if (varI <= 1e-6) continue;
            double numerator = sumIT - sumI*meanT - sumT*meanI + (double)n*meanI*meanT;
            double denom = std::sqrt(varI*varT);
            if (denom <= 1e-6) continue;
            float ncc = (float)(numerator/denom);
            if (!best.found || ncc > best.score) {
                best.found = true; best.score = ncc; best.x = x; best.y = y;
            }
        }
    }
    return best;
#endif
}
}

namespace trading_monitor::track {

void PanelTracker::init(const HeaderTemplate& headerTpl, const ROI& initialHeaderRect, const ROI& initialPanelRect) {
    m_tpl = headerTpl;
    m_lastHeader = initialHeaderRect;
    m_lastPanel = initialPanelRect;
    m_lastScore = 1.0f;
    m_inited = (!m_tpl.gray.empty() && m_tpl.w > 0 && m_tpl.h > 0);
    m_lastExpandedFrame = 0;
    m_lastScale = 1.0f;
}

TrackResult PanelTracker::update(const std::vector<uint8_t>& frameGray, int frameW, int frameH,
                                const TrackerConfig& cfg, uint64_t frameIndex, bool enableEdgeExpansion) {
    TrackResult out;
    if (!m_inited) return out;
    if ((int)frameGray.size() < frameW*frameH) return out;

    int sr = std::max(8, cfg.searchRadiusPx);
    ROI search;
    search.x = m_lastHeader.x - sr;
    search.y = m_lastHeader.y - sr;
    search.w = m_lastHeader.w + 2*sr;
    search.h = m_lastHeader.h + 2*sr;
    search = clampROI(search, frameW, frameH);

    std::vector<uint8_t> searchGray = cropGrayRegion(frameGray, frameW, frameH, search.x, search.y, search.w, search.h);
    int sW = search.w, sH = search.h;

    float searchScale = 1.0f;
    if (cfg.maxSearchW > 0 && sW > cfg.maxSearchW) {
        searchScale = (float)cfg.maxSearchW / (float)sW;
        int dsW = std::max(8, (int)std::lround(sW*searchScale));
        int dsH = std::max(8, (int)std::lround(sH*searchScale));
        searchGray = resizeGrayNearest(searchGray, sW, sH, dsW, dsH);
        sW = dsW; sH = dsH;
    }

    MatchResult best;
    float bestScale = m_lastScale;
    int bestTplW = m_tpl.w;
    int bestTplH = m_tpl.h;
    if (cfg.scaleMultipliers.empty()) {
        best = matchTemplateNCC(searchGray, sW, sH, m_tpl.gray, m_tpl.w, m_tpl.h);
    } else {
        for (float mult : cfg.scaleMultipliers) {
            float s = std::max(cfg.minScale, std::min(cfg.maxScale, m_lastScale * mult));
            int tW = std::max(8, (int)std::lround(m_tpl.w * s * searchScale));
            int tH = std::max(8, (int)std::lround(m_tpl.h * s * searchScale));
            if (tW >= sW || tH >= sH) continue;
            std::vector<uint8_t> tplScaled = resizeGrayNearest(m_tpl.gray, m_tpl.w, m_tpl.h, tW, tH);
            MatchResult m = matchTemplateNCC(searchGray, sW, sH, tplScaled, tW, tH);
            if (!best.found || m.score > best.score) {
                best = m;
                bestScale = s;
                bestTplW = tW;
                bestTplH = tH;
            }
        }
    }

    if (!best.found || best.score < cfg.minTrackScore) {
        out.ok = false;
        out.score = best.found ? best.score : -1.0f;
        return out;
    }

    int mx = best.x, my = best.y;
    if (searchScale != 1.0f) {
        mx = (int)std::lround(mx / searchScale);
        my = (int)std::lround(my / searchScale);
    }

    int headerW = bestTplW;
    int headerH = bestTplH;
    if (searchScale != 1.0f) {
        headerW = (int)std::lround(bestTplW / searchScale);
        headerH = (int)std::lround(bestTplH / searchScale);
    }

    ROI newHeader = m_lastHeader;
    newHeader.x = search.x + mx;
    newHeader.y = search.y + my;
    newHeader.w = headerW;
    newHeader.h = headerH;
    newHeader = clampROI(newHeader, frameW, frameH);

    int dx = newHeader.x - m_lastHeader.x;
    int dy = newHeader.y - m_lastHeader.y;

    ROI newPanel = m_lastPanel;
    newPanel.x += dx;
    newPanel.y += dy;
    newPanel = clampROI(newPanel, frameW, frameH);

    if (enableEdgeExpansion && cfg.reexpandEveryNFrames > 0) {
        if (m_lastExpandedFrame == 0 || (frameIndex - m_lastExpandedFrame) >= (uint64_t)cfg.reexpandEveryNFrames) {
            EdgeExpandConfig ecfg;
            ROI refined = expandPanelByEdges(frameGray, frameW, frameH, newHeader, newPanel, ecfg);
            if (refined.w > 120 && refined.h > 120) {
                newPanel = clampROI(refined, frameW, frameH);
                m_lastExpandedFrame = frameIndex;
            }
        }
    }

    m_lastHeader = newHeader;
    m_lastPanel = newPanel;
    m_lastScore = best.score;
    m_lastScale = bestScale;

    out.ok = true;
    out.score = best.score;
    out.headerRect = newHeader;
    out.panelRect = newPanel;
    return out;
}

} // namespace trading_monitor::track
