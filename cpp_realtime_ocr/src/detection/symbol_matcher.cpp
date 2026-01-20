#include "detection/symbol_matcher.h"

#include <algorithm>
#include <filesystem>
#include <cstring>

#include "utils/stb_image.h"

#if defined(TM_USE_OPENCV)
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#endif

namespace {
static void bgraToGrayStride(const uint8_t* bgra, int w, int h, int strideBytes,
                             std::vector<uint8_t>& outGray) {
    outGray.resize((size_t)w * (size_t)h);
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = bgra + (size_t)y * (size_t)strideBytes;
        uint8_t* out = outGray.data() + (size_t)y * (size_t)w;
        for (int x = 0; x < w; ++x) {
            const uint8_t b = row[x * 4 + 0];
            const uint8_t g = row[x * 4 + 1];
            const uint8_t r = row[x * 4 + 2];
            out[x] = (uint8_t)(((int)r * 77 + (int)g * 150 + (int)b * 29) >> 8);
        }
    }
}

static std::vector<uint8_t> cropGrayRegion(const std::vector<uint8_t>& src, int srcW, int srcH,
                                           int x, int y, int w, int h) {
    x = std::max(0, std::min(x, srcW - 1));
    y = std::max(0, std::min(y, srcH - 1));
    w = std::max(1, std::min(w, srcW - x));
    h = std::max(1, std::min(h, srcH - y));
    std::vector<uint8_t> out((size_t)w * (size_t)h);
    for (int yy = 0; yy < h; ++yy) {
        const uint8_t* row = src.data() + (size_t)(y + yy) * (size_t)srcW + (size_t)x;
        std::memcpy(out.data() + (size_t)yy * (size_t)w, row, (size_t)w);
    }
    return out;
}

static bool loadImageBGRA(const std::string& path, std::vector<uint8_t>& bgra, int& w, int& h) {
    int c = 0;
    stbi_uc* data = stbi_load(path.c_str(), &w, &h, &c, 4);
    if (!data || w <= 0 || h <= 0) {
        if (data) stbi_image_free(data);
        return false;
    }
    bgra.assign(data, data + (size_t)w * (size_t)h * 4);
    stbi_image_free(data);
    return true;
}

struct MatchResult { bool found=false; float score=0; int x=0; int y=0; };
static MatchResult matchTemplateNCC(const std::vector<uint8_t>& img, int iW, int iH,
                                    const std::vector<uint8_t>& tpl, int tW, int tH) {
    MatchResult best;
    if (tW <= 0 || tH <= 0 || iW < tW || iH < tH) return best;
#if defined(TM_USE_OPENCV)
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
    const int n = tW * tH;
    double sumT=0,sumT2=0;
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
                best.found = true;
                best.score = ncc;
                best.x = x;
                best.y = y;
            }
        }
    }
    return best;
#endif
}
}

namespace trading_monitor::detect {

bool SymbolMatcher::loadSymbolTemplates(const std::string& dir, const std::string& symbol, std::string& err) {
    m_templates.clear();
    namespace fs = std::filesystem;
    if (!fs::exists(dir)) {
        err = "Template dir not found: " + dir;
        return false;
    }

    const std::string prefix = symbol + "_";
    for (auto& p : fs::directory_iterator(dir)) {
        if (!p.is_regular_file()) continue;
        const std::string name = p.path().filename().string();
        if (name.rfind(prefix, 0) != 0) continue;

        int w=0,h=0;
        std::vector<uint8_t> bgra;
        if (!loadImageBGRA(p.path().string(), bgra, w, h)) continue;

        SymbolTemplate t;
        t.w = w; t.h = h;
        bgraToGrayStride(bgra.data(), w, h, w*4, t.gray);
        if (!t.gray.empty()) {
            double sum = 0.0;
            double sum2 = 0.0;
            for (uint8_t v : t.gray) {
                sum += v;
                sum2 += (double)v * (double)v;
            }
            const double mean = sum / (double)t.gray.size();
            const double var = sum2 - (double)t.gray.size() * mean * mean;
            if (var > 1e-6) {
                m_templates.push_back(std::move(t));
            }
        }
    }

    if (m_templates.empty()) {
        err = "No templates found for symbol '" + symbol + "' in " + dir;
        return false;
    }
    return true;
}

float SymbolMatcher::matchInGrayROI(const std::vector<uint8_t>& frameGray, int frameW, int frameH,
                                    const ROI& roi) const {
    if (m_templates.empty()) return -1.0f;

    int x = std::max(0, std::min(roi.x, frameW - 1));
    int y = std::max(0, std::min(roi.y, frameH - 1));
    int w = std::max(1, std::min(roi.w, frameW - x));
    int h = std::max(1, std::min(roi.h, frameH - y));

    auto crop = cropGrayRegion(frameGray, frameW, frameH, x, y, w, h);
    float best = -1.0f;
    for (const auto& t : m_templates) {
        auto m = matchTemplateNCC(crop, w, h, t.gray, t.w, t.h);
        if (m.found) best = std::max(best, m.score);
    }
    return best;
}

SymbolMatch SymbolMatcher::matchInGrayROIWithLoc(const std::vector<uint8_t>& frameGray, int frameW, int frameH,
                                                 const ROI& roi) const {
    SymbolMatch out;
    if (m_templates.empty()) return out;

    int x = std::max(0, std::min(roi.x, frameW - 1));
    int y = std::max(0, std::min(roi.y, frameH - 1));
    int w = std::max(1, std::min(roi.w, frameW - x));
    int h = std::max(1, std::min(roi.h, frameH - y));

    auto crop = cropGrayRegion(frameGray, frameW, frameH, x, y, w, h);
    for (const auto& t : m_templates) {
        auto m = matchTemplateNCC(crop, w, h, t.gray, t.w, t.h);
        if (!m.found) continue;
        if (!out.found || m.score > out.score) {
            out.found = true;
            out.score = m.score;
            out.x = x + m.x;
            out.y = y + m.y;
            out.w = t.w;
            out.h = t.h;
        }
    }
    return out;
}

} // namespace trading_monitor::detect