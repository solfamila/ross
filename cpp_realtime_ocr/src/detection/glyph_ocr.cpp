#include "detection/glyph_ocr.h"

#if defined(TM_USE_OPENCV)
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include <algorithm>
#include <cmath>

namespace trading_monitor::detect {

GlyphOCRResult ocrTickerFromRowGray(const std::vector<uint8_t>& rowGray, int w, int h) {
    GlyphOCRResult out;
#if !defined(TM_USE_OPENCV)
    (void)rowGray; (void)w; (void)h;
    return out;
#else
    if (rowGray.empty() || w <= 0 || h <= 0) return out;

    cv::Mat rowMat(h, w, CV_8UC1, (void*)rowGray.data());
    int cw = std::min(w, 220);
    cv::Mat rowCrop = rowMat(cv::Rect(0, 0, cw, h));

    cv::Mat up;
    cv::resize(rowCrop, up, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);

    struct Box { int x, y, w, h; };
    auto findBoxes = [](const cv::Mat& bw, std::vector<Box>& boxesOut) {
        boxesOut.clear();
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bw, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& c : contours) {
            cv::Rect r = cv::boundingRect(c);
            int area = r.width * r.height;
            if (area < 40) continue;
            if (r.height < 12 || r.height > 80) continue;
            if (r.width < 5 || r.width > 60) continue;
            boxesOut.push_back({r.x, r.y, r.width, r.height});
        }
    };

    cv::Mat bw;
    std::vector<Box> boxes;
    cv::threshold(up, bw, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    findBoxes(bw, boxes);
    if (boxes.empty()) {
        cv::threshold(up, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        findBoxes(bw, boxes);
    }
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) { return a.x < b.x; });
    if (boxes.empty()) return out;
    if ((int)boxes.size() > 6) boxes.resize(6);

    int row_h = std::max(18, (int)std::lround(up.rows * 0.9));
    int glyph_w = std::max(12, (int)std::lround(row_h * 0.6));
    float font_scale = 0.6f * (row_h / 30.0f);
    int thickness = (row_h < 28) ? 1 : 2;

    std::string text;
    std::vector<float> scores;
    for (const auto& b : boxes) {
        cv::Rect r(b.x, b.y, b.w, b.h);
        r &= cv::Rect(0, 0, bw.cols, bw.rows);
        if (r.width <= 0 || r.height <= 0) continue;
        cv::Mat patch = bw(r);

        char bestCh = '?';
        double bestSc = -1.0;

        for (char ch = 'A'; ch <= 'Z'; ++ch) {
            cv::Mat glyph(row_h, glyph_w, CV_8UC1, cv::Scalar(0));
            cv::putText(glyph, std::string(1, ch), cv::Point(1, row_h - 6),
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255),
                        thickness, cv::LINE_AA);

            cv::Mat patch_r;
            cv::resize(patch, patch_r, glyph.size(), 0, 0, cv::INTER_AREA);

            cv::Mat res;
            cv::matchTemplate(patch_r, glyph, res, cv::TM_CCORR_NORMED);
            double minv, maxv; cv::Point minp, maxp;
            cv::minMaxLoc(res, &minv, &maxv, &minp, &maxp);
            if (maxv > bestSc) { bestSc = maxv; bestCh = ch; }
        }

        text.push_back(bestCh);
        scores.push_back((float)bestSc);
    }

    out.text = text;
    if (!scores.empty()) {
        double s = 0.0;
        for (float v : scores) s += v;
        out.score = (float)(s / (double)scores.size());
    }
    return out;
#endif
}

} // namespace trading_monitor::detect
