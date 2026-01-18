/**
 * @file main.cpp
 * @brief Trading Screen Monitor - Main entry point
 * 
 * High-performance screen capture with TensorRT OCR
 * Target: <20ms end-to-end latency
 * 
 * Requirements:
 * - CUDA 13.1.1
 * - TensorRT 10.14
 * - Windows 11 with Windows.Graphics.Capture
 */

// Minimal end-to-end pipeline:
// Windows.Graphics.Capture (D3D11) -> CUDA interop -> ROI extraction/preprocess -> TensorRT SVTR -> CTC decode

#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include <winrt/base.h>

#include "capture/d3d11_capture.h"
#include "capture/cuda_interop.h"
#include "detection/change_detector.h"
#include "detection/panel_finder.h"
#include "detection/symbol_matcher.h"
#include "detection/entry_trigger.h"
#include "detection/trigger_roi_builder.h"
#include "detection/yolo_detector.h"
#include "ocr/ctc_decoder.h"
#include "ocr/svtr_inference.h"
#include "processing/roi_extractor.h"
#include "processing/row_detector.h"
#include "types.h"
#include "utils/stb_image.h"
#include "utils/config_loader.h"
#include "utils/roi_selector.h"
#include "utils/roi_overlay.h"
#include "utils/timer.h"
#include "utils/profiler.h"
#include "video/mf_source_reader.h"

#include "tracking/panel_tracker.h"

namespace fs = std::filesystem;

// Global flag for graceful shutdown
std::atomic<bool> g_running{true};

using namespace trading_monitor;

// JSON result entry for structured output
struct OCRResultEntry {
    std::string roiName;
    int x, y, w, h;
    std::string text;
    std::string rawText;
    float confidence;
    double latencyMs;
};

// Write JSON output file
static bool writeJsonResults(const std::string& path, const std::vector<OCRResultEntry>& results,
                             const std::string& imagePath, double totalLatencyMs) {
    std::ofstream out(path);
    if (!out.good()) return false;

    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    out << "{\n";
    out << "  \"timestamp\": " << epoch << ",\n";
    out << "  \"source\": \"" << imagePath << "\",\n";
    out << "  \"total_latency_ms\": " << std::fixed << std::setprecision(2) << totalLatencyMs << ",\n";
    out << "  \"results\": [\n";

    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        out << "    {\n";
        out << "      \"roi\": \"" << r.roiName << "\",\n";
        out << "      \"x\": " << r.x << ", \"y\": " << r.y << ", \"w\": " << r.w << ", \"h\": " << r.h << ",\n";
        // Escape special characters in text
        std::string escapedText = r.text;
        std::string escapedRaw = r.rawText;
        for (auto* s : {&escapedText, &escapedRaw}) {
            size_t pos = 0;
            while ((pos = s->find('\\', pos)) != std::string::npos) { s->replace(pos, 1, "\\\\"); pos += 2; }
            pos = 0;
            while ((pos = s->find('"', pos)) != std::string::npos) { s->replace(pos, 1, "\\\""); pos += 2; }
        }
        out << "      \"text\": \"" << escapedText << "\",\n";
        out << "      \"raw_text\": \"" << escapedRaw << "\",\n";
        out << "      \"confidence\": " << std::fixed << std::setprecision(4) << r.confidence << ",\n";
        out << "      \"latency_ms\": " << std::fixed << std::setprecision(2) << r.latencyMs << "\n";
        out << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }

    out << "  ]\n";
    out << "}\n";
    return out.good();
}

void signalHandler(int signal) {
    (void)signal;
    std::cout << "\nShutdown requested..." << std::endl;
    g_running = false;
}

void printBanner() {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════════╗
║         Trading Screen Monitor v1.0                           ║
║         CUDA 13.1.1 + TensorRT 10.14                         ║
║         Target: <20ms latency                                 ║
╚═══════════════════════════════════════════════════════════════╝
)" << std::endl;
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n\n"
              << "Options:\n"
              << "  --video <path>     Offline mode: decode frames from MP4 (Media Foundation) and exit\n"
              << "  --video-max-frames <n>  Offline mode: decode at most N frames (0 = until EOF)\n"
              << "  --detect-panels    Offline mode (with --video): run panel header template matching\n"
              << "  --panel-template-dir <path>  Panel header templates dir (default: templates/headers)\n"
              << "  --panel-threshold <f>  Min NCC score for header match (default: 0.65)\n"
              << "  --panel-every <n>  In offline detect mode, run detection every N frames (default: 1)\n"
              << "  --config <path>    Path to config file (default: config/roi_config.json)\n"
              << "  --window <title>   Window title to capture (partial match)\n"
              << "  --list-windows     List capturable windows and exit\n"
              << "  --select-window [idx]  Select a window (optional index; prompts if omitted)\n"
              << "  --monitor <idx>    Monitor index to capture (default: 0)\n"
              << "  --list-monitors    List monitors and exit\n"
              << "  --select-monitor [idx] Select a monitor (optional index; prompts if omitted)\n"
              << "  --list-rois        List ROIs from config and exit\n"
              << "  --roi <name>       Only run OCR on ROI with this name\n"
              << "  --select-roi [idx] Select ROI (optional index; prompts if omitted)\n"
              << "  --create-roi [name] Create/update an ROI by dragging on the selected window\n"
              << "  --all-rois         Run OCR on all configured ROIs\n"
              << "  --every <n>        Process every Nth frame (default: 5)\n"
              << "  --engine <path>    TensorRT engine path (default: models/recognition.engine)\n"
              << "  --onnx <path>      ONNX path (build engine if missing)\n"
              << "  --dict <path>      Dictionary path (default: models/ppocr_keys_v1.txt)\n"
              << "  --no-change-detect Disable ROI change detection (always OCR)\n"
              << "  --change-threshold <n>  Hamming distance threshold (default: 5)\n"
              << "  --dump-roi <dir>   Dump preprocessed ROI as PGM for debugging\n"
              << "  --show-roi         Draw ROI rectangles on a click-through overlay\n"
              << "  --ocr-zoom <f>      Pre-upscale/zoom ROI content before OCR (default: 1.0)\n"
              << "  --scan-table <roi> Scan a table ROI by slicing rows and OCRing each row\n"
              << "  --table-rows <n>   Number of rows to scan in table mode (default: 6)\n"
              << "  --row-height <px>  Row ROI height in pixels (default: 18)\n"
              << "  --row-stride <px>  Row-to-row Y step in pixels (default: 18)\n"
              << "  --row-offset-y <px> Y offset from table ROI top (default: 0)\n"
              << "  --columns <defs>   Column definitions: 'name:x,w;name:x,w' e.g. 'symbol:0,50;pnl:145,40'\n"
              << "  --col-x <px>       Single column X offset from row start (default: 0 = full row)\n"
              << "  --col-w <px>       Single column width (default: 0 = full row width)\n"
              << "  --print-all-rows   In table mode, print every row every frame\n"
              << "  --auto-rows        Automatically detect row boundaries in table mode\n"
              << "  --row-detect-mode <intensity|template|hybrid>  Row detection mode (default: intensity)\n"
              << "  --anchor-template <path>  Template image for anchoring (full-screen match)\n"
              << "  --anchor-offset <dx,dy,w,h>  ROI offset from matched template (scaled)\n"
              << "  --anchor-scales <s1,s2,..>  Template scales to search (default: 0.6,0.7,0.75,0.8,0.9,1.0,1.1,1.2)\n"
              << "  --anchor-threshold <f>  Min NCC score for anchor match (default: 0.55)\n"
              << "  --anchor-max-search <px>  Max search width for anchor (0 = full res, default: 640)\n"
              << "  --anchor-search <x,y,w,h>  Restrict anchor search to a region\n"
              << "  --anchor-secondary-template <path>  Secondary template to validate anchor\n"
              << "  --anchor-secondary-offset <dx,dy,w,h>  Expected secondary ROI offset from anchor\n"
              << "  --anchor-secondary-threshold <f>  Min NCC for secondary template (default: 0.6)\n"
              << "  --anchor-every <n>  Re-anchor every N frames in live mode (default: 0 = once)\n"
              << "  --template-dir <path>    Template directory for template matching (default: templates)\n"
              << "  --template-threshold <f> Match threshold for template NCC (default: 0.7)\n"
              << "  --template-row-height <px> Expected row height for template detection (default: 14)\n"
              << "  --template-row-spacing <px> Expected row spacing (default: 2)\n"
              << "  --template-max-rows <n> Max rows to detect via template matching (default: 20)\n"
              << "  --image <path>     Process a static image file instead of live capture\n"
              << "  --test-roi <x,y,w,h>  Use custom ROI coordinates for testing (e.g. 100,200,300,50)\n"
              << "  --json-output <file>  Write OCR results to JSON file (structured output)\n"
              << "  --benchmark        Run benchmark mode\n"
              << "  --verbose          Enable verbose logging\n"
              << "  --help             Show this help message\n"
              << "\n"
              << "YOLOv10 Window Detection:\n"
              << "  --detect-model <path>  YOLOv10 TensorRT engine for window detection\n"
              << "  --detect-onnx <path>   YOLOv10 ONNX path (build engine if missing)\n"
              << "  --detect-confidence <f>  Detection confidence threshold (default: 0.8)\n"
              << "  --detect-every <n>     Run detection every N frames (default: 5)\n"
              << "  --detect-classes <names>  Comma-separated class names (e.g. \"window,table\")\n"
              << "\n"
              << "Column Extraction Examples:\n"
              << "  Multi-column: --columns \"symbol:11,30;pnl:193,25;realized:228,60\"\n"
              << "  Single-column: --col-x 11 --col-w 30\n"
              << std::endl;
}

static bool writePGM8(const std::string& path, const std::vector<uint8_t>& data, int w, int h) {
    std::ofstream out(path, std::ios::binary);
    if (!out.good()) return false;
    out << "P5\n" << w << " " << h << "\n255\n";
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return out.good();
}

static void printWindowsList(const std::vector<std::pair<void*, std::string>>& windows) {
    std::cout << "Available windows:" << std::endl;
    for (size_t i = 0; i < windows.size(); i++) {
        std::cout << "  [" << i << "] " << windows[i].second << std::endl;
    }
    if (windows.empty()) {
        std::cout << "  (none)" << std::endl;
    }
}

static std::string getWindowTitle(void* hwndVoid) {
    HWND hwnd = reinterpret_cast<HWND>(hwndVoid);
    if (!hwnd || !IsWindow(hwnd)) return {};
    char buf[512] = {};
    GetWindowTextA(hwnd, buf, static_cast<int>(sizeof(buf)));
    return std::string(buf);
}

static void printMonitorsList(const std::vector<std::pair<void*, std::string>>& monitors) {
    std::cout << "Available monitors:" << std::endl;
    for (size_t i = 0; i < monitors.size(); i++) {
        std::cout << "  [" << i << "] " << monitors[i].second << std::endl;
    }
    if (monitors.empty()) {
        std::cout << "  (none)" << std::endl;
    }
}

static void printRoisList(const PipelineConfig& cfg) {
    std::cout << "Available ROIs (from config):" << std::endl;
    for (size_t i = 0; i < cfg.rois.size(); i++) {
        const auto& r = cfg.rois[i];
        std::cout << "  [" << i << "] " << r.name << "  (x=" << r.x << ", y=" << r.y
                  << ", w=" << r.w << ", h=" << r.h << ")" << std::endl;
    }
    if (cfg.rois.empty()) {
        std::cout << "  (none)" << std::endl;
    }
}

static bool looksLikeInt(const std::string& s) {
    if (s.empty()) return false;
    size_t i = 0;
    if (s[0] == '+' || s[0] == '-') i = 1;
    if (i >= s.size()) return false;
    for (; i < s.size(); i++) {
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    }
    return true;
}

static std::string filterNumericLike(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char ch : s) {
        const bool keep =
            (ch >= '0' && ch <= '9') ||
            ch == '.' || ch == ',' || ch == '-' || ch == '+' || ch == '$' || ch == '%';
        if (keep) out.push_back(ch);
    }
    return out;
}

static void printROI(const ROI& r) {
    std::cout << "  " << r.name << ": x=" << r.x << " y=" << r.y << " w=" << r.w << " h=" << r.h << "\n";
}

static std::string extractTickerLike(const std::string& s) {
    // Heuristic: find the first run of letters (A-Z) length 2..6.
    std::string best;
    std::string cur;
    cur.reserve(8);

    for (char ch : s) {
        if (std::isalpha(static_cast<unsigned char>(ch))) {
            cur.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
        } else {
            if (cur.size() >= 2 && cur.size() <= 6) {
                if (best.empty()) best = cur;
            }
            cur.clear();
        }
    }
    if (best.empty() && cur.size() >= 2 && cur.size() <= 6) {
        best = cur;
    }
    return best;
}

static ROI clampROIToFrame(const ROI& roi, int frameW, int frameH) {
    ROI out = roi;
    if (frameW <= 0 || frameH <= 0) return out;

    out.x = (std::max)(0, out.x);
    out.y = (std::max)(0, out.y);
    out.w = (std::max)(1, out.w);
    out.h = (std::max)(1, out.h);

    if (out.x >= frameW) out.x = frameW - 1;
    if (out.y >= frameH) out.y = frameH - 1;
    if (out.x + out.w > frameW) out.w = (std::max)(1, frameW - out.x);
    if (out.y + out.h > frameH) out.h = (std::max)(1, frameH - out.y);
    return out;
}

static bool computeWindowToCaptureMapping(HWND hwnd, int frameW, int frameH, float& scaleX, float& scaleY, float& clientOffsetX, float& clientOffsetY) {
    if (!hwnd || !IsWindow(hwnd) || frameW <= 0 || frameH <= 0) return false;

    RECT winRc{};
    if (!GetWindowRect(hwnd, &winRc)) return false;
    const int winW = winRc.right - winRc.left;
    const int winH = winRc.bottom - winRc.top;
    if (winW <= 0 || winH <= 0) return false;

    RECT cliRc{};
    if (!GetClientRect(hwnd, &cliRc)) return false;
    const int cliW = cliRc.right - cliRc.left;
    const int cliH = cliRc.bottom - cliRc.top;
    if (cliW <= 0 || cliH <= 0) return false;

    POINT cliTL{ 0, 0 };
    if (!ClientToScreen(hwnd, &cliTL)) return false;
    const int offX = cliTL.x - winRc.left;
    const int offY = cliTL.y - winRc.top;

    // WGC window capture size may correspond to the window or client area (and can be DPI-scaled).
    // Choose whichever basis best matches the current capture frame dimensions.
    const int errWin = std::abs(frameW - winW) + std::abs(frameH - winH);
    const int errCli = std::abs(frameW - cliW) + std::abs(frameH - cliH);

    if (errCli <= errWin) {
        scaleX = static_cast<float>(frameW) / static_cast<float>(cliW);
        scaleY = static_cast<float>(frameH) / static_cast<float>(cliH);
        clientOffsetX = 0.0f;
        clientOffsetY = 0.0f;
    } else {
        scaleX = static_cast<float>(frameW) / static_cast<float>(winW);
        scaleY = static_cast<float>(frameH) / static_cast<float>(winH);
        clientOffsetX = static_cast<float>(offX);
        clientOffsetY = static_cast<float>(offY);
    }

    return (scaleX > 0.0f && scaleY > 0.0f);
}

static ROI mapClientROIToCapture(HWND hwnd, const ROI& roiClient, int frameW, int frameH) {
    float sx = 1.0f, sy = 1.0f, ox = 0.0f, oy = 0.0f;
    if (!computeWindowToCaptureMapping(hwnd, frameW, frameH, sx, sy, ox, oy)) {
        return clampROIToFrame(roiClient, frameW, frameH);
    }

    ROI out = roiClient;
    const float x = (ox + static_cast<float>(roiClient.x)) * sx;
    const float y = (oy + static_cast<float>(roiClient.y)) * sy;
    const float w = static_cast<float>(roiClient.w) * sx;
    const float h = static_cast<float>(roiClient.h) * sy;

    out.x = static_cast<int>(std::lround(x));
    out.y = static_cast<int>(std::lround(y));
    out.w = (std::max)(1, static_cast<int>(std::lround(w)));
    out.h = (std::max)(1, static_cast<int>(std::lround(h)));
    return clampROIToFrame(out, frameW, frameH);
}

struct AnchorMatchResult {
    bool found = false;
    float score = 0.0f;
    float scale = 1.0f;
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

static void bgraToGray(const uint8_t* bgra, int w, int h, std::vector<uint8_t>& outGray) {
    outGray.resize(static_cast<size_t>(w) * static_cast<size_t>(h));
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = bgra + static_cast<size_t>(y) * static_cast<size_t>(w) * 4;
        uint8_t* out = outGray.data() + static_cast<size_t>(y) * static_cast<size_t>(w);
        for (int x = 0; x < w; ++x) {
            const uint8_t b = row[x * 4 + 0];
            const uint8_t g = row[x * 4 + 1];
            const uint8_t r = row[x * 4 + 2];
            const int gray = (static_cast<int>(r) * 77 + static_cast<int>(g) * 150 + static_cast<int>(b) * 29) >> 8;
            out[x] = static_cast<uint8_t>(gray);
        }
    }
}

// Stride-aware BGRA->grayscale (needed for Media Foundation decode frames).
static void bgraToGrayStride(const uint8_t* bgra, int w, int h, int strideBytes,
                             std::vector<uint8_t>& outGray) {
    outGray.resize(static_cast<size_t>(w) * static_cast<size_t>(h));
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = bgra + static_cast<size_t>(y) * static_cast<size_t>(strideBytes);
        uint8_t* out = outGray.data() + static_cast<size_t>(y) * static_cast<size_t>(w);
        for (int x = 0; x < w; ++x) {
            const uint8_t b = row[x * 4 + 0];
            const uint8_t g = row[x * 4 + 1];
            const uint8_t r = row[x * 4 + 2];
            const int gray = (static_cast<int>(r) * 77 + static_cast<int>(g) * 150 + static_cast<int>(b) * 29) >> 8;
            out[x] = static_cast<uint8_t>(gray);
        }
    }
}

static std::vector<uint8_t> resizeGrayNearest(const std::vector<uint8_t>& src, int srcW, int srcH, int dstW, int dstH) {
    std::vector<uint8_t> dst(static_cast<size_t>(dstW) * static_cast<size_t>(dstH));
    for (int y = 0; y < dstH; ++y) {
        const int sy = (y * srcH) / dstH;
        const uint8_t* srcRow = src.data() + static_cast<size_t>(sy) * static_cast<size_t>(srcW);
        uint8_t* dstRow = dst.data() + static_cast<size_t>(y) * static_cast<size_t>(dstW);
        for (int x = 0; x < dstW; ++x) {
            const int sx = (x * srcW) / dstW;
            dstRow[x] = srcRow[sx];
        }
    }
    return dst;
}

static std::vector<uint8_t> cropGrayRegion(const std::vector<uint8_t>& src, int srcW, int srcH, int x, int y, int w, int h) {
    const int rx = (std::max)(0, (std::min)(x, srcW - 1));
    const int ry = (std::max)(0, (std::min)(y, srcH - 1));
    const int rw = (std::max)(1, (std::min)(w, srcW - rx));
    const int rh = (std::max)(1, (std::min)(h, srcH - ry));

    std::vector<uint8_t> out(static_cast<size_t>(rw) * static_cast<size_t>(rh));
    for (int yy = 0; yy < rh; ++yy) {
        const uint8_t* srcRow = src.data() + static_cast<size_t>(ry + yy) * static_cast<size_t>(srcW) + static_cast<size_t>(rx);
        uint8_t* dstRow = out.data() + static_cast<size_t>(yy) * static_cast<size_t>(rw);
        std::memcpy(dstRow, srcRow, static_cast<size_t>(rw));
    }
    return out;
}

static std::vector<float> parseScalesCsv(const std::string& csv) {
    std::vector<float> scales;
    if (csv.empty()) return scales;
    std::istringstream iss(csv);
    std::string token;
    while (std::getline(iss, token, ',')) {
        try {
            float v = std::stof(token);
            if (v > 0.05f) scales.push_back(v);
        } catch (...) {
        }
    }
    return scales;
}

static bool parseOffsetRect(const std::string& s, float& dx, float& dy, float& w, float& h) {
    if (s.empty()) return false;
    float vals[4] = {0, 0, 0, 0};
    int count = sscanf(s.c_str(), "%f,%f,%f,%f", &vals[0], &vals[1], &vals[2], &vals[3]);
    if (count != 4) return false;
    dx = vals[0];
    dy = vals[1];
    w = vals[2];
    h = vals[3];
    return true;
}

static AnchorMatchResult matchTemplateNCC(
    const std::vector<uint8_t>& imgGray, int imgW, int imgH,
    const std::vector<uint8_t>& tplGray, int tplW, int tplH
) {
    AnchorMatchResult best;
    if (tplW <= 0 || tplH <= 0 || imgW < tplW || imgH < tplH) return best;

    const int n = tplW * tplH;
    double sumT = 0.0;
    double sumT2 = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = tplGray[static_cast<size_t>(i)];
        sumT += v;
        sumT2 += v * v;
    }
    const double meanT = sumT / n;
    const double varT = sumT2 - sumT * meanT;
    if (varT <= 1e-6) return best;

    // Integral images for sum and sumsq
    const int W = imgW + 1;
    const int H = imgH + 1;
    std::vector<double> integral(static_cast<size_t>(W) * static_cast<size_t>(H), 0.0);
    std::vector<double> integral2(static_cast<size_t>(W) * static_cast<size_t>(H), 0.0);
    for (int y = 1; y <= imgH; ++y) {
        double rowSum = 0.0;
        double rowSum2 = 0.0;
        const uint8_t* row = imgGray.data() + static_cast<size_t>(y - 1) * static_cast<size_t>(imgW);
        for (int x = 1; x <= imgW; ++x) {
            const double v = row[x - 1];
            rowSum += v;
            rowSum2 += v * v;
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x);
            integral[idx] = integral[idx - W] + rowSum;
            integral2[idx] = integral2[idx - W] + rowSum2;
        }
    }

    auto rectSum = [&](const std::vector<double>& integ, int x, int y, int w, int h) {
        const int x2 = x + w;
        const int y2 = y + h;
        const size_t A = static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x);
        const size_t B = static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x2);
        const size_t C = static_cast<size_t>(y2) * static_cast<size_t>(W) + static_cast<size_t>(x);
        const size_t D = static_cast<size_t>(y2) * static_cast<size_t>(W) + static_cast<size_t>(x2);
        return integ[D] - integ[B] - integ[C] + integ[A];
    };

    best.score = -1.0f;
    for (int y = 0; y <= imgH - tplH; ++y) {
        for (int x = 0; x <= imgW - tplW; ++x) {
            const double sumI = rectSum(integral, x, y, tplW, tplH);
            const double sumI2 = rectSum(integral2, x, y, tplW, tplH);
            const double meanI = sumI / n;
            const double varI = sumI2 - sumI * meanI;
            if (varI <= 1e-6) continue;

            double sumIT = 0.0;
            for (int ty = 0; ty < tplH; ++ty) {
                const uint8_t* imgRow = imgGray.data() + static_cast<size_t>(y + ty) * static_cast<size_t>(imgW) + static_cast<size_t>(x);
                const uint8_t* tplRow = tplGray.data() + static_cast<size_t>(ty) * static_cast<size_t>(tplW);
                for (int tx = 0; tx < tplW; ++tx) {
                    sumIT += static_cast<double>(imgRow[tx]) * static_cast<double>(tplRow[tx]);
                }
            }

            const double numerator = sumIT - sumI * meanT - sumT * meanI + static_cast<double>(n) * meanI * meanT;
            const double denom = std::sqrt(varI * varT);
            if (denom <= 1e-6) continue;
            const float ncc = static_cast<float>(numerator / denom);
            if (!best.found || ncc > best.score) {
                best.found = true;
                best.score = ncc;
                best.x = x;
                best.y = y;
                best.w = tplW;
                best.h = tplH;
            }
        }
    }

    return best;
}

static AnchorMatchResult findAnchorByTemplate(
    const std::vector<uint8_t>& imgGray, int imgW, int imgH,
    const std::vector<uint8_t>& tplGray, int tplW, int tplH,
    const std::vector<float>& scales, float minScore, float maxSearchW
) {
    AnchorMatchResult best;

    float searchScale = 1.0f;
    if (maxSearchW > 0.0f && imgW > static_cast<int>(maxSearchW)) {
        searchScale = maxSearchW / static_cast<float>(imgW);
    }
    int searchW = static_cast<int>(std::round(imgW * searchScale));
    int searchH = static_cast<int>(std::round(imgH * searchScale));
    if (searchW <= 0 || searchH <= 0) return best;

    const std::vector<uint8_t> imgSearch = (searchScale == 1.0f)
        ? imgGray
        : resizeGrayNearest(imgGray, imgW, imgH, searchW, searchH);

    for (float s : scales) {
        if (s <= 0.05f) continue;
        const int tplScaledW = static_cast<int>(std::round(tplW * s));
        const int tplScaledH = static_cast<int>(std::round(tplH * s));
        if (tplScaledW < 4 || tplScaledH < 4) continue;

        const int tplSearchW = static_cast<int>(std::round(tplScaledW * searchScale));
        const int tplSearchH = static_cast<int>(std::round(tplScaledH * searchScale));
        if (tplSearchW < 4 || tplSearchH < 4 || tplSearchW > searchW || tplSearchH > searchH) continue;

        std::vector<uint8_t> tplScaled = resizeGrayNearest(tplGray, tplW, tplH, tplScaledW, tplScaledH);
        std::vector<uint8_t> tplSearch = (searchScale == 1.0f)
            ? tplScaled
            : resizeGrayNearest(tplScaled, tplScaledW, tplScaledH, tplSearchW, tplSearchH);

        AnchorMatchResult res = matchTemplateNCC(imgSearch, searchW, searchH, tplSearch, tplSearchW, tplSearchH);
        if (res.found && res.score > best.score) {
            best = res;
            best.scale = s;
        }
    }

    if (!best.found || best.score < minScore) {
        best.found = false;
        return best;
    }

    // Map back to original resolution
    if (searchScale != 1.0f) {
        best.x = static_cast<int>(std::round(best.x / searchScale));
        best.y = static_cast<int>(std::round(best.y / searchScale));
    }
    best.w = static_cast<int>(std::round(tplW * best.scale));
    best.h = static_cast<int>(std::round(tplH * best.scale));
    return best;
}

static bool validateSecondaryAnchor(
    const std::vector<uint8_t>& imgGray, int imgW, int imgH,
    const AnchorMatchResult& primary,
    const std::vector<uint8_t>& secondaryTpl, int secondaryTplW, int secondaryTplH,
    bool secondaryOffsetValid, float offX, float offY, float offW, float offH,
    float threshold
) {
    if (secondaryTpl.empty() || secondaryTplW <= 0 || secondaryTplH <= 0) return true;

    int searchX = primary.x;
    int searchY = primary.y;
    int searchW = primary.w;
    int searchH = primary.h;

    if (secondaryOffsetValid) {
        searchX = static_cast<int>(std::lround(primary.x + offX * primary.scale));
        searchY = static_cast<int>(std::lround(primary.y + offY * primary.scale));
        const int w = static_cast<int>(std::lround(offW * primary.scale));
        const int h = static_cast<int>(std::lround(offH * primary.scale));
        if (w > 0) searchW = w;
        if (h > 0) searchH = h;
    }

    searchX = (std::max)(0, (std::min)(searchX, imgW - 1));
    searchY = (std::max)(0, (std::min)(searchY, imgH - 1));
    searchW = (std::max)(1, (std::min)(searchW, imgW - searchX));
    searchH = (std::max)(1, (std::min)(searchH, imgH - searchY));

    const std::vector<uint8_t> searchGray = cropGrayRegion(imgGray, imgW, imgH, searchX, searchY, searchW, searchH);
    AnchorMatchResult sec = matchTemplateNCC(searchGray, searchW, searchH, secondaryTpl, secondaryTplW, secondaryTplH);
    return sec.found && sec.score >= threshold;
}

static bool loadTemplateGray(const std::string& path, std::vector<uint8_t>& tplGray, int& tplW, int& tplH) {
    tplGray.clear();
    tplW = tplH = 0;
    if (path.empty()) return false;

    int w = 0, h = 0, c = 0;
    stbi_uc* data = stbi_load(path.c_str(), &w, &h, &c, 4);
    if (!data) return false;
    tplW = w;
    tplH = h;
    bgraToGray(data, w, h, tplGray);
    stbi_image_free(data);
    return !tplGray.empty();
}

static std::string resolvePathWithFallbacks(const std::string& path) {
    if (path.empty()) return path;
    if (fs::exists(path)) return path;

    // Common when running from build/ or build\Debug/Release.
    const std::string up1 = std::string("../") + path;
    if (fs::exists(up1)) return up1;

    const std::string up2 = std::string("../../") + path;
    if (fs::exists(up2)) return up2;

    return path;
}

static void autoSelectDefaultWindowIfNeeded(
    bool monitorWasSpecified,
    bool windowWasSpecified,
    void*& selectedHwnd,
    std::string& windowTitle
) {
    if (monitorWasSpecified) return;
    if (windowWasSpecified) return;
    if (selectedHwnd) return;
    if (!windowTitle.empty()) return;

    const std::string kPreferredTitle = "WarriorTrading Chatroom - Brave";
    auto windows = enumerateWindows();
    for (const auto& w : windows) {
        if (w.second == kPreferredTitle || w.second.find(kPreferredTitle) != std::string::npos) {
            selectedHwnd = w.first;
            std::cout << "Auto-selected window: " << w.second << std::endl;
            return;
        }
    }
}

int main(int argc, char* argv[]) {
    printBanner();
    
    // Parse command line arguments
    std::string configPath = "config/roi_config.json";
    std::string windowTitle = "";
    int monitorIndex = 0;
    bool monitorSpecified = false;
    bool windowSpecified = false;
    bool benchmarkMode = false;
    bool verbose = false;
    bool enableChangeDetect = true;
    int changeThreshold = 5;

    bool listWindows = false;
    bool selectWindow = false;
    bool listMonitors = false;
    bool selectMonitor = false;
    bool listRois = false;
    bool selectRoi = false;
    bool createRoi = false;
    std::string createRoiName;

    bool dumpRoi = false;
    std::string dumpRoiDir;

    float ocrZoom = 1.0f;

    bool showRoiOverlay = false;

    bool scanTable = false;
    std::string scanTableRoiName;
    int tableRows = 6;
    int rowHeightPx = 18;
    int rowStridePx = 18;
    int rowOffsetYPx = 0;
    bool printAllRows = false;
    bool autoRows = false;
    std::string imagePath;  // Static image input path
    std::string testRoiStr; // Custom ROI coordinates "x,y,w,h"

    // Offline MP4 input (Media Foundation)
    std::string videoPath;
    int videoMaxFrames = 0; // 0 = decode until EOF

    // Entry-trigger mode (timing-first)
    std::string modeStr = "panel-detect"; // panel-detect | entry-trigger
    std::string targetSymbol;
    double delayCompMs = 0.0;
    bool enableProfile = false;
    std::string profileJsonPath;
    std::string profileSummaryPath;

    // Phase 2: Panel auto-detection (offline with --video)
    bool detectPanels = false;
    std::string panelTemplateDir = "templates/headers";
    float panelThreshold = 0.65f;
    int panelEveryN = 1;

    // Column extraction parameters
    std::string columnsDefStr;  // Multiple columns: "symbol:0,50;pnl:145,40"
    int colOffsetX = 0;   // Single column X offset (0 = use full row width)
    int colWidth = 0;     // Single column width (0 = use full row width)

    // Parsed column definitions: vector of (name, x_offset, width)
    struct ColumnDef {
        std::string name;
        int xOffset;
        int width;
    };
    std::vector<ColumnDef> columnDefs;

    std::string rowDetectModeStr = "intensity";
    std::string jsonOutputPath;  // JSON output file path
    std::string templateDir = "templates";
    float templateThreshold = 0.7f;
    int templateRowHeight = 14;
    int templateRowSpacing = 2;
    int templateMaxRows = 20;

    std::string anchorTemplatePath;
    std::string anchorOffsetStr;
    std::string anchorScalesStr;
    float anchorThreshold = 0.55f;
    int anchorEveryN = 0;
    float anchorMaxSearchW = 640.0f;
    std::string anchorSearchStr;

    std::string anchorSecondaryTemplatePath;
    std::string anchorSecondaryOffsetStr;
    float anchorSecondaryThreshold = 0.6f;

    // YOLOv10 window detection parameters
    std::string detectModelPath;     // Path to YOLOv10 engine/ONNX
    std::string detectOnnxPath;      // ONNX for auto-build
    float detectionConfidence = 0.8f;
    int detectEveryN = 5;            // Run detection every N frames
    std::string detectClassNamesStr; // Comma-separated class names
    bool useYoloDetection = false;

    void* selectedHwnd = nullptr;
    void* selectedMonitor = nullptr;

    int selectWindowIndex = -1;
    int selectMonitorIndex = -1;
    int selectRoiIndex = -1;

    std::string enginePath = "models/en_PP-OCRv4_rec.engine";
    std::string onnxPath = "models/en_PP-OCRv4_rec.onnx";
    std::string dictPath = "models/en_PP-OCRv4_dict.txt";

    std::string onlyRoiName;
    bool allRois = false;
    int everyN = 5;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--video" && i + 1 < argc) {
            videoPath = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            modeStr = argv[++i];
        } else if (arg == "--target" && i + 1 < argc) {
            targetSymbol = argv[++i];
        } else if (arg == "--delay-compensation-ms" && i + 1 < argc) {
            delayCompMs = std::stod(argv[++i]);
        } else if (arg == "--profile") {
            enableProfile = true;
        } else if (arg == "--profile-output" && i + 1 < argc) {
            profileJsonPath = argv[++i];
        } else if (arg == "--profile-summary" && i + 1 < argc) {
            profileSummaryPath = argv[++i];
        } else if (arg == "--video-max-frames" && i + 1 < argc) {
            videoMaxFrames = (std::max)(0, std::stoi(argv[++i]));
        } else if (arg == "--detect-panels") {
            detectPanels = true;
        } else if (arg == "--panel-template-dir" && i + 1 < argc) {
            panelTemplateDir = argv[++i];
        } else if (arg == "--panel-threshold" && i + 1 < argc) {
            panelThreshold = std::stof(argv[++i]);
        } else if (arg == "--panel-every" && i + 1 < argc) {
            panelEveryN = (std::max)(1, std::stoi(argv[++i]));
        } else if (arg == "--config" && i + 1 < argc) {
            configPath = argv[++i];
        } else if (arg == "--window" && i + 1 < argc) {
            windowTitle = argv[++i];
            windowSpecified = true;
        } else if (arg == "--list-windows") {
            listWindows = true;
        } else if (arg == "--select-window") {
            selectWindow = true;
            windowSpecified = true;
            if (i + 1 < argc) {
                std::string next = argv[i + 1];
                if (!next.empty() && next[0] != '-' && looksLikeInt(next)) {
                    selectWindowIndex = std::stoi(next);
                    i++;
                }
            }
        } else if (arg == "--monitor" && i + 1 < argc) {
            monitorIndex = std::stoi(argv[++i]);
            monitorSpecified = true;
        } else if (arg == "--list-monitors") {
            listMonitors = true;
        } else if (arg == "--select-monitor") {
            selectMonitor = true;
            monitorSpecified = true;
            if (i + 1 < argc) {
                std::string next = argv[i + 1];
                if (!next.empty() && next[0] != '-' && looksLikeInt(next)) {
                    selectMonitorIndex = std::stoi(next);
                    i++;
                }
            }
        } else if (arg == "--list-rois") {
            listRois = true;
        } else if (arg == "--roi" && i + 1 < argc) {
            onlyRoiName = argv[++i];
        } else if (arg == "--select-roi") {
            selectRoi = true;
            if (i + 1 < argc) {
                std::string next = argv[i + 1];
                if (!next.empty() && next[0] != '-' && looksLikeInt(next)) {
                    selectRoiIndex = std::stoi(next);
                    i++;
                }
            }
        } else if (arg == "--create-roi") {
            createRoi = true;
            if (i + 1 < argc) {
                std::string next = argv[i + 1];
                if (!next.empty() && next[0] != '-') {
                    createRoiName = next;
                    i++;
                }
            }
        } else if (arg == "--all-rois") {
            allRois = true;
        } else if (arg == "--every" && i + 1 < argc) {
            everyN = (std::max)(1, std::stoi(argv[++i]));
        } else if (arg == "--engine" && i + 1 < argc) {
            enginePath = argv[++i];
        } else if (arg == "--onnx" && i + 1 < argc) {
            onnxPath = argv[++i];
        } else if (arg == "--dict" && i + 1 < argc) {
            dictPath = argv[++i];
        } else if (arg == "--no-change-detect") {
            enableChangeDetect = false;
        } else if (arg == "--change-threshold" && i + 1 < argc) {
            changeThreshold = (std::max)(0, std::stoi(argv[++i]));
        } else if (arg == "--dump-roi" && i + 1 < argc) {
            dumpRoi = true;
            dumpRoiDir = argv[++i];
        } else if (arg == "--ocr-zoom" && i + 1 < argc) {
            ocrZoom = (std::max)(1.0f, std::stof(argv[++i]));
        } else if (arg == "--show-roi") {
            showRoiOverlay = true;
        } else if (arg == "--scan-table") {
            scanTable = true;
            if (i + 1 < argc) {
                std::string next = argv[i + 1];
                if (!next.empty() && next[0] != '-') {
                    scanTableRoiName = next;
                    i++;
                }
            }
        } else if (arg == "--table-rows" && i + 1 < argc) {
            tableRows = (std::max)(1, std::stoi(argv[++i]));
        } else if (arg == "--row-height" && i + 1 < argc) {
            rowHeightPx = (std::max)(1, std::stoi(argv[++i]));
        } else if (arg == "--row-stride" && i + 1 < argc) {
            rowStridePx = (std::max)(1, std::stoi(argv[++i]));
        } else if (arg == "--row-offset-y" && i + 1 < argc) {
            rowOffsetYPx = std::stoi(argv[++i]);
        } else if (arg == "--columns" && i + 1 < argc) {
            columnsDefStr = argv[++i];
        } else if (arg == "--col-x" && i + 1 < argc) {
            colOffsetX = std::stoi(argv[++i]);
        } else if (arg == "--col-w" && i + 1 < argc) {
            colWidth = (std::max)(1, std::stoi(argv[++i]));
        } else if (arg == "--print-all-rows") {
            printAllRows = true;
        } else if (arg == "--auto-rows") {
            autoRows = true;
        } else if (arg == "--row-detect-mode" && i + 1 < argc) {
            rowDetectModeStr = argv[++i];
        } else if (arg == "--anchor-template" && i + 1 < argc) {
            anchorTemplatePath = argv[++i];
        } else if (arg == "--anchor-offset" && i + 1 < argc) {
            anchorOffsetStr = argv[++i];
        } else if (arg == "--anchor-scales" && i + 1 < argc) {
            anchorScalesStr = argv[++i];
        } else if (arg == "--anchor-threshold" && i + 1 < argc) {
            anchorThreshold = std::stof(argv[++i]);
        } else if (arg == "--anchor-max-search" && i + 1 < argc) {
            anchorMaxSearchW = std::stof(argv[++i]);
        } else if (arg == "--anchor-search" && i + 1 < argc) {
            anchorSearchStr = argv[++i];
        } else if (arg == "--anchor-secondary-template" && i + 1 < argc) {
            anchorSecondaryTemplatePath = argv[++i];
        } else if (arg == "--anchor-secondary-offset" && i + 1 < argc) {
            anchorSecondaryOffsetStr = argv[++i];
        } else if (arg == "--anchor-secondary-threshold" && i + 1 < argc) {
            anchorSecondaryThreshold = std::stof(argv[++i]);
        } else if (arg == "--anchor-every" && i + 1 < argc) {
            anchorEveryN = (std::max)(0, std::stoi(argv[++i]));
        } else if (arg == "--template-dir" && i + 1 < argc) {
            templateDir = argv[++i];
        } else if (arg == "--template-threshold" && i + 1 < argc) {
            templateThreshold = std::stof(argv[++i]);
        } else if (arg == "--template-row-height" && i + 1 < argc) {
            templateRowHeight = (std::max)(1, std::stoi(argv[++i]));
        } else if (arg == "--template-row-spacing" && i + 1 < argc) {
            templateRowSpacing = (std::max)(0, std::stoi(argv[++i]));
        } else if (arg == "--template-max-rows" && i + 1 < argc) {
            templateMaxRows = (std::max)(1, std::stoi(argv[++i]));
        } else if (arg == "--image" && i + 1 < argc) {
            imagePath = argv[++i];
        } else if (arg == "--test-roi" && i + 1 < argc) {
            testRoiStr = argv[++i];
        } else if (arg == "--json-output" && i + 1 < argc) {
            jsonOutputPath = argv[++i];
        } else if (arg == "--detect-model" && i + 1 < argc) {
            detectModelPath = argv[++i];
            useYoloDetection = true;
        } else if (arg == "--detect-onnx" && i + 1 < argc) {
            detectOnnxPath = argv[++i];
            useYoloDetection = true;
        } else if (arg == "--detect-confidence" && i + 1 < argc) {
            detectionConfidence = std::stof(argv[++i]);
        } else if (arg == "--detect-every" && i + 1 < argc) {
            detectEveryN = (std::max)(1, std::stoi(argv[++i]));
        } else if (arg == "--detect-classes" && i + 1 < argc) {
            detectClassNamesStr = argv[++i];
        } else if (arg == "--benchmark") {
            benchmarkMode = true;
        } else if (arg == "--verbose") {
            verbose = true;
        }
    }
    
    auto parseRowDetectMode = [](const std::string& mode) {
        std::string m = mode;
        for (auto& c : m) c = static_cast<char>(std::tolower(c));
        if (m == "template") return RowDetectionMode::TEMPLATE_BASED;
        if (m == "hybrid") return RowDetectionMode::HYBRID;
        return RowDetectionMode::INTENSITY_BASED;
    };

    // ---------------------------------------------------------------------
    // Phase 1: Offline MP4 decode mode (no CUDA/WinRT required)
    // ---------------------------------------------------------------------
    if (!videoPath.empty()) {
        std::string err;
        std::string resolved = resolvePathWithFallbacks(videoPath);

        const bool entryTriggerMode = (modeStr == "entry-trigger");
        if (entryTriggerMode && targetSymbol.empty()) {
            std::cerr << "ERROR: --mode entry-trigger requires --target <SYMBOL>\n";
            return 1;
        }

        trading_monitor::detect::PanelFinder panelFinder;
        trading_monitor::detect::PanelFinderConfig panelCfg;
        panelCfg.hdrThreshold = panelThreshold;
        panelCfg.maxSearchW = 640.0f;

        if (detectPanels || entryTriggerMode) {
            const std::string resolvedDir = resolvePathWithFallbacks(panelTemplateDir);
            const std::string positionsHdr = (fs::path(resolvedDir) / "positions_hdr.png").string();
            const std::string orderHdr = (fs::path(resolvedDir) / "order_hdr.png").string();
            const std::string quoteHdr = (fs::path(resolvedDir) / "quote_hdr.png").string();
            if (!panelFinder.loadTemplates(positionsHdr, orderHdr, quoteHdr, err)) {
                std::cerr << "ERROR: Panel template load failed: " << err << "\n";
                std::cerr << "Expected files in: " << resolvedDir << "\n";
                std::cerr << "  positions_hdr.png, order_hdr.png, quote_hdr.png\n";
                return 1;
            }
        }

        trading_monitor::video::MFSourceReaderDecoder dec;
        if (!dec.open(fs::path(resolved).wstring(), err)) {
            std::cerr << "ERROR: Video open failed: " << err << "\n";
            return 1;
        }

        std::cout << "Offline video decode (Media Foundation)\n";
        std::cout << "  Source: " << resolved << "\n";
        std::cout << "  Size:   " << dec.width() << "x" << dec.height() << "\n";
        if (dec.fps() > 0.0) {
            std::cout << "  FPS~:   " << std::fixed << std::setprecision(3) << dec.fps() << "\n";
        }

        trading_monitor::video::VideoFrameBGRA frame;
        uint64_t count = 0;
        const auto t0 = std::chrono::steady_clock::now();

        uint64_t detectCount = 0;

        // Entry-trigger pipeline state (optional)
        trading_monitor::detect::SymbolMatcher sym;
        trading_monitor::detect::EntryTrigger entry({});
        trading_monitor::detect::TriggerRoiBuilder trigBuilder;
        trading_monitor::detect::TriggerRoiConfig trigCfg;
        trading_monitor::detect::TriggerRoiResult trigRoi;
        trading_monitor::track::PanelTracker tracker;
        trading_monitor::track::TrackerConfig trackCfg;
        trading_monitor::Profiler profiler(enableProfile);
        uint64_t lastTrigBuildFrame = 0;

        trading_monitor::track::HeaderTemplate hdrTpl;
        if (entryTriggerMode) {
            // Load symbol templates (expected: templates/symbols/<SYMBOL>_*.png)
            const std::string symDir = resolvePathWithFallbacks("templates/symbols");
            if (!sym.loadSymbolTemplates(symDir, targetSymbol, err)) {
                std::cerr << "ERROR: Symbol template load failed: " << err << "\n";
                return 1;
            }
            trading_monitor::detect::EntryTriggerConfig ecfg;
            ecfg.delayCompensationMs = delayCompMs;
            entry = trading_monitor::detect::EntryTrigger(ecfg);
            entry.setTargetSymbol(targetSymbol);

            // Load the positions header template into a HeaderTemplate for tracking.
            const std::string resolvedDir = resolvePathWithFallbacks(panelTemplateDir);
            const std::string positionsHdr = (fs::path(resolvedDir) / "positions_hdr.png").string();
            if (!loadTemplateGray(positionsHdr, hdrTpl.gray, hdrTpl.w, hdrTpl.h)) {
                std::cerr << "ERROR: Failed to load positions header for tracking: " << positionsHdr << "\n";
                return 1;
            }
        }

        while (g_running) {
            if (videoMaxFrames > 0 && count >= static_cast<uint64_t>(videoMaxFrames)) break;
            if (!dec.readFrame(frame, err)) break;
            count++;

            if (entryTriggerMode) {
                profiler.beginFrame(frame.frameIndex);

                // BGRA->gray (stride-aware)
                std::vector<uint8_t> gray;
                bgraToGrayStride(frame.pixels.data(), frame.width, frame.height, frame.strideBytes, gray);
                profiler.markGray();

                // Acquire positions panel (once) or re-acquire if tracking fails
                static bool havePanel = false;
                static trading_monitor::detect::FoundPanels found{};

                if (!havePanel) {
                    found = panelFinder.findPanelsFromBGRA(frame.pixels.data(), frame.width, frame.height, frame.strideBytes, panelCfg);
                    profiler.markFind();
                    if (!found.hasPositions) {
                        profiler.endFrame();
                        continue;
                    }
                    tracker.init(hdrTpl, found.positionsHeader, found.positionsPanel);
                    havePanel = true;
                } else {
                    auto tr = tracker.update(gray, frame.width, frame.height, trackCfg, frame.frameIndex, true);
                    profiler.markTrack();
                    if (!tr.ok) {
                        havePanel = false;
                        profiler.endFrame();
                        continue;
                    }
                    found.positionsPanel = tr.panelRect;
                    found.positionsHeader = tr.headerRect;
                }

                // Build trigger ROI periodically (no hardcoded columns)
                if (!trigRoi.ok || (frame.frameIndex - lastTrigBuildFrame) >= (uint64_t)trigCfg.rebuildEveryNFrames) {
                    trigRoi = trigBuilder.build(gray, frame.width, frame.height, found.positionsPanel, trigCfg);
                    lastTrigBuildFrame = frame.frameIndex;
                    profiler.markRoiBuild();
                }
                if (!trigRoi.ok) {
                    profiler.endFrame();
                    continue;
                }

                // Match target symbol in trigger ROI
                float trigScore = sym.matchInGrayROI(gray, frame.width, frame.height, trigRoi.triggerRoi);
                profiler.markMatch(trigScore);

                auto ev = entry.update(frame.frameIndex, frame.timestampMs, -1.f, -1.f, trigScore);
                profiler.markState(entry.state().armed, entry.state().triggered);
                profiler.endFrame();

                if (ev.fired) {
                    profiler.flush(profileJsonPath, profileSummaryPath);
                    std::cout << "\n[entry-trigger] " << targetSymbol
                              << " window_ms=(" << std::fixed << std::setprecision(2)
                              << ev.absentLastTimeMs << "," << ev.presentFirstTimeMs << "]"
                              << " est_ms=(" << ev.absentLastTimeEstMs << "," << ev.presentFirstTimeEstMs << "]\n";
                    return 0;
                }
                continue;
            }

            if (detectPanels && ((count - 1) % static_cast<uint64_t>(panelEveryN) == 0)) {
                const auto found = panelFinder.findPanelsFromBGRA(
                    frame.pixels.data(), frame.width, frame.height, frame.strideBytes, panelCfg);

                std::cout << "\n[panel-detect] frame=" << count << " ts_ms=" << std::fixed << std::setprecision(2)
                          << frame.timestampMs << "\n";

                if (found.hasPositions) {
                    std::cout << "  positions: score=" << std::fixed << std::setprecision(3) << found.scorePositions << "\n";
                    printROI(found.positionsHeader);
                    printROI(found.positionsPanel);
                } else {
                    std::cout << "  positions: (not found)\n";
                }

                if (found.hasOrder) {
                    std::cout << "  order: score=" << std::fixed << std::setprecision(3) << found.scoreOrder << "\n";
                    printROI(found.orderHeader);
                    printROI(found.orderPanel);
                } else {
                    std::cout << "  order: (not found)\n";
                }

                if (found.hasQuote) {
                    std::cout << "  quote: score=" << std::fixed << std::setprecision(3) << found.scoreQuote << "\n";
                    printROI(found.quoteHeader);
                    printROI(found.quotePanel);
                } else {
                    std::cout << "  quote: (not found)\n";
                }

                detectCount++;
            }
        }

        if (entryTriggerMode) {
            profiler.flush(profileJsonPath, profileSummaryPath);
        }

        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "  Decoded frames: " << count << "\n";
        if (detectPanels) {
            std::cout << "  Panel detections: " << detectCount << "\n";
        }
        std::cout << "  Wall time:      " << std::fixed << std::setprecision(2) << ms << " ms\n";

        if (!err.empty() && err != "EOF") {
            std::cout << "  Note: decode ended with: " << err << "\n";
        }

        return 0;
    }

    // Setup signal handler
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    // Check CUDA
    int cudaDeviceCount = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&cudaDeviceCount);
    if (cudaStatus != cudaSuccess || cudaDeviceCount == 0) {
        std::cerr << "ERROR: No CUDA devices found. "
                  << "Ensure NVIDIA drivers and CUDA 13.1.1 are installed.\n";
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "CUDA Device: " << prop.name 
              << " (Compute " << prop.major << "." << prop.minor << ")\n";
    std::cout << "CUDA Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB\n";
    std::cout << std::endl;

    // WinRT apartment for Windows.Graphics.Capture
    winrt::init_apartment(winrt::apartment_type::multi_threaded);

    // Window/monitor discovery helpers (can exit early)
    if (listWindows) {
        printWindowsList(enumerateWindows());
        return 0;
    }
    if (listMonitors) {
        printMonitorsList(enumerateMonitors());
        return 0;
    }

    // Resolve paths early so ROI listing/selection works when running from build/.
    configPath = resolvePathWithFallbacks(configPath);
    enginePath = resolvePathWithFallbacks(enginePath);
    dictPath = resolvePathWithFallbacks(dictPath);
    if (!onnxPath.empty()) {
        onnxPath = resolvePathWithFallbacks(onnxPath);
    }
    if (!anchorTemplatePath.empty()) {
        anchorTemplatePath = resolvePathWithFallbacks(anchorTemplatePath);
    }
    if (!anchorSecondaryTemplatePath.empty()) {
        anchorSecondaryTemplatePath = resolvePathWithFallbacks(anchorSecondaryTemplatePath);
    }

    const auto configureRowDetector = [&](RowDetector& detector) -> bool {
        detector.setMinRowHeight(6);
        detector.setTextIntensityThreshold(60);

        RowDetectionMode mode = parseRowDetectMode(rowDetectModeStr);
        detector.setDetectionMode(mode);

        if (mode == RowDetectionMode::TEMPLATE_BASED || mode == RowDetectionMode::HYBRID) {
            std::string resolvedTemplateDir = resolvePathWithFallbacks(templateDir);
            if (!detector.loadTemplates(resolvedTemplateDir)) {
                std::cerr << "WARNING: Failed to load templates from " << resolvedTemplateDir
                          << ", falling back to intensity mode." << std::endl;
                detector.setDetectionMode(RowDetectionMode::INTENSITY_BASED);
                return false;
            }

            if (auto* matcher = detector.getTemplateMatcher()) {
                trading::TemplateMatcherConfig cfg = matcher->getConfig();
                cfg.matchThreshold = templateThreshold;
                cfg.rowHeight = templateRowHeight;
                cfg.rowSpacing = templateRowSpacing;
                cfg.maxRows = templateMaxRows;
                matcher->setConfig(cfg);
            }
            return true;
        }

        return false;
    };

    PipelineConfig cfgForRoiList = loadPipelineConfig(configPath);
    if (listRois) {
        if (cfgForRoiList.rois.empty()) {
            std::cerr << "ERROR: No ROIs loaded from config: " << configPath << std::endl;
            return 1;
        }
        printRoisList(cfgForRoiList);
        return 0;
    }

    if (selectWindow) {
        auto windows = enumerateWindows();
        printWindowsList(windows);
        if (windows.empty()) {
            std::cerr << "ERROR: No capturable windows found." << std::endl;
            return 1;
        }
        int idx = selectWindowIndex;
        if (idx < 0) {
            std::cout << "Select window index: ";
            if (!(std::cin >> idx)) {
                std::cerr << "ERROR: Failed to read index." << std::endl;
                return 1;
            }
        }
        if (idx < 0 || idx >= static_cast<int>(windows.size())) {
            std::cerr << "ERROR: Index out of range." << std::endl;
            return 1;
        }
        selectedHwnd = windows[static_cast<size_t>(idx)].first;
        std::cout << "Selected window: [" << idx << "] " << windows[static_cast<size_t>(idx)].second << std::endl;
        windowTitle = "";
    }

    if (selectMonitor) {
        auto monitors = enumerateMonitors();
        printMonitorsList(monitors);
        if (monitors.empty()) {
            std::cerr << "ERROR: No monitors found." << std::endl;
            return 1;
        }
        int idx = selectMonitorIndex;
        if (idx < 0) {
            std::cout << "Select monitor index: ";
            if (!(std::cin >> idx)) {
                std::cerr << "ERROR: Failed to read index." << std::endl;
                return 1;
            }
        }
        if (idx < 0 || idx >= static_cast<int>(monitors.size())) {
            std::cerr << "ERROR: Index out of range." << std::endl;
            return 1;
        }
        selectedMonitor = monitors[static_cast<size_t>(idx)].first;
        // We will still call initializeMonitor(monitorIndex) later; convert to index.
        monitorIndex = idx;
    }

    if (createRoi) {
        if (!createRoiName.empty() && !onlyRoiName.empty() && createRoiName != onlyRoiName) {
            std::cerr << "ERROR: Both --roi and --create-roi specify different names." << std::endl;
            return 1;
        }

        if (createRoiName.empty()) {
            std::cout << "Enter new ROI name: ";
            if (!(std::cin >> createRoiName)) {
                std::cerr << "ERROR: Failed to read ROI name." << std::endl;
                return 1;
            }
        }

        // If user didn't specify a capture target, try to auto-pick the preferred window.
        autoSelectDefaultWindowIfNeeded(monitorSpecified, windowSpecified, selectedHwnd, windowTitle);

        // ROI creation requires a window target (not a monitor capture).
        if (!selectedHwnd && windowTitle.empty()) {
            std::cerr << "ERROR: --create-roi requires a window target. Use --select-window or --window <title>." << std::endl;
            std::cerr << "Tip: If your target is 'WarriorTrading Chatroom - Brave', just run --create-roi and it will auto-select when available." << std::endl;
            return 1;
        }

        // If user specified --window <title>, resolve it to an HWND for the overlay.
        if (!selectedHwnd && !windowTitle.empty()) {
            auto windows = enumerateWindows();
            for (const auto& w : windows) {
                if (w.second.find(windowTitle) != std::string::npos) {
                    selectedHwnd = w.first;
                    break;
                }
            }
            if (!selectedHwnd) {
                std::cerr << "ERROR: Window not found for title: " << windowTitle << std::endl;
                return 1;
            }
        }

        ROI newRoi;
        newRoi.name = createRoiName;
        std::string err;
        std::cout << "ROI select on window: " << getWindowTitle(selectedHwnd) << std::endl;
        std::cout << "  Drag a rectangle. ESC or right-click to cancel." << std::endl;
        if (!selectROIInteractive(selectedHwnd, newRoi, err)) {
            std::cerr << "ERROR: ROI selection failed: " << err << std::endl;
            return 1;
        }

        std::string writeErr;
        if (!upsertROIConfig(configPath, newRoi, writeErr)) {
            std::cerr << "ERROR: Failed to write ROI into config: " << writeErr << std::endl;
            return 1;
        }

        std::cout << "Saved ROI '" << newRoi.name << "' to config: " << configPath << std::endl;
        std::cout << "  x=" << newRoi.x << " y=" << newRoi.y << " w=" << newRoi.w << " h=" << newRoi.h << std::endl;
        std::cout << "Tip: run OCR with: --roi " << newRoi.name << std::endl;
        return 0;
    }

    if (selectRoi) {
        if (cfgForRoiList.rois.empty()) {
            std::cerr << "ERROR: No ROIs loaded from config: " << configPath << std::endl;
            return 1;
        }
        printRoisList(cfgForRoiList);
        int idx = selectRoiIndex;
        if (idx < 0) {
            std::cout << "Select ROI index: ";
            if (!(std::cin >> idx)) {
                std::cerr << "ERROR: Failed to read index." << std::endl;
                return 1;
            }
        }
        if (idx < 0 || idx >= static_cast<int>(cfgForRoiList.rois.size())) {
            std::cerr << "ERROR: ROI index out of range." << std::endl;
            return 1;
        }
        // Convert into the existing selection mechanism.
        onlyRoiName = cfgForRoiList.rois[static_cast<size_t>(idx)].name;
        allRois = false;
    }
    
    // If user didn't specify a capture target, try to auto-pick the preferred window
    // before printing the startup summary.
    autoSelectDefaultWindowIfNeeded(monitorSpecified, windowSpecified, selectedHwnd, windowTitle);

    if (!benchmarkMode) {
        std::cout << "Config: " << configPath << "\n";
        if (selectedHwnd) {
            std::cout << "Capture: selected window (HWND)\n";
        } else if (!windowTitle.empty()) {
            std::cout << "Capture: window contains \"" << windowTitle << "\"\n";
        } else {
            std::cout << "Capture: monitor index " << monitorIndex << "\n";
        }
        std::cout << "Engine: " << enginePath << "\n";
        if (!onnxPath.empty()) {
            std::cout << "ONNX:   " << onnxPath << "\n";
        }
        std::cout << "Dict:   " << dictPath << "\n";
        std::cout << "Change: " << (enableChangeDetect ? "on" : "off");
        if (enableChangeDetect) {
            std::cout << " (threshold=" << changeThreshold << ")";
        }
        std::cout << "\n";
        std::cout << "Every:  " << everyN << " frame(s)\n";
        if (scanTable) {
            std::cout << "Mode:   scan table rows";
            if (autoRows) {
                std::cout << " (auto-detect)";
            }
            std::cout << "\n";
            if (!scanTableRoiName.empty()) {
                std::cout << "Table:  " << scanTableRoiName << "\n";
            }
            if (!autoRows) {
                std::cout << "Rows:   " << tableRows
                          << " (height=" << rowHeightPx
                          << ", stride=" << rowStridePx
                          << ", yOff=" << rowOffsetYPx << ")\n";
            } else {
                std::cout << "Rows:   auto-detected from pixel analysis\n";
            }
        }
        if (!onlyRoiName.empty()) {
            std::cout << "ROI:    " << onlyRoiName << "\n";
        } else if (!allRois) {
            std::cout << "ROI:    first ROI only (use --all-rois)\n";
        }
        std::cout << std::endl;
    }
    
    if (benchmarkMode) {
        std::cout << "\nBenchmark mode requested. Running CUDA kernel timing test...\n";
        
        // Simple CUDA benchmark
        trading_monitor::CudaTimer timer;
        
        // Allocate test buffer
        float* d_buffer = nullptr;
        size_t bufferSize = 1920 * 1080 * sizeof(float);
        cudaMalloc(&d_buffer, bufferSize);
        
        // Warm up
        cudaMemset(d_buffer, 0, bufferSize);
        cudaDeviceSynchronize();
        
        // Benchmark memset
        timer.start();
        cudaMemset(d_buffer, 0, bufferSize);
        timer.stop();
        
        std::cout << "CUDA memset (1920x1080 float): " << timer.elapsedMs() << " ms\n";
        
        cudaFree(d_buffer);
        return 0;
    }

    // Load ROIs
    PipelineConfig cfg = cfgForRoiList;
    if (cfg.rois.empty()) {
        std::cerr << "ERROR: No ROIs loaded from config: " << configPath << std::endl;
        return 1;
    }

    std::vector<float> anchorScales = parseScalesCsv(anchorScalesStr);
    if (anchorScales.empty()) {
        anchorScales = {0.6f, 0.7f, 0.75f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
    }
    float anchorDx = 0.0f, anchorDy = 0.0f, anchorDw = 0.0f, anchorDh = 0.0f;
    const bool anchorOffsetValid = parseOffsetRect(anchorOffsetStr, anchorDx, anchorDy, anchorDw, anchorDh);
    float anchorSearchDx = 0.0f, anchorSearchDy = 0.0f, anchorSearchDw = 0.0f, anchorSearchDh = 0.0f;
    const bool anchorSearchValid = parseOffsetRect(anchorSearchStr, anchorSearchDx, anchorSearchDy, anchorSearchDw, anchorSearchDh);
    float anchorSecondaryDx = 0.0f, anchorSecondaryDy = 0.0f, anchorSecondaryDw = 0.0f, anchorSecondaryDh = 0.0f;
    const bool anchorSecondaryOffsetValid = parseOffsetRect(anchorSecondaryOffsetStr, anchorSecondaryDx, anchorSecondaryDy, anchorSecondaryDw, anchorSecondaryDh);
    std::vector<uint8_t> anchorTplGray;
    int anchorTplW = 0, anchorTplH = 0;
    if (!anchorTemplatePath.empty()) {
        if (!loadTemplateGray(anchorTemplatePath, anchorTplGray, anchorTplW, anchorTplH)) {
            std::cerr << "WARNING: Failed to load anchor template: " << anchorTemplatePath << std::endl;
            anchorTemplatePath.clear();
        }
    }
    std::vector<uint8_t> anchorSecondaryTplGray;
    int anchorSecondaryTplW = 0, anchorSecondaryTplH = 0;
    if (!anchorSecondaryTemplatePath.empty()) {
        if (!loadTemplateGray(anchorSecondaryTemplatePath, anchorSecondaryTplGray, anchorSecondaryTplW, anchorSecondaryTplH)) {
            std::cerr << "WARNING: Failed to load secondary anchor template: " << anchorSecondaryTemplatePath << std::endl;
            anchorSecondaryTemplatePath.clear();
        }
    }

    ROI tableROI{};
    size_t tableRoiIndex = static_cast<size_t>(-1);
    if (scanTable) {
        if (scanTableRoiName.empty()) {
            std::cout << "Enter table ROI name to scan: ";
            if (!(std::cin >> scanTableRoiName)) {
                std::cerr << "ERROR: Failed to read table ROI name." << std::endl;
                return 1;
            }
        }
        for (size_t i = 0; i < cfg.rois.size(); i++) {
            if (cfg.rois[i].name == scanTableRoiName) {
                tableROI = cfg.rois[i];
                tableRoiIndex = i;
                break;
            }
        }
        if (tableRoiIndex == static_cast<size_t>(-1)) {
            std::cerr << "ERROR: Table ROI not found in config: " << scanTableRoiName << std::endl;
            printRoisList(cfg);
            std::cerr << "Tip: Create it with: --create-roi " << scanTableRoiName << std::endl;
            return 1;
        }
    }

    // Parse column definitions
    // Format: "name:x,w;name:x,w" e.g. "symbol:0,50;pnl:145,40"
    if (!columnsDefStr.empty()) {
        std::istringstream iss(columnsDefStr);
        std::string colSpec;
        while (std::getline(iss, colSpec, ';')) {
            size_t colonPos = colSpec.find(':');
            size_t commaPos = colSpec.find(',');
            if (colonPos != std::string::npos && commaPos != std::string::npos && commaPos > colonPos) {
                ColumnDef def;
                def.name = colSpec.substr(0, colonPos);
                def.xOffset = std::stoi(colSpec.substr(colonPos + 1, commaPos - colonPos - 1));
                def.width = std::stoi(colSpec.substr(commaPos + 1));
                if (def.width > 0) {
                    columnDefs.push_back(def);
                    std::cout << "Column: " << def.name << " (x=" << def.xOffset << ", w=" << def.width << ")\n";
                }
            }
        }
    } else if (colWidth > 0) {
        // Single column mode via --col-x and --col-w
        ColumnDef def;
        def.name = "col";
        def.xOffset = colOffsetX;
        def.width = colWidth;
        columnDefs.push_back(def);
        std::cout << "Column: x=" << colOffsetX << ", w=" << colWidth << "\n";
    }

    std::vector<size_t> roiIndices;
    if (!onlyRoiName.empty()) {
        for (size_t i = 0; i < cfg.rois.size(); i++) {
            if (cfg.rois[i].name == onlyRoiName) {
                roiIndices.push_back(i);
                break;
            }
        }
        if (roiIndices.empty()) {
            std::cerr << "ERROR: ROI not found in config: " << onlyRoiName << std::endl;
            return 1;
        }
    } else if (allRois) {
        roiIndices.resize(cfg.rois.size());
        for (size_t i = 0; i < cfg.rois.size(); i++) roiIndices[i] = i;
    } else {
        roiIndices.push_back(0);
    }

    // Load SVTR model + decoder
    SVTRInference svtr;
    if (!svtr.loadModel(enginePath)) {
        if (onnxPath.empty()) {
            std::cerr << "ERROR: Failed to load engine and no --onnx provided to build.\n";
            return 1;
        }
        std::cout << "Engine load failed; building from ONNX...\n";
        if (!svtr.buildModel(onnxPath, enginePath, true)) {
            std::cerr << "ERROR: Failed to build/load model from ONNX." << std::endl;
            return 1;
        }
    }

    int timesteps = 0, numClasses = 0;
    svtr.getOutputDims(timesteps, numClasses);

    CTCDecoder decoder;
    if (!decoder.loadDictionary(dictPath)) {
        std::cerr << "WARNING: Failed to load dictionary: " << dictPath << std::endl;
    } else {
        decoder.ensurePaddleOCRDictionarySize(numClasses);
        if (verbose) {
            std::cout << "Dictionary size: " << decoder.getDictionarySize() << "\n";
        }
    }

    // ==========================================================================
    // YOLO WINDOW DETECTOR INITIALIZATION (optional)
    // ==========================================================================
    std::unique_ptr<YOLODetector> yoloDetector;
    if (useYoloDetection) {
        yoloDetector = std::make_unique<YOLODetector>();
        yoloDetector->setConfidenceThreshold(detectionConfidence);

        // Parse class names
        std::vector<std::string> classNames;
        if (!detectClassNamesStr.empty()) {
            std::stringstream ss(detectClassNamesStr);
            std::string name;
            while (std::getline(ss, name, ',')) {
                classNames.push_back(name);
            }
        } else {
            classNames = {"window"}; // Default
        }

        // Try to load engine, fallback to build from ONNX
        std::string yoloEnginePath = detectModelPath.empty()
            ? "../models/yolov10n_window.engine"
            : detectModelPath;

        if (!yoloDetector->loadModel(yoloEnginePath, classNames)) {
            if (!detectOnnxPath.empty()) {
                std::cout << "YOLO engine load failed; building from ONNX...\n";
                if (!yoloDetector->buildModel(detectOnnxPath, yoloEnginePath, true)) {
                    std::cerr << "WARNING: Failed to build YOLO detector: " << yoloDetector->getLastError() << "\n";
                    std::cerr << "Falling back to template matching.\n";
                    yoloDetector.reset();
                    useYoloDetection = false;
                } else {
                    std::cout << "YOLO detector built and loaded successfully.\n";
                }
            } else {
                std::cerr << "WARNING: Failed to load YOLO detector: " << yoloDetector->getLastError() << "\n";
                std::cerr << "Falling back to template matching.\n";
                yoloDetector.reset();
                useYoloDetection = false;
            }
        }

        if (useYoloDetection && verbose) {
            std::cout << "YOLO window detector loaded (confidence threshold: " << detectionConfidence << ")\n";
        }
    }

    // ==========================================================================
    // STATIC IMAGE MODE: Process a single image file instead of live capture
    // ==========================================================================
    if (!imagePath.empty()) {
        std::cout << "\n=== STATIC IMAGE MODE ===\n";
        std::cout << "Loading image: " << imagePath << "\n";

        int imgWidth = 0, imgHeight = 0, imgChannels = 0;
        stbi_uc* imgData = stbi_load(imagePath.c_str(), &imgWidth, &imgHeight, &imgChannels, 4);
        if (!imgData) {
            std::cerr << "ERROR: Failed to load image: " << imagePath << std::endl;
            std::cerr << "Reason: " << (stbi_failure_reason() ? stbi_failure_reason() : "unknown") << std::endl;
            return 1;
        }

        std::cout << "Image loaded: " << imgWidth << "x" << imgHeight << " (" << imgChannels << " channels)\n";

        // Parse --test-roi if provided (format: x,y,w,h)
        ROI customTestROI{};
        bool useCustomTestROI = false;
        if (!testRoiStr.empty()) {
            int vals[4] = {0};
            int count = sscanf(testRoiStr.c_str(), "%d,%d,%d,%d", &vals[0], &vals[1], &vals[2], &vals[3]);
            if (count == 4) {
                customTestROI.name = "test_roi";
                customTestROI.x = vals[0];
                customTestROI.y = vals[1];
                customTestROI.w = vals[2];
                customTestROI.h = vals[3];
                useCustomTestROI = true;
                std::cout << "Using custom test ROI: (" << vals[0] << "," << vals[1]
                          << ") " << vals[2] << "x" << vals[3] << "\n";

                // Override scanTable and tableROI if using test ROI for table scan
                if (scanTable) {
                    tableROI = customTestROI;
                    tableROI.name = "custom_table";
                }
            } else {
                std::cerr << "WARNING: Invalid --test-roi format. Expected: x,y,w,h\n";
            }
        }

        // YOLO detection for static image mode
        if (useYoloDetection && yoloDetector) {
            auto detections = yoloDetector->detectCPU(imgData, imgWidth, imgHeight);
            if (!detections.empty()) {
                const auto& best = detections[0];
                ROI yoloROI = best.toROI(best.className);

                if (verbose) {
                    std::cout << "[YOLO] Detected " << best.className
                              << " (class " << best.classId << ")"
                              << " at (" << yoloROI.x << "," << yoloROI.y << ") "
                              << yoloROI.w << "x" << yoloROI.h
                              << " conf=" << best.confidence << "\n";
                }

                // Apply offset if configured
                if (anchorOffsetValid && scanTable) {
                    tableROI.x = yoloROI.x + static_cast<int>(anchorDx);
                    tableROI.y = yoloROI.y + static_cast<int>(anchorDy);
                    tableROI.w = anchorDw > 0 ? static_cast<int>(anchorDw) : yoloROI.w;
                    tableROI.h = anchorDh > 0 ? static_cast<int>(anchorDh) : yoloROI.h;
                    if (verbose) {
                        std::cout << "[YOLO] Table ROI after offset: (" << tableROI.x << "," << tableROI.y
                                  << ") " << tableROI.w << "x" << tableROI.h << "\n";
                    }
                }
            } else if (verbose) {
                std::cout << "[YOLO] No detections above confidence threshold.\n";
            }
        }

        float anchorScaleForColumns = 1.0f;
        if (!useYoloDetection && scanTable && !anchorTemplatePath.empty() && !anchorTplGray.empty()) {
            std::vector<uint8_t> imgGray;
            bgraToGray(imgData, imgWidth, imgHeight, imgGray);

            int searchX = 0, searchY = 0, searchW = imgWidth, searchH = imgHeight;
            if (anchorSearchValid) {
                searchX = static_cast<int>(std::lround(anchorSearchDx));
                searchY = static_cast<int>(std::lround(anchorSearchDy));
                searchW = static_cast<int>(std::lround(anchorSearchDw));
                searchH = static_cast<int>(std::lround(anchorSearchDh));
                searchW = (std::max)(1, searchW);
                searchH = (std::max)(1, searchH);
            }

            std::vector<uint8_t> searchGray = (anchorSearchValid)
                ? cropGrayRegion(imgGray, imgWidth, imgHeight, searchX, searchY, searchW, searchH)
                : imgGray;
            const int searchImgW = anchorSearchValid ? (std::max)(1, (std::min)(searchW, imgWidth - (std::max)(0, searchX))) : imgWidth;
            const int searchImgH = anchorSearchValid ? (std::max)(1, (std::min)(searchH, imgHeight - (std::max)(0, searchY))) : imgHeight;

            AnchorMatchResult anchor = findAnchorByTemplate(
                searchGray, searchImgW, searchImgH,
                anchorTplGray, anchorTplW, anchorTplH,
                anchorScales, anchorThreshold, anchorMaxSearchW
            );

            if (anchor.found && anchorSearchValid) {
                anchor.x += searchX;
                anchor.y += searchY;
            }

            if (anchor.found) {
                if (!anchorSecondaryTemplatePath.empty() && !anchorSecondaryTplGray.empty()) {
                    const bool ok = validateSecondaryAnchor(
                        imgGray, imgWidth, imgHeight,
                        anchor, anchorSecondaryTplGray, anchorSecondaryTplW, anchorSecondaryTplH,
                        anchorSecondaryOffsetValid, anchorSecondaryDx, anchorSecondaryDy, anchorSecondaryDw, anchorSecondaryDh,
                        anchorSecondaryThreshold
                    );
                    if (!ok) {
                        anchor.found = false;
                    }
                }
            }

            if (anchor.found) {
                ROI anchored = tableROI;
                anchorScaleForColumns = anchor.scale;
                if (anchorOffsetValid) {
                    anchored.x = static_cast<int>(std::lround(anchor.x + anchorDx * anchor.scale));
                    anchored.y = static_cast<int>(std::lround(anchor.y + anchorDy * anchor.scale));
                    anchored.w = anchorDw > 0.0f ? static_cast<int>(std::lround(anchorDw * anchor.scale)) : anchor.w;
                    anchored.h = anchorDh > 0.0f ? static_cast<int>(std::lround(anchorDh * anchor.scale)) : anchor.h;
                } else {
                    anchored.x = anchor.x;
                    anchored.y = anchor.y;
                    anchored.w = anchor.w;
                    anchored.h = anchor.h;
                }
                anchored = clampROIToFrame(anchored, imgWidth, imgHeight);
                if (verbose) {
                    std::cout << "Anchor match: score=" << anchor.score << " scale=" << anchor.scale
                              << " at (" << anchor.x << "," << anchor.y << ") "
                              << anchor.w << "x" << anchor.h << "\n";
                    std::cout << "Anchored table ROI: (" << anchored.x << "," << anchored.y << ") "
                              << anchored.w << "x" << anchored.h << "\n";
                }
                tableROI = anchored;
            } else {
                std::cerr << "WARNING: Anchor template match failed (score < " << anchorThreshold << ")\n";
            }
        }

        // Initialize ROI extractor
        ROIExtractor roiExtractor;
        const size_t numColumns = columnDefs.empty() ? 1 : columnDefs.size();
        const size_t maxRows = scanTable ? static_cast<size_t>(tableRows) * numColumns : cfg.rois.size();
        if (!roiExtractor.initialize(maxRows, cfg.upscaleFactor)) {
            std::cerr << "ERROR: ROIExtractor initialization failed." << std::endl;
            stbi_image_free(imgData);
            return 1;
        }
        roiExtractor.setPreUpscaleFactor(ocrZoom);

        // Initialize row detector if auto-rows enabled
        RowDetector rowDetector;
        bool templateActive = false;
        if (autoRows && scanTable) {
            if (!rowDetector.initialize(tableROI.w, tableROI.h)) {
                std::cerr << "ERROR: RowDetector initialization failed." << std::endl;
                stbi_image_free(imgData);
                return 1;
            }
            templateActive = configureRowDetector(rowDetector);
        }

        // Create CUDA stream
        cudaStream_t stream = nullptr;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

        // Allocate GPU buffer for the full image using cudaMallocPitch for proper alignment
        uint8_t* d_imageData = nullptr;
        size_t d_imagePitch = 0;
        const size_t srcStride = static_cast<size_t>(imgWidth) * 4;
        cudaError_t allocErr = cudaMallocPitch(&d_imageData, &d_imagePitch,
                                                static_cast<size_t>(imgWidth) * 4,
                                                static_cast<size_t>(imgHeight));
        if (allocErr != cudaSuccess) {
            std::cerr << "ERROR: Failed to allocate GPU memory: " << cudaGetErrorString(allocErr) << std::endl;
            stbi_image_free(imgData);
            return 1;
        }

        // Copy image data with proper pitch
        cudaMemcpy2D(d_imageData, d_imagePitch, imgData, srcStride,
                     static_cast<size_t>(imgWidth) * 4, static_cast<size_t>(imgHeight),
                     cudaMemcpyHostToDevice);

        // Create CUDA texture from the image
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr = d_imageData;
        resDesc.res.pitch2D.width = static_cast<size_t>(imgWidth);
        resDesc.res.pitch2D.height = static_cast<size_t>(imgHeight);
        resDesc.res.pitch2D.pitchInBytes = d_imagePitch;
        resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();

        cudaTextureDesc texDesc{};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;  // Bilinear interpolation for upscaling
        texDesc.readMode = cudaReadModeNormalizedFloat;  // Auto-convert uint8 to [0,1] float
        texDesc.normalizedCoords = 0;

        cudaTextureObject_t texObj = 0;
        cudaError_t texErr = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
        if (texErr != cudaSuccess) {
            std::cerr << "ERROR: Failed to create texture object: " << cudaGetErrorString(texErr) << std::endl;
            cudaFree(d_imageData);
            stbi_image_free(imgData);
            return 1;
        }

        std::cout << "CUDA texture created successfully (pitch=" << d_imagePitch << ").\n\n";

        std::vector<float> h_output(static_cast<size_t>(timesteps) * static_cast<size_t>(numClasses));

        // Build list of row ROIs
        std::vector<ROI> rowRoisToProcess;

        if (scanTable) {
            std::cout << "=== TABLE SCAN MODE ===\n";
            std::cout << "Table ROI: " << tableROI.name << " (x=" << tableROI.x << ", y=" << tableROI.y
                      << ", w=" << tableROI.w << ", h=" << tableROI.h << ")\n\n";

            if (autoRows) {
                std::cout << "Auto-detecting rows...\n";
                std::vector<DetectedRow> detectedRows;

                if (templateActive) {
                    detectedRows = rowDetector.detectRowsGPU(texObj, tableROI, stream);
                } else {
                    // Extract table region for row detection (CPU intensity-based)
                    const size_t tableStride = static_cast<size_t>(tableROI.w) * 4;
                    const size_t tableSize = tableStride * static_cast<size_t>(tableROI.h);
                    std::vector<uint8_t> tableData(tableSize);

                    // Copy table region from image
                    for (int y = 0; y < tableROI.h && (tableROI.y + y) < imgHeight; y++) {
                        const int srcY = tableROI.y + y;
                        const int srcX = tableROI.x;
                        if (srcX >= 0 && srcX < imgWidth) {
                            const size_t srcOffset = static_cast<size_t>(srcY) * srcStride + static_cast<size_t>(srcX) * 4;
                            const size_t dstOffset = static_cast<size_t>(y) * tableStride;
                            const size_t copyWidth = static_cast<size_t>((std::min)(tableROI.w, imgWidth - srcX)) * 4;
                            memcpy(tableData.data() + dstOffset, imgData + srcOffset, copyWidth);
                        }
                    }

                    // Detect rows
                    detectedRows = rowDetector.detectRows(
                        tableData.data(), tableROI.w, tableROI.h, static_cast<int>(tableStride)
                    );
                }
                std::cout << "Detected " << detectedRows.size() << " rows\n";

                for (size_t i = 0; i < detectedRows.size(); i++) {
                    const auto& dr = detectedRows[i];
                    std::cout << "  Row " << i << ": y=" << dr.yStart << "-" << dr.yEnd
                              << " (h=" << dr.height << ")";
                    if (dr.hasGreenPnL) std::cout << " [GREEN]";
                    if (dr.hasRedPnL) std::cout << " [RED]";
                    std::cout << " conf=" << dr.confidence << "\n";
                }
                std::cout << "\n";

                // Convert to ROIs
                rowRoisToProcess = rowDetector.rowsToROIs(detectedRows, tableROI, scanTableRoiName + "_auto");
            } else {
                // Fixed row parameters
                for (int r = 0; r < tableRows; r++) {
                    ROI rowRoi;
                    rowRoi.name = scanTableRoiName + "_row" + std::to_string(r);
                    rowRoi.x = tableROI.x;
                    rowRoi.y = tableROI.y + rowOffsetYPx + r * rowStridePx;
                    rowRoi.w = tableROI.w;
                    rowRoi.h = rowHeightPx;
                    rowRoisToProcess.push_back(rowRoi);
                }
            }

            // Apply column extraction if columns are defined
            if (!columnDefs.empty()) {
                std::vector<ROI> columnRois;
                for (const auto& rowRoi : rowRoisToProcess) {
                    for (const auto& col : columnDefs) {
                        ROI colRoi;
                        colRoi.name = rowRoi.name + "_" + col.name;
                        const int scaledOffsetX = static_cast<int>(std::lround(col.xOffset * anchorScaleForColumns));
                        const int scaledWidth = static_cast<int>(std::lround(col.width * anchorScaleForColumns));
                        colRoi.x = rowRoi.x + scaledOffsetX;
                        colRoi.y = rowRoi.y;
                        colRoi.w = (std::max)(1, scaledWidth);
                        colRoi.h = rowRoi.h;
                        columnRois.push_back(colRoi);
                    }
                }
                rowRoisToProcess = std::move(columnRois);
                std::cout << "Expanded to " << rowRoisToProcess.size() << " column ROIs\n\n";
            }
        } else {
            // Use configured ROIs
            for (size_t idx : roiIndices) {
                if (idx < cfg.rois.size()) {
                    rowRoisToProcess.push_back(cfg.rois[idx]);
                }
            }
        }

        std::cout << "=== OCR RESULTS ===\n";

        // JSON result collection
        std::vector<OCRResultEntry> jsonResults;
        auto startProcessing = std::chrono::high_resolution_clock::now();

        // Process each ROI
        for (size_t r = 0; r < rowRoisToProcess.size(); r++) {
            auto roiStart = std::chrono::high_resolution_clock::now();
            const ROI& roi = rowRoisToProcess[r];

            // Clamp ROI to image bounds
            ROI clampedRoi = roi;
            clampedRoi.x = (std::max)(0, (std::min)(roi.x, imgWidth - 1));
            clampedRoi.y = (std::max)(0, (std::min)(roi.y, imgHeight - 1));
            clampedRoi.w = (std::max)(1, (std::min)(roi.w, imgWidth - clampedRoi.x));
            clampedRoi.h = (std::max)(1, (std::min)(roi.h, imgHeight - clampedRoi.y));

            if (verbose) {
                std::cout << "  Extracting ROI: " << roi.name << " @ (" << clampedRoi.x << "," << clampedRoi.y
                          << ") " << clampedRoi.w << "x" << clampedRoi.h << "\n";
            }

            float* d_input = roiExtractor.extractROI(texObj, clampedRoi, r, stream);
            if (!d_input) {
                std::cerr << "  [" << roi.name << "] Failed to extract ROI\n";
                continue;
            }

            // Debug: dump first row's extracted ROI to check what's being sent to OCR
            if (verbose && r == 0) {
                cudaStreamSynchronize(stream);
                std::vector<float> h_debug(320 * 48 * 3);
                cudaMemcpy(h_debug.data(), d_input, h_debug.size() * sizeof(float), cudaMemcpyDeviceToHost);
                // Check if data has any non-zero values
                float minVal = 1e9f, maxVal = -1e9f, sum = 0.0f;
                for (float v : h_debug) { minVal = (std::min)(minVal, v); maxVal = (std::max)(maxVal, v); sum += v; }
                std::cout << "  [DEBUG] ROI buffer: min=" << minVal << " max=" << maxVal
                          << " avg=" << (sum / h_debug.size()) << "\n";
            }

            if (!svtr.infer(d_input, 320, 48, stream)) {
                std::cerr << "  [" << roi.name << "] Inference failed\n";
                continue;
            }
            cudaStreamSynchronize(stream);

            float* d_probs = svtr.getOutputProbs();
            cudaMemcpy(h_output.data(), d_probs, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

            CTCResult res = decoder.decode(h_output.data(), timesteps, numClasses, 0);
            const std::string ticker = extractTickerLike(res.text);
            const std::string numeric = filterNumericLike(res.text);

            auto roiEnd = std::chrono::high_resolution_clock::now();
            double roiLatencyMs = std::chrono::duration<double, std::milli>(roiEnd - roiStart).count();

            // Build display text
            std::string displayText;
            if (!ticker.empty()) displayText = ticker + " ";
            displayText += numeric;

            std::cout << "  [" << roi.name << "] " << displayText;
            if (verbose) {
                std::cout << " (raw='" << res.text << "', conf=" << res.confidence << ")";
            }
            std::cout << "\n";

            // Collect JSON result
            if (!jsonOutputPath.empty()) {
                OCRResultEntry entry;
                entry.roiName = roi.name;
                entry.x = clampedRoi.x;
                entry.y = clampedRoi.y;
                entry.w = clampedRoi.w;
                entry.h = clampedRoi.h;
                entry.text = displayText;
                entry.rawText = res.text;
                entry.confidence = res.confidence;
                entry.latencyMs = roiLatencyMs;
                jsonResults.push_back(entry);
            }
        }

        auto endProcessing = std::chrono::high_resolution_clock::now();
        double totalLatencyMs = std::chrono::duration<double, std::milli>(endProcessing - startProcessing).count();

        // Write JSON output if requested
        if (!jsonOutputPath.empty()) {
            if (writeJsonResults(jsonOutputPath, jsonResults, imagePath, totalLatencyMs)) {
                std::cout << "\nJSON results written to: " << jsonOutputPath << "\n";
            } else {
                std::cerr << "ERROR: Failed to write JSON output to: " << jsonOutputPath << "\n";
            }
        }

        std::cout << "\n=== PROCESSING COMPLETE (" << std::fixed << std::setprecision(1) << totalLatencyMs << "ms) ===\n";

        // Cleanup
        cudaDestroyTextureObject(texObj);
        cudaFree(d_imageData);
        cudaStreamDestroy(stream);
        stbi_image_free(imgData);

        return 0;
    }
    // ==========================================================================
    // END STATIC IMAGE MODE
    // ==========================================================================

    // Capture + interop
    D3D11Capture capture;
    if (!capture.initializeDevice()) {
        std::cerr << "ERROR: Failed to initialize D3D11 device: " << capture.getLastError() << std::endl;
        return 1;
    }

    // If user didn't specify a capture target, try to auto-pick the preferred window.
    autoSelectDefaultWindowIfNeeded(monitorSpecified, windowSpecified, selectedHwnd, windowTitle);

    if (selectedHwnd) {
        if (!capture.initializeWindow(selectedHwnd)) {
            std::cerr << "ERROR: Failed to initialize selected window capture: " << capture.getLastError() << std::endl;
            return 1;
        }
    } else if (!windowTitle.empty()) {
        if (!capture.initializeWindow(windowTitle)) {
            std::cerr << "ERROR: Failed to initialize window capture: " << capture.getLastError() << std::endl;
            return 1;
        }
    } else {
        if (!capture.initializeMonitor(monitorIndex)) {
            std::cerr << "ERROR: Failed to initialize monitor capture: " << capture.getLastError() << std::endl;
            return 1;
        }
    }

    CudaD3D11Interop interop;
    if (!interop.initialize(capture.getDevice())) {
        std::cerr << "ERROR: CUDA interop init failed: " << interop.getLastError() << std::endl;
        return 1;
    }
    CudaTexturePool texPool(interop);

    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    ROIExtractor roiExtractor;
    if (scanTable) {
        if (!roiExtractor.initialize(static_cast<size_t>(tableRows), cfg.upscaleFactor)) {
            std::cerr << "ERROR: ROIExtractor initialization failed." << std::endl;
            return 1;
        }
    } else {
        if (!roiExtractor.initialize(cfg.rois, cfg.upscaleFactor)) {
            std::cerr << "ERROR: ROIExtractor initialization failed." << std::endl;
            return 1;
        }
    }
    roiExtractor.setPreUpscaleFactor(ocrZoom);

    ChangeDetector changeDetector;
    if (enableChangeDetect) {
        const size_t numCols = columnDefs.empty() ? 1 : columnDefs.size();
        const size_t changeCount = scanTable ? static_cast<size_t>(tableRows) * numCols : cfg.rois.size();
        changeDetector.initialize(changeCount, changeThreshold);
        changeDetector.invalidateAll();
    }

    // Row detector for automatic row boundary detection
    RowDetector rowDetector;
    bool templateActive = false;
    if (autoRows && scanTable) {
        if (!rowDetector.initialize(tableROI.w, tableROI.h)) {
            std::cerr << "ERROR: RowDetector initialization failed." << std::endl;
            return 1;
        }
        templateActive = configureRowDetector(rowDetector);
        std::cout << "Auto-row detection enabled for table ROI: " << scanTableRoiName << std::endl;
    }

    std::vector<DetectedRow> detectedRows;

    std::vector<float> h_output(static_cast<size_t>(timesteps) * static_cast<size_t>(numClasses));
    std::vector<float> h_roi_plane;
    std::vector<uint8_t> h_roi_img;
    std::mutex printMutex;

    std::vector<bool> dumped;
    if (dumpRoi) {
        dumped.assign(roiExtractor.getROICount(), false);
        if (dumpRoiDir.empty()) {
            dumpRoiDir = "roi_dumps";
        }
        std::error_code ec;
        fs::create_directories(dumpRoiDir, ec);
    }

    std::vector<std::string> lastRowKey;
    if (scanTable) {
        lastRowKey.assign(static_cast<size_t>(tableRows), std::string());
    }

    ROIOverlayHandle overlayHandle;
    if (showRoiOverlay) {
        // We can only overlay a window target.
        void* overlayTarget = selectedHwnd;
        if (!overlayTarget && !windowTitle.empty()) {
            auto windows = enumerateWindows();
            for (const auto& w : windows) {
                if (w.second.find(windowTitle) != std::string::npos) {
                    overlayTarget = w.first;
                    break;
                }
            }
        }

        if (!overlayTarget) {
            std::cerr << "WARNING: --show-roi works only with a window target (use --select-window / auto window)." << std::endl;
        } else {
            std::vector<ROI> overlayRois;
            if (scanTable) {
                // Show the configured table ROI and the computed row slices (client coords).
                overlayRois.push_back(tableROI);
                for (int r = 0; r < tableRows; r++) {
                    ROI row;
                    row.name = scanTableRoiName + "_row" + std::to_string(r);
                    row.x = tableROI.x;
                    row.y = tableROI.y + rowOffsetYPx + r * rowStridePx;
                    row.w = tableROI.w;
                    row.h = rowHeightPx;
                    overlayRois.push_back(row);
                }
            } else {
                for (size_t idx : roiIndices) {
                    if (idx < cfg.rois.size()) overlayRois.push_back(cfg.rois[idx]);
                }
            }

            std::string overlayErr;
            if (startROIOverlay(overlayTarget, overlayRois, overlayHandle, overlayErr)) {
                std::cout << "ROI overlay enabled (click-through)." << std::endl;
            } else {
                std::cerr << "WARNING: Failed to start ROI overlay: " << overlayErr << std::endl;
            }
        }
    }

    std::cout << "Starting capture... Press Ctrl+C to stop." << std::endl;

    AnchorMatchResult liveAnchor;
    bool anchorLocked = false;
    uint64_t lastAnchorFrame = 0;
    ROI anchoredTableROI = tableROI;

    capture.startCapture([&](const CaptureFrame& frame) {
        if (!g_running.load()) return;
        if (everyN > 1 && (frame.frameNumber % static_cast<uint64_t>(everyN) != 0)) return;

        ID3D11Texture2D* tex = frame.texture;
        if (!tex) return;

        cudaGraphicsResource_t resource = texPool.getOrRegister(tex);
        if (!resource) return;

        ScopedCudaMap mapped(interop, resource, stream);
        if (!mapped.isMapped()) return;

        cudaArray_t array = mapped.getArray();
        if (!array) return;

        cudaTextureObject_t texObj = interop.createTextureObject(array);
        if (!texObj) return;

        HWND hwndForMap = selectedHwnd ? reinterpret_cast<HWND>(selectedHwnd) : nullptr;

        // YOLO Window Detection (runs every detectEveryN frames)
        static uint64_t lastYoloFrame = 0;
        static ROI yoloDetectedROI{};
        static bool yoloDetectionValid = false;

        if (useYoloDetection && yoloDetector) {
            const bool shouldRunYolo = (frame.frameNumber - lastYoloFrame >= static_cast<uint64_t>(detectEveryN));
            if (shouldRunYolo) {
                auto detections = yoloDetector->detect(texObj, frame.width, frame.height, stream);
                if (!detections.empty()) {
                    // Use highest confidence detection
                    const auto& best = detections[0];
                    yoloDetectedROI = best.toROI(best.className);
                    yoloDetectedROI = clampROIToFrame(yoloDetectedROI, frame.width, frame.height);
                    yoloDetectionValid = true;
                    lastYoloFrame = frame.frameNumber;

                    if (verbose) {
                        std::lock_guard<std::mutex> lock(printMutex);
                        std::cout << "[YOLO] Detected " << best.className
                                  << " at (" << yoloDetectedROI.x << "," << yoloDetectedROI.y
                                  << ") " << yoloDetectedROI.w << "x" << yoloDetectedROI.h
                                  << " conf=" << best.confidence << std::endl;
                    }
                }
            }
        }

        ROI activeTableROI = tableROI;

        // Use YOLO detection if available, otherwise fall back to template matching
        if (useYoloDetection && yoloDetectionValid) {
            // Apply configured offset from detected window to get table ROI
            if (anchorOffsetValid) {
                activeTableROI.x = yoloDetectedROI.x + static_cast<int>(anchorDx);
                activeTableROI.y = yoloDetectedROI.y + static_cast<int>(anchorDy);
                activeTableROI.w = anchorDw > 0 ? static_cast<int>(anchorDw) : yoloDetectedROI.w;
                activeTableROI.h = anchorDh > 0 ? static_cast<int>(anchorDh) : yoloDetectedROI.h;
            } else {
                activeTableROI = yoloDetectedROI;
            }
            activeTableROI = clampROIToFrame(activeTableROI, frame.width, frame.height);
        } else if (scanTable && !anchorTemplatePath.empty() && !anchorTplGray.empty()) {
            const bool shouldAnchor = !anchorLocked || (anchorEveryN > 0 && (frame.frameNumber - lastAnchorFrame) >= static_cast<uint64_t>(anchorEveryN));
            if (shouldAnchor) {
                std::vector<uint8_t> hostBGRA(static_cast<size_t>(frame.width) * static_cast<size_t>(frame.height) * 4);
                cudaMemcpy2DFromArray(
                    hostBGRA.data(),
                    static_cast<size_t>(frame.width) * 4,
                    array,
                    0,
                    0,
                    static_cast<size_t>(frame.width) * 4,
                    static_cast<size_t>(frame.height),
                    cudaMemcpyDeviceToHost
                );
                std::vector<uint8_t> hostGray;
                bgraToGray(hostBGRA.data(), frame.width, frame.height, hostGray);
                int searchX = 0, searchY = 0, searchW = frame.width, searchH = frame.height;
                if (anchorSearchValid) {
                    searchX = static_cast<int>(std::lround(anchorSearchDx));
                    searchY = static_cast<int>(std::lround(anchorSearchDy));
                    searchW = static_cast<int>(std::lround(anchorSearchDw));
                    searchH = static_cast<int>(std::lround(anchorSearchDh));
                    searchW = (std::max)(1, searchW);
                    searchH = (std::max)(1, searchH);
                }
                std::vector<uint8_t> searchGray = (anchorSearchValid)
                    ? cropGrayRegion(hostGray, frame.width, frame.height, searchX, searchY, searchW, searchH)
                    : hostGray;
                const int searchImgW = anchorSearchValid ? (std::max)(1, (std::min)(searchW, frame.width - (std::max)(0, searchX))) : frame.width;
                const int searchImgH = anchorSearchValid ? (std::max)(1, (std::min)(searchH, frame.height - (std::max)(0, searchY))) : frame.height;
                AnchorMatchResult anchor = findAnchorByTemplate(
                    searchGray, searchImgW, searchImgH,
                    anchorTplGray, anchorTplW, anchorTplH,
                    anchorScales, anchorThreshold, anchorMaxSearchW
                );
                if (anchor.found && anchorSearchValid) {
                    anchor.x += searchX;
                    anchor.y += searchY;
                }
                if (anchor.found && !anchorSecondaryTemplatePath.empty() && !anchorSecondaryTplGray.empty()) {
                    const bool ok = validateSecondaryAnchor(
                        hostGray, frame.width, frame.height,
                        anchor, anchorSecondaryTplGray, anchorSecondaryTplW, anchorSecondaryTplH,
                        anchorSecondaryOffsetValid, anchorSecondaryDx, anchorSecondaryDy, anchorSecondaryDw, anchorSecondaryDh,
                        anchorSecondaryThreshold
                    );
                    if (!ok) {
                        anchor.found = false;
                    }
                }
                if (anchor.found) {
                    ROI anchored = tableROI;
                    if (anchorOffsetValid) {
                        anchored.x = static_cast<int>(std::lround(anchor.x + anchorDx * anchor.scale));
                        anchored.y = static_cast<int>(std::lround(anchor.y + anchorDy * anchor.scale));
                        anchored.w = anchorDw > 0.0f ? static_cast<int>(std::lround(anchorDw * anchor.scale)) : anchor.w;
                        anchored.h = anchorDh > 0.0f ? static_cast<int>(std::lround(anchorDh * anchor.scale)) : anchor.h;
                    } else {
                        anchored.x = anchor.x;
                        anchored.y = anchor.y;
                        anchored.w = anchor.w;
                        anchored.h = anchor.h;
                    }
                    anchoredTableROI = clampROIToFrame(anchored, frame.width, frame.height);
                    liveAnchor = anchor;
                    anchorLocked = true;
                    lastAnchorFrame = frame.frameNumber;
                    if (verbose) {
                        std::lock_guard<std::mutex> lock(printMutex);
                        std::cout << "[anchor] score=" << anchor.score << " scale=" << anchor.scale
                                  << " at (" << anchor.x << "," << anchor.y << ") "
                                  << anchor.w << "x" << anchor.h
                                  << " -> table (" << anchoredTableROI.x << "," << anchoredTableROI.y << ") "
                                  << anchoredTableROI.w << "x" << anchoredTableROI.h
                                  << std::endl;
                    }
                } else if (verbose) {
                    std::lock_guard<std::mutex> lock(printMutex);
                    std::cout << "[anchor] match failed (score < " << anchorThreshold << ")" << std::endl;
                }
            }
            if (anchorLocked) {
                activeTableROI = anchoredTableROI;
            }
        }

        auto dumpPlane0 = [&](float* d_input, size_t bufferIndex, const std::string& tag, const ROI& roiClientToLog, const ROI& roiCapToLog) {
            if (!dumpRoi) return;
            if (bufferIndex >= dumped.size() || dumped[bufferIndex]) return;

            const int dumpW = 320;
            const int dumpH = 48;
            const size_t planeElems = static_cast<size_t>(dumpW) * static_cast<size_t>(dumpH);
            h_roi_plane.resize(planeElems);
            h_roi_img.resize(planeElems);

            cudaStreamSynchronize(stream);
            cudaMemcpy(h_roi_plane.data(), d_input, planeElems * sizeof(float), cudaMemcpyDeviceToHost);

            float minV = 1e9f, maxV = -1e9f, sumV = 0.0f;
            for (size_t i = 0; i < planeElems; i++) {
                float v = h_roi_plane[i];
                if (!(v == v)) v = -1.0f;
                minV = (std::min)(minV, v);
                maxV = (std::max)(maxV, v);
                sumV += v;
                float u = (v + 1.0f) * 0.5f;
                if (u < 0.0f) u = 0.0f;
                if (u > 1.0f) u = 1.0f;
                h_roi_img[i] = static_cast<uint8_t>(u * 255.0f + 0.5f);
            }
            const float meanV = static_cast<float>(sumV / static_cast<double>(planeElems));
            const int activeW = roiExtractor.getActiveOutputWidth(bufferIndex);

            const std::string base = tag + "_frame" + std::to_string(frame.frameNumber);
            const std::string outPath = (fs::path(dumpRoiDir) / (base + ".pgm")).string();
            const bool ok = writePGM8(outPath, h_roi_img, dumpW, dumpH);

            std::lock_guard<std::mutex> lock(printMutex);
            std::cout << "[dump] " << tag
                      << " activeW=" << activeW
                      << " roiClient=(x=" << roiClientToLog.x << ",y=" << roiClientToLog.y
                      << ",w=" << roiClientToLog.w << ",h=" << roiClientToLog.h << ")"
                      << " roiCap=(x=" << roiCapToLog.x << ",y=" << roiCapToLog.y
                      << ",w=" << roiCapToLog.w << ",h=" << roiCapToLog.h << ")"
                      << " stats(min=" << minV << ", max=" << maxV << ", mean=" << meanV << ")"
                      << " -> " << (ok ? outPath : std::string("(write failed)"))
                      << std::endl;

            dumped[bufferIndex] = true;
        };

        if (scanTable) {
            // Build list of row ROIs - either from auto-detection or fixed parameters
            std::vector<ROI> rowRoisClient;

            if (autoRows) {
                // Auto-detect rows: first extract the table ROI as raw BGRA
                ROI tableCapRoi = hwndForMap
                    ? mapClientROIToCapture(hwndForMap, activeTableROI, frame.width, frame.height)
                    : clampROIToFrame(activeTableROI, frame.width, frame.height);

                // Detect rows every N frames or use cached results
                static uint64_t lastDetectionFrame = 0;
                const uint64_t detectionInterval = templateActive ? 5 : 30;  // Re-detect every N frames

                if (frame.frameNumber - lastDetectionFrame >= detectionInterval || detectedRows.empty()) {
                    if (templateActive) {
                        auto newRows = rowDetector.detectRowsGPU(texObj, tableCapRoi, stream);
                        if (!newRows.empty()) {
                            detectedRows = std::move(newRows);

                            const float scaleY = (activeTableROI.h > 0)
                                ? static_cast<float>(tableCapRoi.h) / static_cast<float>(activeTableROI.h)
                                : 1.0f;

                            if (scaleY > 0.0f && scaleY != 1.0f) {
                                for (auto& dr : detectedRows) {
                                    dr.yStart = static_cast<int>(std::lround(dr.yStart / scaleY));
                                    dr.yEnd = static_cast<int>(std::lround(dr.yEnd / scaleY));
                                    dr.height = (std::max)(1, dr.yEnd - dr.yStart + 1);
                                }
                            }

                            lastDetectionFrame = frame.frameNumber;

                            if (verbose) {
                                std::lock_guard<std::mutex> lock(printMutex);
                                std::cout << "[auto-rows] Detected " << detectedRows.size() << " rows in table" << std::endl;
                            }
                        } else if (detectedRows.empty()) {
                            std::cerr << "[auto-rows] Template detection failed; keeping last rows." << std::endl;
                        }
                    } else {
                        // Fallback: generate synthetic rows when template mode is disabled
                        detectedRows.clear();

                        const int estimatedRowHeight = 10;
                        const int estimatedRowGap = 1;
                        const int headerOffset = 47;  // Skip header rows

                        for (int y = headerOffset; y + estimatedRowHeight <= activeTableROI.h; y += estimatedRowHeight + estimatedRowGap) {
                            DetectedRow dr;
                            dr.yStart = y;
                            dr.yEnd = y + estimatedRowHeight - 1;
                            dr.height = estimatedRowHeight;
                            dr.hasGreenPnL = false;
                            dr.hasRedPnL = false;
                            dr.confidence = 0.7f;
                            detectedRows.push_back(dr);

                            if (detectedRows.size() >= 10) break;  // Max 10 rows
                        }

                        lastDetectionFrame = frame.frameNumber;

                        if (verbose && !detectedRows.empty()) {
                            std::lock_guard<std::mutex> lock(printMutex);
                            std::cout << "[auto-rows] Using synthetic row layout" << std::endl;
                        }
                    }
                }

                // Convert detected rows to ROIs
                rowRoisClient = rowDetector.rowsToROIs(detectedRows, activeTableROI, scanTableRoiName + "_auto");
            } else {
                // Fixed row parameters
                for (int r = 0; r < tableRows; r++) {
                    ROI rowRoi;
                    rowRoi.name = scanTableRoiName + "_row" + std::to_string(r);
                    rowRoi.x = activeTableROI.x;
                    rowRoi.y = activeTableROI.y + rowOffsetYPx + r * rowStridePx;
                    rowRoi.w = activeTableROI.w;
                    rowRoi.h = rowHeightPx;
                    rowRoisClient.push_back(rowRoi);
                }
            }

            // Apply column extraction if columns are defined
            if (!columnDefs.empty()) {
                const float columnScale = anchorLocked ? liveAnchor.scale : 1.0f;
                std::vector<ROI> columnRois;
                for (const auto& rowRoi : rowRoisClient) {
                    for (const auto& col : columnDefs) {
                        ROI colRoi;
                        colRoi.name = rowRoi.name + "_" + col.name;
                        const int scaledOffsetX = static_cast<int>(std::lround(col.xOffset * columnScale));
                        const int scaledWidth = static_cast<int>(std::lround(col.width * columnScale));
                        colRoi.x = rowRoi.x + scaledOffsetX;
                        colRoi.y = rowRoi.y;
                        colRoi.w = (std::max)(1, scaledWidth);
                        colRoi.h = rowRoi.h;
                        columnRois.push_back(colRoi);
                    }
                }
                rowRoisClient = std::move(columnRois);
            }

            // Process each row/column
            for (size_t r = 0; r < rowRoisClient.size(); r++) {
                const ROI& rowRoiClient = rowRoisClient[r];

                ROI rowRoiCap = hwndForMap
                    ? mapClientROIToCapture(hwndForMap, rowRoiClient, frame.width, frame.height)
                    : clampROIToFrame(rowRoiClient, frame.width, frame.height);

                const size_t bufferIndex = r;
                float* d_input = roiExtractor.extractROI(texObj, rowRoiCap, bufferIndex, stream);
                if (!d_input) continue;

                dumpPlane0(d_input, bufferIndex, rowRoiClient.name, rowRoiClient, rowRoiCap);

                if (enableChangeDetect && bufferIndex < changeDetector.getSlotCount()) {
                    const int activeW = roiExtractor.getActiveOutputWidth(bufferIndex);
                    if (activeW > 0) {
                        const bool changed = changeDetector.hasChangedStrided(
                            bufferIndex,
                            d_input,
                            320,
                            activeW,
                            48,
                            stream
                        );
                        if (!changed && !printAllRows) {
                            continue;
                        }
                    }
                }

                if (!svtr.infer(d_input, 320, 48, stream)) continue;
                cudaStreamSynchronize(stream);

                float* d_probs = svtr.getOutputProbs();
                cudaMemcpy(h_output.data(), d_probs, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

                CTCResult res = decoder.decode(h_output.data(), timesteps, numClasses, 0);
                const std::string ticker = extractTickerLike(res.text);
                const std::string numeric = filterNumericLike(res.text);
                const std::string rowKey = ticker + "|" + numeric;

                const bool shouldPrint = printAllRows || (bufferIndex < lastRowKey.size() && rowKey != lastRowKey[bufferIndex]);
                if (bufferIndex < lastRowKey.size()) {
                    lastRowKey[bufferIndex] = rowKey;
                }

                if (shouldPrint) {
                    std::lock_guard<std::mutex> lock(printMutex);
                    std::cout << "[frame " << frame.frameNumber << "] "
                              << rowRoiClient.name << ": "
                              << (ticker.empty() ? std::string("(no ticker)") : ticker)
                              << "  " << numeric;
                    if (verbose) {
                        std::cout << " (raw='" << res.text << "', conf=" << res.confidence << ")";
                    }
                    std::cout << std::endl;
                }
            }
        } else {
            for (size_t idx : roiIndices) {
                const ROI roiClient = cfg.rois[idx];
                const ROI roiCap = hwndForMap
                    ? mapClientROIToCapture(hwndForMap, roiClient, frame.width, frame.height)
                    : clampROIToFrame(roiClient, frame.width, frame.height);

                float* d_input = roiExtractor.extractROI(texObj, roiCap, idx, stream);
                if (!d_input) continue;

                if (dumpRoi && idx < dumped.size() && !dumped[idx]) {
                    dumpPlane0(d_input, idx, cfg.rois[idx].name, roiClient, roiCap);
                }

                if (enableChangeDetect) {
                    const int activeW = roiExtractor.getActiveOutputWidth(idx);
                    if (activeW > 0) {
                        const bool changed = changeDetector.hasChangedStrided(
                            idx,
                            d_input,
                            320,
                            activeW,
                            48,
                            stream
                        );
                        if (!changed) {
                            continue;
                        }
                    }
                }

                if (!svtr.infer(d_input, 320, 48, stream)) continue;
                cudaStreamSynchronize(stream);

                float* d_probs = svtr.getOutputProbs();
                cudaMemcpy(h_output.data(), d_probs, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

                CTCResult res = decoder.decode(h_output.data(), timesteps, numClasses, 0);
                std::string numeric = filterNumericLike(res.text);

                std::lock_guard<std::mutex> lock(printMutex);
                std::cout << "[frame " << frame.frameNumber << "] "
                          << cfg.rois[idx].name << ": "
                          << numeric;
                if (verbose) {
                    std::cout << " (raw='" << res.text << "', conf=" << res.confidence << ")";
                }
                std::cout << std::endl;
            }
        }

        interop.destroyTextureObject(texObj);
    });

    while (g_running.load() && capture.isCapturing()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    capture.stopCapture();
    stopROIOverlay(overlayHandle);
    cudaStreamDestroy(stream);
    
    return 0;
}

