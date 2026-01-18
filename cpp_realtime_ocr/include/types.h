#pragma once
/**
 * @file types.h
 * @brief Common type definitions for Trading Screen Monitor
 */

#include <optional>
#include <string>
#include <vector>
#include <chrono>
#include <cstdint>

namespace trading_monitor {

/**
 * @brief ROI (Region of Interest) configuration
 */
struct ROI {
    std::string name;   ///< ROI identifier (e.g., "lightspeed_position_row")
    int x;              ///< Top-left X coordinate
    int y;              ///< Top-left Y coordinate  
    int w;              ///< Width in pixels
    int h;              ///< Height in pixels
};

/**
 * @brief Parsed position data from OCR
 */
struct PositionData {
    std::optional<int> shares;              ///< Number of shares held
    std::optional<double> avgPrice;         ///< Average entry price
    std::optional<double> unrealizedPnl;    ///< Unrealized P&L
    std::optional<double> realizedPnl;      ///< Realized P&L
    std::string rawText;                    ///< Raw OCR output
    double confidence = 0.0;                ///< OCR confidence (0-1)
    
    bool isValid() const {
        return shares.has_value() && avgPrice.has_value();
    }
};

/**
 * @brief OCR result for a single text region
 */
struct OCRResult {
    std::string text;           ///< Recognized text
    float confidence;           ///< Confidence score (0-1)
    std::vector<float> boxes;   ///< Bounding box coordinates [x1,y1,x2,y2,...]
};

/**
 * @brief Frame capture metadata
 */
struct FrameInfo {
    uint64_t frameNumber;                   ///< Sequential frame number
    std::chrono::steady_clock::time_point captureTime;  ///< Capture timestamp
    int width;                              ///< Frame width
    int height;                             ///< Frame height
    bool hasChanged;                        ///< True if content changed from previous frame
};

/**
 * @brief Pipeline timing statistics
 */
struct PipelineStats {
    double captureMs = 0.0;         ///< Screen capture time
    double interopMs = 0.0;         ///< D3D11-CUDA interop time
    double preprocessMs = 0.0;      ///< ROI extraction + preprocessing
    double inferenceMs = 0.0;       ///< TensorRT inference time
    double decodeMs = 0.0;          ///< CTC decoding time
    double parseMs = 0.0;           ///< Text parsing time
    double totalMs = 0.0;           ///< Total pipeline time
    
    void reset() {
        captureMs = interopMs = preprocessMs = 0.0;
        inferenceMs = decodeMs = parseMs = totalMs = 0.0;
    }
};

/**
 * @brief Configuration for the OCR pipeline
 */
struct PipelineConfig {
    // ROI settings
    std::vector<ROI> rois;
    
    // Preprocessing
    float upscaleFactor = 2.0f;         ///< ROI upscaling factor
    float contrastAlpha = 1.3f;         ///< Contrast enhancement factor
    float contrastBeta = 0.0f;          ///< Brightness adjustment
    
    // Model paths
    std::string detEnginePrefix;         ///< Detection engine path (optional)
    std::string recEnginePath;           ///< Recognition engine path
    std::string dictPath;                ///< Character dictionary path
    
    // Performance
    int maxBatchSize = 1;               ///< TensorRT batch size
    bool useFP16 = true;                ///< Use FP16 inference
    bool enableChangeDetection = true;  ///< Skip unchanged frames
    
    // Capture
    int targetFPS = 60;                 ///< Target capture rate
    std::string captureWindowTitle;     ///< Window to capture (empty = monitor)
};

} // namespace trading_monitor

