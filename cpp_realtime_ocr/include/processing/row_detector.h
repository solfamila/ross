#pragma once
/**
 * @file row_detector.h
 * @brief Dynamic row detection for trading table windows
 * 
 * Analyzes pixel intensity and color patterns to automatically detect
 * row boundaries in position/order tables.
 */

#include "types.h"
#include "processing/template_matcher.h"
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <cstdint>

namespace trading_monitor {

/**
 * @brief Detection mode for row detector
 */
enum class RowDetectionMode {
    INTENSITY_BASED,   ///< Original intensity/color analysis (CPU)
    TEMPLATE_BASED,    ///< Template matching (GPU)
    HYBRID             ///< Template for table location, intensity for rows
};

/**
 * @brief Detected row information
 */
struct DetectedRow {
    int yStart;         ///< Row start Y coordinate (relative to table ROI)
    int yEnd;           ///< Row end Y coordinate (relative to table ROI)
    int height;         ///< Row height in pixels
    bool hasGreenPnL;   ///< True if row contains green (positive) P&L
    bool hasRedPnL;     ///< True if row contains red (negative) P&L
    float confidence;   ///< Detection confidence (0.0-1.0)
};

/**
 * @brief Dynamic row detector for trading tables
 * 
 * Uses pixel analysis to find text rows in position/order tables:
 * - Intensity analysis to find text vs background
 * - Color analysis to identify P&L indicators (green=profit, red=loss)
 * - Row boundary detection based on horizontal gaps
 */
class RowDetector {
public:
    RowDetector();
    ~RowDetector();

    /**
     * @brief Initialize the row detector
     * @param maxWidth Maximum table width to support
     * @param maxHeight Maximum table height to support
     * @return true if successful
     */
    bool initialize(int maxWidth, int maxHeight);

    /**
     * @brief Detect rows from CPU image data (BGRA format)
     * @param imageData Pointer to BGRA image data
     * @param width Image width
     * @param height Image height
     * @param stride Row stride in bytes
     * @return Vector of detected rows
     */
    std::vector<DetectedRow> detectRows(
        const uint8_t* imageData,
        int width,
        int height,
        int stride
    );

    /**
     * @brief Detect rows from GPU texture
     * @param texObj CUDA texture object
     * @param tableRoi Table ROI coordinates
     * @param stream CUDA stream
     * @return Vector of detected rows
     */
    std::vector<DetectedRow> detectRowsGPU(
        cudaTextureObject_t texObj,
        const ROI& tableRoi,
        cudaStream_t stream = 0
    );

    /**
     * @brief Convert detected rows to ROI objects
     * @param rows Detected rows
     * @param tableRoi Parent table ROI (for absolute positioning)
     * @param baseName Base name for generated ROIs
     * @return Vector of ROI objects
     */
    std::vector<ROI> rowsToROIs(
        const std::vector<DetectedRow>& rows,
        const ROI& tableRoi,
        const std::string& baseName = "detected_row"
    );

    /**
     * @brief Set minimum row height threshold
     */
    void setMinRowHeight(int pixels) { m_minRowHeight = pixels; }

    /**
     * @brief Set minimum intensity for text detection
     */
    void setTextIntensityThreshold(int threshold) { m_textIntensityThreshold = threshold; }

    /**
     * @brief Set color thresholds for P&L detection
     */
    void setColorThresholds(int greenThreshold, int redThreshold) {
        m_greenThreshold = greenThreshold;
        m_redThreshold = redThreshold;
    }

    /**
     * @brief Set detection mode
     */
    void setDetectionMode(RowDetectionMode mode) { m_detectionMode = mode; }
    RowDetectionMode getDetectionMode() const { return m_detectionMode; }

    /**
     * @brief Load templates for template-based detection
     * @param templateDir Directory containing .tmpl files
     * @return true if templates loaded successfully
     */
    bool loadTemplates(const std::string& templateDir);

    /**
     * @brief Detect rows using GPU template matching
     * @param d_grayscaleImage GPU pointer to grayscale image
     * @param width Image width
     * @param height Image height
     * @param pitch Image pitch (bytes per row)
     * @param stream CUDA stream
     * @return Vector of detected rows
     */
    std::vector<DetectedRow> detectRowsTemplateGPU(
        const uint8_t* d_grayscaleImage,
        int width,
        int height,
        int pitch,
        cudaStream_t stream = nullptr
    );

    /**
     * @brief Get template matcher for advanced configuration
     */
    trading::TemplateMatcher* getTemplateMatcher() { return m_templateMatcher.get(); }

    /**
     * @brief Get last table detection result
     */
    const trading::TableDetection& getLastTableDetection() const { return m_lastTableDetection; }

    void cleanup();

private:
    // Detection mode
    RowDetectionMode m_detectionMode = RowDetectionMode::INTENSITY_BASED;

    // Row detection parameters
    int m_minRowHeight = 10;
    int m_maxRowHeight = 25;  // Allow for expanded rows
    int m_textIntensityThreshold = 80;
    int m_greenThreshold = 100;
    int m_redThreshold = 100;
    int m_minGapRows = 2;

    // GPU buffers
    uint8_t* m_d_imageBuffer = nullptr;
    float* m_d_rowIntensity = nullptr;
    int* m_d_colorCounts = nullptr;
    int m_maxWidth = 0;
    int m_maxHeight = 0;

    // Grayscale buffer for template matching (uint8)
    uint8_t* m_d_grayBuffer = nullptr;
    size_t m_grayPitch = 0;
    int m_grayWidth = 0;
    int m_grayHeight = 0;

    // CPU working buffers
    std::vector<float> m_rowIntensity;
    std::vector<int> m_greenCounts;
    std::vector<int> m_redCounts;

    // Template matching (for TEMPLATE_BASED and HYBRID modes)
    std::unique_ptr<trading::TemplateMatcher> m_templateMatcher;
    trading::TableDetection m_lastTableDetection;

    // Internal methods
    void analyzeRowIntensity(const uint8_t* data, int width, int height, int stride);
    void analyzeRowColors(const uint8_t* data, int width, int height, int stride);
    std::vector<DetectedRow> findRowBoundaries();

    // Convert template rows to DetectedRow format
    std::vector<DetectedRow> convertTemplateRows(
        const std::vector<trading::DetectedTableRow>& templateRows,
        const uint8_t* imageData, int width, int stride);
};

} // namespace trading_monitor

