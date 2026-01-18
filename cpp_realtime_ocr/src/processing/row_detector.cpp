/**
 * @file row_detector.cpp
 * @brief Implementation of dynamic row detection for trading tables
 */

#include "processing/row_detector.h"
#include "processing/cuda_kernels.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <filesystem>

namespace trading_monitor {

RowDetector::RowDetector() : m_templateMatcher(std::make_unique<trading::TemplateMatcher>()) {}

RowDetector::~RowDetector() {
    cleanup();
}

bool RowDetector::initialize(int maxWidth, int maxHeight) {
    cleanup();

    m_maxWidth = maxWidth;
    m_maxHeight = maxHeight;

    // Allocate CPU working buffers
    m_rowIntensity.resize(static_cast<size_t>(maxHeight));
    m_greenCounts.resize(static_cast<size_t>(maxHeight));
    m_redCounts.resize(static_cast<size_t>(maxHeight));

    // GPU buffers allocated on demand in detectRowsGPU
    return true;
}

void RowDetector::cleanup() {
    if (m_d_imageBuffer) {
        cudaFree(m_d_imageBuffer);
        m_d_imageBuffer = nullptr;
    }
    if (m_d_rowIntensity) {
        cudaFree(m_d_rowIntensity);
        m_d_rowIntensity = nullptr;
    }
    if (m_d_colorCounts) {
        cudaFree(m_d_colorCounts);
        m_d_colorCounts = nullptr;
    }
    if (m_d_grayBuffer) {
        cudaFree(m_d_grayBuffer);
        m_d_grayBuffer = nullptr;
    }
    m_grayPitch = 0;
    m_grayWidth = 0;
    m_grayHeight = 0;
    m_rowIntensity.clear();
    m_greenCounts.clear();
    m_redCounts.clear();
}

void RowDetector::analyzeRowIntensity(const uint8_t* data, int width, int height, int stride) {
    // Calculate average intensity per row
    for (int y = 0; y < height; y++) {
        const uint8_t* row = data + y * stride;
        float sum = 0.0f;
        for (int x = 0; x < width; x++) {
            // BGRA format
            const uint8_t* pixel = row + x * 4;
            float intensity = (pixel[0] + pixel[1] + pixel[2]) / 3.0f;
            sum += intensity;
        }
        m_rowIntensity[y] = sum / static_cast<float>(width);
    }
}

void RowDetector::analyzeRowColors(const uint8_t* data, int width, int height, int stride) {
    // Count green and red pixels per row (for P&L detection)
    // Focus on right side of table where P&L values typically are
    const int pnlStartX = width * 2 / 3;  // Last 1/3 of width
    
    for (int y = 0; y < height; y++) {
        const uint8_t* row = data + y * stride;
        int greenCount = 0;
        int redCount = 0;
        
        for (int x = pnlStartX; x < width; x++) {
            const uint8_t* pixel = row + x * 4;
            // BGRA format: pixel[0]=B, pixel[1]=G, pixel[2]=R, pixel[3]=A
            uint8_t g = pixel[1];
            uint8_t r = pixel[2];

            // Green pixel: high G, low R
            if (g > m_greenThreshold && r < 80) {
                greenCount++;
            }
            // Red pixel: high R, low G
            if (r > m_redThreshold && g < 80) {
                redCount++;
            }
        }
        
        m_greenCounts[y] = greenCount;
        m_redCounts[y] = redCount;
    }
}

std::vector<DetectedRow> RowDetector::findRowBoundaries() {
    std::vector<DetectedRow> rows;
    const int height = static_cast<int>(m_rowIntensity.size());

    // Trading app row height is typically 10-16 pixels
    // Lightspeed rows are about 11px apart
    const int kRowHeight = 14;
    const int kMinRowSpacing = 10;  // Minimum Y distance between row starts

    // Find row centers by looking for peaks in intensity/color
    std::vector<int> rowCenters;

    for (int y = 2; y < height - 2; y++) {
        float intensity = m_rowIntensity[y];
        int greenPx = m_greenCounts[y];
        int redPx = m_redCounts[y];

        // Check if this line has significant content
        bool hasContent = (intensity > m_textIntensityThreshold) ||
                         (greenPx > 3) || (redPx > 3);

        if (hasContent) {
            // Check if this is a local maximum (peak detection)
            float prevIntensity = m_rowIntensity[y - 1];
            float nextIntensity = m_rowIntensity[y + 1];
            int prevGreen = m_greenCounts[y - 1];
            int nextGreen = m_greenCounts[y + 1];
            int prevRed = m_redCounts[y - 1];
            int nextRed = m_redCounts[y + 1];

            float currScore = intensity + greenPx * 0.5f + redPx * 0.5f;
            float prevScore = prevIntensity + prevGreen * 0.5f + prevRed * 0.5f;
            float nextScore = nextIntensity + nextGreen * 0.5f + nextRed * 0.5f;

            // Only add if this is a peak and far enough from last center
            if (currScore >= prevScore && currScore >= nextScore) {
                if (rowCenters.empty() || (y - rowCenters.back()) >= kMinRowSpacing) {
                    rowCenters.push_back(y);
                }
            }
        }
    }

    // Convert centers to row boundaries with fixed height
    for (int center : rowCenters) {
        int rowStart = (std::max)(0, center - kRowHeight / 2);
        int rowEnd = (std::min)(height - 1, center + kRowHeight / 2);
        int rowHeight = rowEnd - rowStart + 1;

        if (rowHeight >= m_minRowHeight) {
            DetectedRow dr;
            dr.yStart = rowStart;
            dr.yEnd = rowEnd;
            dr.height = rowHeight;

            // Check P&L colors in this row range
            dr.hasGreenPnL = false;
            dr.hasRedPnL = false;
            for (int ry = rowStart; ry <= rowEnd; ry++) {
                if (m_greenCounts[ry] > 3) dr.hasGreenPnL = true;
                if (m_redCounts[ry] > 3) dr.hasRedPnL = true;
            }

            dr.confidence = 0.8f;
            if (dr.hasGreenPnL || dr.hasRedPnL) {
                dr.confidence = 0.95f;
            }

            rows.push_back(dr);
        }
    }

    return rows;
}

std::vector<DetectedRow> RowDetector::detectRows(
    const uint8_t* imageData,
    int width,
    int height,
    int stride
) {
    if (!imageData || width <= 0 || height <= 0) {
        return {};
    }

    // Ensure buffers are sized correctly
    if (static_cast<int>(m_rowIntensity.size()) < height) {
        m_rowIntensity.resize(static_cast<size_t>(height));
        m_greenCounts.resize(static_cast<size_t>(height));
        m_redCounts.resize(static_cast<size_t>(height));
    }

    // Analyze image
    analyzeRowIntensity(imageData, width, height, stride);
    analyzeRowColors(imageData, width, height, stride);

    // Find row boundaries
    return findRowBoundaries();
}

std::vector<DetectedRow> RowDetector::detectRowsGPU(
    cudaTextureObject_t texObj,
    const ROI& tableRoi,
    cudaStream_t stream
) {
    if (m_detectionMode == RowDetectionMode::INTENSITY_BASED) {
        std::cerr << "RowDetector::detectRowsGPU: intensity mode not supported on GPU" << std::endl;
        return {};
    }

    if (!m_templateMatcher) {
        std::cerr << "RowDetector::detectRowsGPU: template matcher not initialized" << std::endl;
        return {};
    }

    if (m_templateMatcher->getTemplates().empty()) {
        std::cerr << "RowDetector::detectRowsGPU: no templates loaded" << std::endl;
        return {};
    }

    // Ensure template matcher uses the current stream
    m_templateMatcher->initialize(stream);

    const int width = tableRoi.w;
    const int height = tableRoi.h;

    if (width <= 0 || height <= 0) {
        return {};
    }

    // Allocate grayscale buffer if needed
    if (!m_d_grayBuffer || width != m_grayWidth || height != m_grayHeight) {
        if (m_d_grayBuffer) cudaFree(m_d_grayBuffer);
        cudaError_t err = cudaMallocPitch(&m_d_grayBuffer, &m_grayPitch,
                                          static_cast<size_t>(width), static_cast<size_t>(height));
        if (err != cudaSuccess) {
            std::cerr << "RowDetector::detectRowsGPU: failed to allocate grayscale buffer: "
                      << cudaGetErrorString(err) << std::endl;
            m_d_grayBuffer = nullptr;
            m_grayPitch = 0;
            return {};
        }
        m_grayWidth = width;
        m_grayHeight = height;
    }

    // Convert table ROI to grayscale uint8 on GPU
    cudaError_t convErr = trading_monitor::cuda::launchBgraTexToGrayU8ROI(
        texObj,
        m_d_grayBuffer,
        m_grayPitch,
        tableRoi.x,
        tableRoi.y,
        width,
        height,
        stream
    );

    if (convErr != cudaSuccess) {
        std::cerr << "RowDetector::detectRowsGPU: grayscale conversion failed: "
                  << cudaGetErrorString(convErr) << std::endl;
        return {};
    }

    return detectRowsTemplateGPU(m_d_grayBuffer, width, height, static_cast<int>(m_grayPitch), stream);
}

std::vector<ROI> RowDetector::rowsToROIs(
    const std::vector<DetectedRow>& rows,
    const ROI& tableRoi,
    const std::string& baseName
) {
    std::vector<ROI> rois;
    rois.reserve(rows.size());

    // Minimum row height for OCR - helps with very thin detected rows
    constexpr int MIN_ROW_HEIGHT = 14;
    // Padding to add above/below detected row for better OCR coverage
    constexpr int VERTICAL_PADDING = 2;

    for (size_t i = 0; i < rows.size(); i++) {
        const auto& row = rows[i];

        ROI roi;
        roi.name = baseName + "_" + std::to_string(i);
        roi.x = tableRoi.x;

        // Calculate padded bounds
        int yStart = row.yStart - VERTICAL_PADDING;
        int yEnd = row.yEnd + VERTICAL_PADDING;
        int height = yEnd - yStart;

        // Ensure minimum height
        if (height < MIN_ROW_HEIGHT) {
            int extraPadding = (MIN_ROW_HEIGHT - height) / 2;
            yStart -= extraPadding;
            height = MIN_ROW_HEIGHT;
        }

        // Clamp to table bounds
        if (yStart < 0) yStart = 0;
        if (yStart + height > tableRoi.h) {
            height = tableRoi.h - yStart;
        }

        roi.y = tableRoi.y + yStart;
        roi.h = height;
        roi.w = tableRoi.w;

        rois.push_back(roi);
    }

    return rois;
}

bool RowDetector::loadTemplates(const std::string& templateDir) {
    if (!m_templateMatcher) {
        m_templateMatcher = std::make_unique<trading::TemplateMatcher>();
    }

    if (!m_templateMatcher->initialize()) {
        std::cerr << "RowDetector: Failed to initialize template matcher" << std::endl;
        return false;
    }

    // Load all .tmpl files from directory
    std::filesystem::path dirPath(templateDir);
    if (!std::filesystem::exists(dirPath)) {
        std::cerr << "RowDetector: Template directory not found: " << templateDir << std::endl;
        return false;
    }

    int loaded = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
        if (entry.path().extension() == ".tmpl") {
            std::string name = entry.path().stem().string();
            trading::TemplateType type = trading::TemplateType::ROW_BACKGROUND;

            // Determine type from name
            if (name.find("header") != std::string::npos) {
                type = trading::TemplateType::TABLE_HEADER;
            } else if (name.find("col_") != std::string::npos) {
                type = trading::TemplateType::COLUMN_HEADER;
            } else if (name.find("separator") != std::string::npos) {
                type = trading::TemplateType::ROW_SEPARATOR;
            }

            if (m_templateMatcher->loadTemplate(entry.path().string(), type, name)) {
                loaded++;
                std::cout << "RowDetector: Loaded template: " << name << std::endl;
            } else {
                std::cerr << "RowDetector: Failed to load template: " << entry.path() << std::endl;
            }
        }
    }

    if (loaded == 0) {
        std::cerr << "RowDetector: No templates loaded from " << templateDir << std::endl;
        return false;
    }

    std::cout << "RowDetector: Loaded " << loaded << " templates" << std::endl;
    return true;
}

std::vector<DetectedRow> RowDetector::detectRowsTemplateGPU(
    const uint8_t* d_grayscaleImage,
    int width,
    int height,
    int pitch,
    cudaStream_t stream
) {
    std::vector<DetectedRow> result;

    if (!m_templateMatcher) {
        std::cerr << "RowDetector: Template matcher not initialized" << std::endl;
        return result;
    }

    // Detect table and rows using template matching
    m_lastTableDetection = m_templateMatcher->detectTable(
        d_grayscaleImage, width, height, pitch,
        0, 0, width, height  // Search entire image
    );

    if (m_lastTableDetection.confidence < 0.5f) {
        std::cerr << "RowDetector: Table not detected (confidence: "
                  << m_lastTableDetection.confidence << ")" << std::endl;
        return result;
    }

    // Convert template rows to DetectedRow format
    // Note: tmplRow.yStart/yEnd are already absolute coordinates within the ROI
    // We keep them as-is since rowsToROIs will add tableRoi.y
    for (const auto& tmplRow : m_lastTableDetection.rows) {
        DetectedRow row;
        row.yStart = tmplRow.yStart;  // Already relative to ROI origin
        row.yEnd = tmplRow.yEnd;
        row.height = tmplRow.height;
        row.confidence = tmplRow.confidence;
        row.hasGreenPnL = false;  // Would need color analysis
        row.hasRedPnL = false;

        if (row.height >= m_minRowHeight && row.height <= m_maxRowHeight) {
            result.push_back(row);
        }
    }

    return result;
}

std::vector<DetectedRow> RowDetector::convertTemplateRows(
    const std::vector<trading::DetectedTableRow>& templateRows,
    const uint8_t* imageData, int width, int stride
) {
    std::vector<DetectedRow> result;
    result.reserve(templateRows.size());

    for (const auto& tmplRow : templateRows) {
        DetectedRow row;
        row.yStart = tmplRow.yStart;
        row.yEnd = tmplRow.yEnd;
        row.height = tmplRow.height;
        row.confidence = tmplRow.confidence;
        row.hasGreenPnL = false;
        row.hasRedPnL = false;

        // Analyze colors in this row if image data available
        if (imageData) {
            const int pnlStartX = width * 2 / 3;
            for (int y = row.yStart; y <= row.yEnd && y >= 0; y++) {
                const uint8_t* rowPtr = imageData + y * stride;
                for (int x = pnlStartX; x < width; x++) {
                    const uint8_t* pixel = rowPtr + x * 4;
                    uint8_t g = pixel[1];
                    uint8_t r = pixel[2];

                    if (g > m_greenThreshold && r < 80) row.hasGreenPnL = true;
                    if (r > m_redThreshold && g < 80) row.hasRedPnL = true;
                }
            }
        }

        result.push_back(row);
    }

    return result;
}

} // namespace trading_monitor

