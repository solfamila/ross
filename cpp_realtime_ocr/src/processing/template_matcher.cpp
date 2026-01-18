/**
 * Template Matcher Implementation
 * CPU-side logic for template matching and table detection
 */

#include "processing/template_matcher.h"
#include <fstream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <iostream>

// External CUDA kernel launchers
extern "C" {
    cudaError_t launchNCCTemplateMatch(
        const unsigned char* d_image,
        int imgWidth, int imgHeight, int pitch,
        const float* d_template,
        int templateWidth, int templateHeight,
        float templateSum, float templateSqSum,
        int searchX, int searchY, int searchWidth, int searchHeight,
        float* d_scores,
        cudaStream_t stream
    );

    cudaError_t launchSSDTemplateMatch(
        const unsigned char* d_image,
        int imgWidth, int imgHeight, int pitch,
        const float* d_template,
        int templateWidth, int templateHeight,
        int searchX, int searchY, int searchWidth, int searchHeight,
        float* d_scores,
        cudaStream_t stream
    );

    cudaError_t launchFindMaxScore(
        const float* d_scores,
        int width, int height,
        float* d_maxScore,
        int* d_maxX,
        int* d_maxY,
        cudaStream_t stream
    );

    cudaError_t launchDetectRowBoundaries(
        const unsigned char* d_image,
        int imgWidth, int imgHeight, int pitch,
        int tableX, int tableY, int tableWidth, int tableHeight,
        int expectedRowHeight,
        int* d_rowStarts,
        int* d_rowEnds,
        float* d_rowConfidences,
        int* d_rowCount,
        int maxRows,
        cudaStream_t stream
    );
}

namespace trading {

TemplateMatcher::TemplateMatcher() = default;

TemplateMatcher::~TemplateMatcher() {
    freeGPUResources();
}

bool TemplateMatcher::initialize(cudaStream_t stream) {
    m_stream = stream;
    m_initialized = true;
    return true;
}

bool TemplateMatcher::loadTemplate(const uint8_t* pixels, int width, int height,
                                   TemplateType type, const std::string& name) {
    if (!pixels || width <= 0 || height <= 0) {
        m_lastError = "Invalid template data";
        return false;
    }

    TemplateData tmpl;
    tmpl.width = width;
    tmpl.height = height;
    tmpl.type = type;
    tmpl.name = name;
    tmpl.pixels.assign(pixels, pixels + width * height);

    // Compute template statistics
    computeTemplateStats(tmpl);

    // Upload to GPU
    if (!uploadTemplateToGPU(tmpl)) {
        return false;
    }

    m_templates.push_back(std::move(tmpl));
    return true;
}

bool TemplateMatcher::loadTemplate(const std::string& path, TemplateType type,
                                   const std::string& name) {
    // Simple raw grayscale file loader (width, height, then pixels)
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        m_lastError = "Failed to open template file: " + path;
        return false;
    }

    int32_t width, height;
    file.read(reinterpret_cast<char*>(&width), sizeof(width));
    file.read(reinterpret_cast<char*>(&height), sizeof(height));

    if (width <= 0 || height <= 0 || width > 1000 || height > 1000) {
        m_lastError = "Invalid template dimensions in file";
        return false;
    }

    std::vector<uint8_t> pixels(width * height);
    file.read(reinterpret_cast<char*>(pixels.data()), pixels.size());

    if (!file) {
        m_lastError = "Failed to read template data";
        return false;
    }

    return loadTemplate(pixels.data(), width, height, type, name);
}

void TemplateMatcher::computeTemplateStats(TemplateData& tmpl) {
    tmpl.templateSum = 0.0f;
    tmpl.templateSqSum = 0.0f;

    for (size_t i = 0; i < tmpl.pixels.size(); i++) {
        float val = static_cast<float>(tmpl.pixels[i]);
        tmpl.templateSum += val;
        tmpl.templateSqSum += val * val;
    }
}

bool TemplateMatcher::uploadTemplateToGPU(TemplateData& tmpl) {
    size_t templateSize = tmpl.width * tmpl.height;

    // Allocate GPU memory for template (as float for kernel compatibility)
    cudaError_t err = cudaMalloc(&tmpl.d_template, templateSize * sizeof(float));
    if (err != cudaSuccess) {
        m_lastError = "Failed to allocate GPU template memory";
        return false;
    }

    // Convert to float and upload
    std::vector<float> floatTemplate(templateSize);
    for (size_t i = 0; i < templateSize; i++) {
        floatTemplate[i] = static_cast<float>(tmpl.pixels[i]);
    }

    err = cudaMemcpy(tmpl.d_template, floatTemplate.data(),
                     templateSize * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(tmpl.d_template);
        tmpl.d_template = nullptr;
        m_lastError = "Failed to upload template to GPU";
        return false;
    }

    return true;
}

void TemplateMatcher::freeGPUResources() {
    for (auto& tmpl : m_templates) {
        if (tmpl.d_template) {
            cudaFree(tmpl.d_template);
            tmpl.d_template = nullptr;
        }
        if (tmpl.d_templateSq) {
            cudaFree(tmpl.d_templateSq);
            tmpl.d_templateSq = nullptr;
        }
    }

    if (d_matchScores) {
        cudaFree(d_matchScores);
        d_matchScores = nullptr;
        m_scoreBufferSize = 0;
    }
}

TemplateMatch TemplateMatcher::matchTemplate(const uint8_t* d_image, int width, int height,
                                              int pitch, const TemplateData& tmpl,
                                              int searchX, int searchY,
                                              int searchWidth, int searchHeight) {
    TemplateMatch result = {0, 0, 0.0f, false};

    if (!tmpl.d_template) {
        m_lastError = "Template not uploaded to GPU";
        return result;
    }

    // Ensure score buffer is large enough
    int scoreSize = searchWidth * searchHeight;
    if (scoreSize > m_scoreBufferSize) {
        if (d_matchScores) cudaFree(d_matchScores);
        cudaError_t err = cudaMalloc(&d_matchScores, scoreSize * sizeof(float));
        if (err != cudaSuccess) {
            m_lastError = "Failed to allocate score buffer";
            return result;
        }
        m_scoreBufferSize = scoreSize;
    }

    // Run template matching kernel
    cudaError_t err;
    if (m_config.useNCC) {
        err = launchNCCTemplateMatch(
            d_image, width, height, pitch,
            tmpl.d_template, tmpl.width, tmpl.height,
            tmpl.templateSum, tmpl.templateSqSum,
            searchX, searchY, searchWidth, searchHeight,
            d_matchScores, m_stream
        );
    } else {
        err = launchSSDTemplateMatch(
            d_image, width, height, pitch,
            tmpl.d_template, tmpl.width, tmpl.height,
            searchX, searchY, searchWidth, searchHeight,
            d_matchScores, m_stream
        );
    }

    if (err != cudaSuccess) {
        m_lastError = "Template matching kernel failed";
        return result;
    }

    // Find maximum score
    float* d_maxScore;
    int* d_maxX;
    int* d_maxY;
    cudaMalloc(&d_maxScore, sizeof(float));
    cudaMalloc(&d_maxX, sizeof(int));
    cudaMalloc(&d_maxY, sizeof(int));

    err = launchFindMaxScore(d_matchScores, searchWidth, searchHeight,
                             d_maxScore, d_maxX, d_maxY, m_stream);

    if (err == cudaSuccess) {
        cudaStreamSynchronize(m_stream);

        float maxScore;
        int maxX, maxY;
        cudaMemcpy(&maxScore, d_maxScore, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&maxX, d_maxX, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&maxY, d_maxY, sizeof(int), cudaMemcpyDeviceToHost);

        result.x = searchX + maxX;
        result.y = searchY + maxY;
        result.score = maxScore;
        result.valid = (maxScore >= m_config.matchThreshold);
    }

    cudaFree(d_maxScore);
    cudaFree(d_maxX);
    cudaFree(d_maxY);

    return result;
}

TableDetection TemplateMatcher::detectTable(const uint8_t* d_image, int width, int height,
                                            int pitch, int searchX, int searchY,
                                            int searchWidth, int searchHeight) {
    TableDetection result = {0, 0, 0, 0, 0.0f, {}};

    // Find header template
    const TemplateData* headerTemplate = nullptr;
    for (const auto& tmpl : m_templates) {
        if (tmpl.type == TemplateType::TABLE_HEADER) {
            headerTemplate = &tmpl;
            break;
        }
    }

    if (!headerTemplate) {
        m_lastError = "No header template loaded";
        return result;
    }

    // Match header template
    TemplateMatch headerMatch = matchTemplate(d_image, width, height, pitch,
                                              *headerTemplate, searchX, searchY,
                                              searchWidth, searchHeight);

    std::cerr << "[TemplateMatcher] Header match: x=" << headerMatch.x
              << " y=" << headerMatch.y << " score=" << headerMatch.score
              << " valid=" << headerMatch.valid
              << " (threshold=" << m_config.matchThreshold << ")" << std::endl;

    if (!headerMatch.valid) {
        m_lastError = "Header template not found";
        return result;
    }

    // Table starts at header position
    result.tableX = headerMatch.x;
    result.tableY = headerMatch.y;
    result.tableWidth = headerTemplate->width;  // Will be refined
    result.confidence = headerMatch.score;

    // Estimate table height (search for end or use max rows)
    int estimatedTableHeight = m_config.maxRows * (m_config.rowHeight + m_config.rowSpacing);
    result.tableHeight = std::min(estimatedTableHeight, height - result.tableY);

    // Detect rows within table
    result.rows = detectRows(d_image, width, height, pitch,
                             result.tableX, result.tableY + headerTemplate->height,
                             result.tableWidth, result.tableHeight - headerTemplate->height);

    return result;
}

std::vector<DetectedTableRow> TemplateMatcher::detectRows(const uint8_t* d_image,
                                                          int width, int height, int pitch,
                                                          int tableX, int tableY,
                                                          int tableWidth, int tableHeight) {
    std::vector<DetectedTableRow> rows;

    std::cerr << "[TemplateMatcher::detectRows] Using row template matching. tableX=" << tableX
              << " tableY=" << tableY << " tableWidth=" << tableWidth
              << " tableHeight=" << tableHeight << std::endl;

    // Collect all row templates (ROW_BACKGROUND type, named row_*)
    std::vector<const TemplateData*> rowTemplates;
    for (const auto& tmpl : m_templates) {
        if (tmpl.type == TemplateType::ROW_BACKGROUND &&
            tmpl.name.find("row_") == 0) {
            rowTemplates.push_back(&tmpl);
        }
    }

    std::cerr << "[TemplateMatcher::detectRows] Found " << rowTemplates.size() << " row templates" << std::endl;

    if (rowTemplates.empty()) {
        // Fallback: use fixed row height if no row templates
        std::cerr << "[TemplateMatcher::detectRows] No row templates, using fixed height stepping" << std::endl;
        int numRows = tableHeight / m_config.rowHeight;
        for (int i = 0; i < numRows && i < m_config.maxRows; i++) {
            DetectedTableRow row;
            row.yStart = tableY + i * m_config.rowHeight;
            row.yEnd = row.yStart + m_config.rowHeight - 1;
            row.height = m_config.rowHeight;
            row.confidence = 0.8f;
            row.hasContent = true;
            rows.push_back(row);
        }
        return rows;
    }

    // Match each row template to find actual row positions
    // Search within the table area, starting from tableY
    int searchHeight = tableHeight;
    int lastRowEnd = tableY;

    for (size_t i = 0; i < rowTemplates.size() && rows.size() < static_cast<size_t>(m_config.maxRows); i++) {
        const TemplateData* rowTmpl = rowTemplates[i];

        // Search for this row template starting from after the last found row
        int searchStartY = lastRowEnd;
        int remainingHeight = (tableY + tableHeight) - searchStartY;

        if (remainingHeight < rowTmpl->height) break;

        TemplateMatch match = matchTemplate(d_image, width, height, pitch,
                                            *rowTmpl, tableX, searchStartY,
                                            tableWidth, remainingHeight);

        std::cerr << "[TemplateMatcher::detectRows] Row " << i << " (" << rowTmpl->name
                  << "): match at y=" << match.y << " score=" << match.score
                  << " valid=" << match.valid << std::endl;

        if (match.valid && match.score >= 0.5f) {  // Lower threshold for rows
            DetectedTableRow row;
            row.yStart = match.y;
            row.yEnd = match.y + rowTmpl->height - 1;
            row.height = rowTmpl->height;
            row.confidence = match.score;
            row.hasContent = true;
            rows.push_back(row);

            lastRowEnd = row.yEnd + 1;
        }
    }

    std::cerr << "[TemplateMatcher::detectRows] Detected " << rows.size() << " rows via template matching" << std::endl;

    return rows;
}

bool TemplateMatcher::generateTemplatesFromImage(const uint8_t* imageData, int imgWidth, int imgHeight,
                                                  int roiX, int roiY, int roiWidth, int roiHeight) {
    // Extract header template (first row of the table ROI)
    int headerHeight = m_config.rowHeight + 2;  // Header is typically slightly taller

    std::vector<uint8_t> headerPixels(roiWidth * headerHeight);
    for (int y = 0; y < headerHeight; y++) {
        for (int x = 0; x < roiWidth; x++) {
            headerPixels[y * roiWidth + x] = imageData[(roiY + y) * imgWidth + (roiX + x)];
        }
    }

    if (!loadTemplate(headerPixels.data(), roiWidth, headerHeight,
                      TemplateType::TABLE_HEADER, "table_header")) {
        return false;
    }

    // Extract a sample row template
    int rowY = roiY + headerHeight + m_config.rowSpacing;
    std::vector<uint8_t> rowPixels(roiWidth * m_config.rowHeight);
    for (int y = 0; y < m_config.rowHeight; y++) {
        for (int x = 0; x < roiWidth; x++) {
            rowPixels[y * roiWidth + x] = imageData[(rowY + y) * imgWidth + (roiX + x)];
        }
    }

    if (!loadTemplate(rowPixels.data(), roiWidth, m_config.rowHeight,
                      TemplateType::ROW_BACKGROUND, "row_background")) {
        return false;
    }

    return true;
}

bool TemplateMatcher::saveTemplates(const std::string& directory) {
    for (const auto& tmpl : m_templates) {
        std::string path = directory + "/" + tmpl.name + ".tmpl";
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            m_lastError = "Failed to create template file: " + path;
            return false;
        }

        int32_t width = tmpl.width;
        int32_t height = tmpl.height;
        file.write(reinterpret_cast<const char*>(&width), sizeof(width));
        file.write(reinterpret_cast<const char*>(&height), sizeof(height));
        file.write(reinterpret_cast<const char*>(tmpl.pixels.data()), tmpl.pixels.size());
    }
    return true;
}

} // namespace trading
