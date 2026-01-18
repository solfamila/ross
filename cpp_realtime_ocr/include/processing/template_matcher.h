#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <memory>

namespace trading {

// Template match result
struct TemplateMatch {
    int x;           // Match location X
    int y;           // Match location Y
    float score;     // Match confidence (0-1 for NCC, lower is better for SSD)
    bool valid;      // Whether match is valid
};

// Detected row from template matching
struct DetectedTableRow {
    int yStart;          // Row start Y (relative to table)
    int yEnd;            // Row end Y
    int height;          // Row height
    float confidence;    // Detection confidence
    bool hasContent;     // Whether row has content
};

// Table detection result
struct TableDetection {
    int tableX;          // Table top-left X
    int tableY;          // Table top-left Y
    int tableWidth;      // Table width
    int tableHeight;     // Table height
    float confidence;    // Detection confidence
    std::vector<DetectedTableRow> rows;  // Detected rows
};

// Template type enum
enum class TemplateType {
    TABLE_HEADER,        // Table header pattern
    ROW_SEPARATOR,       // Row separator line
    ROW_BACKGROUND,      // Row background pattern
    COLUMN_HEADER        // Individual column headers
};

// Template data structure
struct TemplateData {
    std::vector<uint8_t> pixels;  // Grayscale template pixels
    int width;
    int height;
    TemplateType type;
    std::string name;
    
    // GPU resources (managed by TemplateMatcher)
    float* d_template = nullptr;
    float* d_templateSq = nullptr;  // Pre-computed squared values
    float templateSum = 0.0f;
    float templateSqSum = 0.0f;
};

// Configuration for template matcher
struct TemplateMatcherConfig {
    float matchThreshold = 0.7f;     // Minimum NCC score for valid match
    int rowHeight = 14;              // Expected row height in pixels
    int rowSpacing = 2;              // Spacing between rows
    int maxRows = 20;                // Maximum rows to detect
    bool useNCC = true;              // Use NCC (true) or SSD (false)
    int searchStepX = 1;             // X search step (1 = every pixel)
    int searchStepY = 1;             // Y search step
};

class TemplateMatcher {
public:
    TemplateMatcher();
    ~TemplateMatcher();

    // Initialize with CUDA stream
    bool initialize(cudaStream_t stream = nullptr);
    
    // Load template from file (grayscale PNG)
    bool loadTemplate(const std::string& path, TemplateType type, const std::string& name);
    
    // Load template from memory
    bool loadTemplate(const uint8_t* pixels, int width, int height, 
                      TemplateType type, const std::string& name);
    
    // Generate templates from reference image region
    bool generateTemplatesFromImage(const uint8_t* imageData, int imgWidth, int imgHeight,
                                    int roiX, int roiY, int roiWidth, int roiHeight);
    
    // Detect table in image (returns table bounds and row positions)
    TableDetection detectTable(const uint8_t* d_image, int width, int height, 
                               int pitch, int searchX, int searchY, 
                               int searchWidth, int searchHeight);
    
    // Detect rows within a known table region
    std::vector<DetectedTableRow> detectRows(const uint8_t* d_image, int width, int height,
                                             int pitch, int tableX, int tableY,
                                             int tableWidth, int tableHeight);
    
    // Single template match
    TemplateMatch matchTemplate(const uint8_t* d_image, int width, int height, int pitch,
                                const TemplateData& tmpl, int searchX, int searchY,
                                int searchWidth, int searchHeight);
    
    // Configuration
    void setConfig(const TemplateMatcherConfig& config) { m_config = config; }
    const TemplateMatcherConfig& getConfig() const { return m_config; }
    
    // Get templates
    const std::vector<TemplateData>& getTemplates() const { return m_templates; }
    
    // Save templates to files
    bool saveTemplates(const std::string& directory);
    
    // Get last error
    const std::string& getLastError() const { return m_lastError; }

private:
    // Upload template to GPU
    bool uploadTemplateToGPU(TemplateData& tmpl);
    
    // Free GPU resources
    void freeGPUResources();
    
    // Compute template statistics
    void computeTemplateStats(TemplateData& tmpl);

    std::vector<TemplateData> m_templates;
    TemplateMatcherConfig m_config;
    cudaStream_t m_stream = nullptr;
    bool m_initialized = false;
    std::string m_lastError;
    
    // GPU work buffers
    float* d_matchScores = nullptr;
    int m_scoreBufferSize = 0;
};

} // namespace trading

