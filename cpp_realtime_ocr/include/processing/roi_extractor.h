#pragma once
/**
 * @file roi_extractor.h
 * @brief ROI extraction and preprocessing pipeline
 */

#include "types.h"
#include "processing/cuda_kernels.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace trading_monitor {

/**
 * @brief ROI extraction and preprocessing manager
 * 
 * Coordinates CUDA kernels for image preprocessing pipeline.
 */
class ROIExtractor {
public:
    ROIExtractor();
    ~ROIExtractor();
    
    /**
     * @brief Initialize with ROI configuration
     * @param rois List of ROIs to extract
     * @param upscaleFactor Upscaling factor for ROIs
     * @return true if successful
     */
    bool initialize(const std::vector<ROI>& rois, float upscaleFactor = 2.0f);

    /**
     * @brief Initialize with a fixed number of ROI buffers (for dynamic ROIs)
     * @param roiCount Number of buffers to allocate
     * @param upscaleFactor Upscaling factor for ROIs
     * @return true if successful
     */
    bool initialize(size_t roiCount, float upscaleFactor = 2.0f);
    
    /**
     * @brief Extract and preprocess ROI from texture
     * @param texObj CUDA texture object from D3D11 texture
     * @param roiIndex Index of ROI to extract
     * @param stream CUDA stream
     * @return Device pointer to preprocessed data
     */
    float* extractROI(
        cudaTextureObject_t texObj,
        size_t roiIndex,
        cudaStream_t stream = 0
    );

    /**
     * @brief Extract and preprocess a specific ROI into a chosen buffer index
     * @param texObj CUDA texture object
     * @param roi ROI definition (coordinates in capture space)
     * @param bufferIndex Which internal buffer to write into
     * @param stream CUDA stream
     * @return Device pointer to preprocessed data
     */
    float* extractROI(
        cudaTextureObject_t texObj,
        const ROI& roi,
        size_t bufferIndex,
        cudaStream_t stream = 0
    );
    
    /**
     * @brief Get output dimensions for a ROI
     */
    void getOutputDimensions(size_t roiIndex, int& width, int& height) const;

    /**
     * @brief Get active (unpadded) output width for last extraction
     *
     * The extractor pads to 320px width for the model input; this returns
     * the actual content width (<= 320) used for the ROI.
     */
    int getActiveOutputWidth(size_t roiIndex) const;
    
    /**
     * @brief Get number of configured ROIs
     */
    size_t getROICount() const;

    /**
     * @brief Set pre-OCR zoom factor (>= 1.0). Values > 1 zoom into ROI center.
     */
    void setPreUpscaleFactor(float factor);
    
    /**
     * @brief Free all allocated buffers
     */
    void cleanup();
    
private:
    struct ROIBuffer {
        float* data = nullptr;
        int width = 0;
        int height = 0;
        int activeWidth = 0;
    };
    
    std::vector<ROI> m_rois;
    std::vector<ROIBuffer> m_buffers;
    float m_upscaleFactor = 2.0f;
    float m_preUpscaleFactor = 1.0f;
    
    cuda::ContrastParams m_contrastParams{1.3f, 0.0f};
};

} // namespace trading_monitor

