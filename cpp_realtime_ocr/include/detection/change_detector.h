#pragma once
/**
 * @file change_detector.h
 * @brief Frame change detection using perceptual hashing
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace trading_monitor {

/**
 * @brief Change detector using perceptual hashing
 * 
 * Skips OCR processing when frame content hasn't changed.
 */
class ChangeDetector {
public:
    ChangeDetector();
    ~ChangeDetector();
    
    /**
     * @brief Initialize for given number of ROIs
     * @param numROIs Number of ROIs to track
     * @param threshold Hamming distance threshold for "changed" (default: 5)
     */
    void initialize(size_t numROIs, int threshold = 5);
    
    /**
     * @brief Check if ROI content has changed
     * @param roiIndex Index of ROI
     * @param imageData Device pointer to grayscale image
     * @param width Image width
     * @param height Image height
     * @param stream CUDA stream
     * @return true if content has changed
     */
    bool hasChanged(
        size_t roiIndex,
        const float* imageData,
        int width,
        int height,
        cudaStream_t stream = 0
    );

    /**
     * @brief Check if ROI content has changed for strided images
     *
     * @param inputStride Row stride in pixels (not bytes)
     */
    bool hasChangedStrided(
        size_t roiIndex,
        const float* imageData,
        int inputStride,
        int width,
        int height,
        cudaStream_t stream = 0
    );
    
    /**
     * @brief Force next check to report as changed
     * @param roiIndex Index of ROI
     */
    void invalidate(size_t roiIndex);
    
    /**
     * @brief Invalidate all ROIs
     */
    void invalidateAll();
    
    /**
     * @brief Get current hash for a ROI
     */
    uint64_t getCurrentHash(size_t roiIndex) const;

    /**
     * @brief Get number of ROI slots
     */
    size_t getSlotCount() const { return m_previousHashes.size(); }

private:
    std::vector<uint64_t> m_previousHashes;
    std::vector<bool> m_valid;
    uint64_t* m_deviceHash = nullptr;
    int m_threshold = 5;
    
    int hammingDistance(uint64_t a, uint64_t b) const;
};

} // namespace trading_monitor

