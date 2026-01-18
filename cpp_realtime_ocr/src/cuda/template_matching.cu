/**
 * CUDA Template Matching Kernels
 * Implements fast NCC and SSD template matching for table/row detection
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace trading {
namespace cuda {

// Shared memory tile size for template caching
constexpr int TILE_SIZE = 16;
constexpr int MAX_TEMPLATE_SIZE = 64;  // Max template dimension for shared memory

/**
 * Kernel: Compute sum and sum of squares for image regions
 * Used for NCC normalization
 */
__global__ void computeRegionStats(
    const unsigned char* __restrict__ image,
    int imgWidth, int imgHeight, int pitch,
    int templateWidth, int templateHeight,
    int searchX, int searchY, int searchWidth, int searchHeight,
    float* __restrict__ regionSums,
    float* __restrict__ regionSqSums
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x + searchX;
    int ty = blockIdx.y * blockDim.y + threadIdx.y + searchY;

    if (tx >= searchX + searchWidth || ty >= searchY + searchHeight) return;
    if (tx + templateWidth > imgWidth || ty + templateHeight > imgHeight) return;

    float sum = 0.0f;
    float sqSum = 0.0f;

    for (int dy = 0; dy < templateHeight; dy++) {
        const unsigned char* row = image + (ty + dy) * pitch + tx;
        for (int dx = 0; dx < templateWidth; dx++) {
            float val = static_cast<float>(row[dx]);
            sum += val;
            sqSum += val * val;
        }
    }

    int outIdx = (ty - searchY) * searchWidth + (tx - searchX);
    regionSums[outIdx] = sum;
    regionSqSums[outIdx] = sqSum;
}

/**
 * Kernel: Normalized Cross-Correlation (NCC) template matching
 * Uses pre-computed template statistics for speed
 */
__global__ void nccTemplateMatch(
    const unsigned char* __restrict__ image,
    int imgWidth, int imgHeight, int pitch,
    const float* __restrict__ templateData,
    int templateWidth, int templateHeight,
    float templateSum, float templateSqSum,
    int searchX, int searchY, int searchWidth, int searchHeight,
    float* __restrict__ scores
) {
    // Shared memory for template (if small enough)
    __shared__ float s_template[MAX_TEMPLATE_SIZE * MAX_TEMPLATE_SIZE];

    int tx = blockIdx.x * blockDim.x + threadIdx.x + searchX;
    int ty = blockIdx.y * blockDim.y + threadIdx.y + searchY;

    // Load template into shared memory (first threads only)
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int templateSize = templateWidth * templateHeight;
    if (templateSize <= MAX_TEMPLATE_SIZE * MAX_TEMPLATE_SIZE) {
        for (int i = tid; i < templateSize; i += blockDim.x * blockDim.y) {
            s_template[i] = templateData[i];
        }
    }
    __syncthreads();

    if (tx >= searchX + searchWidth || ty >= searchY + searchHeight) return;
    if (tx + templateWidth > imgWidth || ty + templateHeight > imgHeight) return;

    // Compute cross-correlation and image region stats
    float crossCorr = 0.0f;
    float imgSum = 0.0f;
    float imgSqSum = 0.0f;

    const float* tmpl = (templateSize <= MAX_TEMPLATE_SIZE * MAX_TEMPLATE_SIZE)
                        ? s_template : templateData;

    for (int dy = 0; dy < templateHeight; dy++) {
        const unsigned char* imgRow = image + (ty + dy) * pitch + tx;
        const float* tmplRow = tmpl + dy * templateWidth;

        for (int dx = 0; dx < templateWidth; dx++) {
            float imgVal = static_cast<float>(imgRow[dx]);
            float tmplVal = tmplRow[dx];

            crossCorr += imgVal * tmplVal;
            imgSum += imgVal;
            imgSqSum += imgVal * imgVal;
        }
    }

    // Compute NCC
    float n = static_cast<float>(templateSize);
    float imgMean = imgSum / n;
    float tmplMean = templateSum / n;

    float imgVar = imgSqSum - imgSum * imgMean;
    float tmplVar = templateSqSum - templateSum * tmplMean;

    float denom = sqrtf(imgVar * tmplVar);
    float ncc = 0.0f;

    if (denom > 1e-6f) {
        float numerator = crossCorr - imgSum * tmplMean - templateSum * imgMean + n * imgMean * tmplMean;
        ncc = numerator / denom;
    }

    int outIdx = (ty - searchY) * searchWidth + (tx - searchX);
    scores[outIdx] = ncc;
}

/**
 * Kernel: Sum of Squared Differences (SSD) template matching
 * Faster than NCC but less robust to lighting changes
 */
__global__ void ssdTemplateMatch(
    const unsigned char* __restrict__ image,
    int imgWidth, int imgHeight, int pitch,
    const float* __restrict__ templateData,
    int templateWidth, int templateHeight,
    int searchX, int searchY, int searchWidth, int searchHeight,
    float* __restrict__ scores
) {
    __shared__ float s_template[MAX_TEMPLATE_SIZE * MAX_TEMPLATE_SIZE];

    int tx = blockIdx.x * blockDim.x + threadIdx.x + searchX;
    int ty = blockIdx.y * blockDim.y + threadIdx.y + searchY;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int templateSize = templateWidth * templateHeight;
    if (templateSize <= MAX_TEMPLATE_SIZE * MAX_TEMPLATE_SIZE) {
        for (int i = tid; i < templateSize; i += blockDim.x * blockDim.y) {
            s_template[i] = templateData[i];
        }
    }
    __syncthreads();

    if (tx >= searchX + searchWidth || ty >= searchY + searchHeight) return;
    if (tx + templateWidth > imgWidth || ty + templateHeight > imgHeight) return;

    float ssd = 0.0f;
    const float* tmpl = (templateSize <= MAX_TEMPLATE_SIZE * MAX_TEMPLATE_SIZE)
                        ? s_template : templateData;

    for (int dy = 0; dy < templateHeight; dy++) {
        const unsigned char* imgRow = image + (ty + dy) * pitch + tx;
        const float* tmplRow = tmpl + dy * templateWidth;

        for (int dx = 0; dx < templateWidth; dx++) {
            float diff = static_cast<float>(imgRow[dx]) - tmplRow[dx];
            ssd += diff * diff;
        }
    }

    // Normalize SSD to 0-1 range (inverted so higher is better match)
    float maxSSD = 255.0f * 255.0f * static_cast<float>(templateSize);
    float normalizedScore = 1.0f - (ssd / maxSSD);

    int outIdx = (ty - searchY) * searchWidth + (tx - searchX);
    scores[outIdx] = normalizedScore;
}

/**
 * Kernel: Find maximum score location (reduction)
 */
__global__ void findMaxScore(
    const float* __restrict__ scores,
    int width, int height,
    float* __restrict__ maxScore,
    int* __restrict__ maxX,
    int* __restrict__ maxY
) {
    __shared__ float s_maxScore[256];
    __shared__ int s_maxIdx[256];

    int tid = threadIdx.x;
    int totalSize = width * height;

    // Each thread finds max in its portion
    float localMax = -1e10f;
    int localIdx = 0;

    for (int i = tid; i < totalSize; i += blockDim.x) {
        if (scores[i] > localMax) {
            localMax = scores[i];
            localIdx = i;
        }
    }

    s_maxScore[tid] = localMax;
    s_maxIdx[tid] = localIdx;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_maxScore[tid + stride] > s_maxScore[tid]) {
                s_maxScore[tid] = s_maxScore[tid + stride];
                s_maxIdx[tid] = s_maxIdx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *maxScore = s_maxScore[0];
        *maxX = s_maxIdx[0] % width;
        *maxY = s_maxIdx[0] / width;
    }
}

/**
 * Kernel: Detect horizontal lines (row separators)
 * Scans for consistent horizontal patterns
 */
__global__ void detectHorizontalLines(
    const unsigned char* __restrict__ image,
    int imgWidth, int imgHeight, int pitch,
    int searchX, int searchY, int searchWidth, int searchHeight,
    int minLineLength,
    unsigned char targetIntensity,
    unsigned char tolerance,
    int* __restrict__ lineYPositions,
    float* __restrict__ lineScores,
    int* __restrict__ lineCount,
    int maxLines
) {
    int ty = blockIdx.x * blockDim.x + threadIdx.x + searchY;

    if (ty >= searchY + searchHeight) return;

    // Count pixels matching target intensity in this row
    int matchCount = 0;
    const unsigned char* row = image + ty * pitch + searchX;

    for (int x = 0; x < searchWidth; x++) {
        int diff = abs(static_cast<int>(row[x]) - static_cast<int>(targetIntensity));
        if (diff <= tolerance) {
            matchCount++;
        }
    }

    float matchRatio = static_cast<float>(matchCount) / static_cast<float>(searchWidth);

    // If sufficient match, record as potential line
    if (matchRatio > 0.8f && matchCount >= minLineLength) {
        int idx = atomicAdd(lineCount, 1);
        if (idx < maxLines) {
            lineYPositions[idx] = ty;
            lineScores[idx] = matchRatio;
        }
    }
}

/**
 * Kernel: Detect row boundaries based on intensity transitions
 */
__global__ void detectRowBoundaries(
    const unsigned char* __restrict__ image,
    int imgWidth, int imgHeight, int pitch,
    int tableX, int tableY, int tableWidth, int tableHeight,
    int expectedRowHeight,
    int* __restrict__ rowStarts,
    int* __restrict__ rowEnds,
    float* __restrict__ rowConfidences,
    int* __restrict__ rowCount,
    int maxRows
) {
    // Single thread scans for row transitions
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Alternative approach: use fixed row height stepping
    // Most trading tables have consistent row heights
    int numRows = 0;

    // Calculate how many rows can fit
    int possibleRows = tableHeight / expectedRowHeight;
    if (possibleRows > maxRows) possibleRows = maxRows;

    // Step through rows at expected intervals
    for (int i = 0; i < possibleRows && numRows < maxRows; i++) {
        int rowStart = tableY + i * expectedRowHeight;
        int rowEnd = rowStart + expectedRowHeight - 1;

        // Ensure we don't go past the table
        if (rowEnd >= tableY + tableHeight) break;

        // Check if this row has content (not empty)
        float avgIntensity = 0.0f;
        for (int y = rowStart; y <= rowEnd; y++) {
            const unsigned char* rowPtr = image + y * pitch + tableX;
            for (int x = 0; x < tableWidth; x++) {
                avgIntensity += static_cast<float>(rowPtr[x]);
            }
        }
        avgIntensity /= static_cast<float>((rowEnd - rowStart + 1) * tableWidth);

        // Row is valid if it has some content (not all black or all white)
        bool hasContent = avgIntensity > 20.0f && avgIntensity < 240.0f;

        if (hasContent) {
            rowStarts[numRows] = rowStart;
            rowEnds[numRows] = rowEnd;
            rowConfidences[numRows] = 0.9f;  // Fixed confidence for consistent spacing
            numRows++;
        }
    }

    *rowCount = numRows;
}


// ============================================================================
// Host wrapper functions
// ============================================================================

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
) {
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (searchWidth + blockSize.x - 1) / blockSize.x,
        (searchHeight + blockSize.y - 1) / blockSize.y
    );

    nccTemplateMatch<<<gridSize, blockSize, 0, stream>>>(
        d_image, imgWidth, imgHeight, pitch,
        d_template, templateWidth, templateHeight,
        templateSum, templateSqSum,
        searchX, searchY, searchWidth, searchHeight,
        d_scores
    );

    return cudaGetLastError();
}

cudaError_t launchSSDTemplateMatch(
    const unsigned char* d_image,
    int imgWidth, int imgHeight, int pitch,
    const float* d_template,
    int templateWidth, int templateHeight,
    int searchX, int searchY, int searchWidth, int searchHeight,
    float* d_scores,
    cudaStream_t stream
) {
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (searchWidth + blockSize.x - 1) / blockSize.x,
        (searchHeight + blockSize.y - 1) / blockSize.y
    );

    ssdTemplateMatch<<<gridSize, blockSize, 0, stream>>>(
        d_image, imgWidth, imgHeight, pitch,
        d_template, templateWidth, templateHeight,
        searchX, searchY, searchWidth, searchHeight,
        d_scores
    );

    return cudaGetLastError();
}

cudaError_t launchFindMaxScore(
    const float* d_scores,
    int width, int height,
    float* d_maxScore,
    int* d_maxX,
    int* d_maxY,
    cudaStream_t stream
) {
    findMaxScore<<<1, 256, 0, stream>>>(
        d_scores, width, height,
        d_maxScore, d_maxX, d_maxY
    );

    return cudaGetLastError();
}

cudaError_t launchDetectHorizontalLines(
    const unsigned char* d_image,
    int imgWidth, int imgHeight, int pitch,
    int searchX, int searchY, int searchWidth, int searchHeight,
    int minLineLength,
    unsigned char targetIntensity,
    unsigned char tolerance,
    int* d_lineYPositions,
    float* d_lineScores,
    int* d_lineCount,
    int maxLines,
    cudaStream_t stream
) {
    int blockSize = 256;
    int gridSize = (searchHeight + blockSize - 1) / blockSize;

    // Reset line count
    cudaMemsetAsync(d_lineCount, 0, sizeof(int), stream);

    detectHorizontalLines<<<gridSize, blockSize, 0, stream>>>(
        d_image, imgWidth, imgHeight, pitch,
        searchX, searchY, searchWidth, searchHeight,
        minLineLength, targetIntensity, tolerance,
        d_lineYPositions, d_lineScores, d_lineCount, maxLines
    );

    return cudaGetLastError();
}

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
) {
    // Reset row count
    cudaMemsetAsync(d_rowCount, 0, sizeof(int), stream);

    detectRowBoundaries<<<1, 1, 0, stream>>>(
        d_image, imgWidth, imgHeight, pitch,
        tableX, tableY, tableWidth, tableHeight,
        expectedRowHeight,
        d_rowStarts, d_rowEnds, d_rowConfidences, d_rowCount, maxRows
    );

    return cudaGetLastError();
}

} // extern "C"

} // namespace cuda
} // namespace trading

