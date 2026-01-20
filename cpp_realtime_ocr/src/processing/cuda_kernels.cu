/**
 * @file cuda_kernels.cu
 * @brief CUDA kernel implementations for image preprocessing
 * 
 * Optimized for CUDA 13.1.1 with:
 * - Texture memory for hardware-accelerated interpolation
 * - Coalesced memory access patterns
 * - Fast math intrinsics
 */

#include "processing/cuda_kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace trading_monitor {
namespace cuda {

// Block dimensions for 2D kernels
constexpr int BLOCK_SIZE_X = 16;
constexpr int BLOCK_SIZE_Y = 16;

// Block size for 1D kernels
constexpr int BLOCK_SIZE_1D = 256;

/**
 * @brief BGRA to grayscale with ROI extraction using texture
 */
__global__ void bgraToGrayROIKernel(
    cudaTextureObject_t texObj,
    float* __restrict__ output,
    int roiX, int roiY,
    int outWidth, int outHeight,
    float invScale
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= outWidth || y >= outHeight) return;
    
    // Calculate source coordinates with bilinear interpolation
    const float srcX = roiX + x * invScale + 0.5f;
    const float srcY = roiY + y * invScale + 0.5f;
    
    // Fetch BGRA as float4 (texture returns normalized [0,1])
    float4 bgra = tex2D<float4>(texObj, srcX, srcY);
    
    // Convert BGRA to grayscale using luminosity weights
    // Note: D3D11 BGRA format: x=B, y=G, z=R, w=A
    float gray = 0.114f * bgra.x + 0.587f * bgra.y + 0.299f * bgra.z;
    
    // Write to output
    output[y * outWidth + x] = gray;
}

/**
 * @brief BGRA ROI to grayscale replicated into 3-channel NCHW with stride
 */
__global__ void bgraToNCHW3ROIKernelStrided(
    cudaTextureObject_t texObj,
    float* __restrict__ outputNCHW,
    int outputStrideW,
    int roiX, int roiY,
    int outWidth, int outHeight,
    float invScale
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outWidth || y >= outHeight) return;

    const float srcX = roiX + x * invScale + 0.5f;
    const float srcY = roiY + y * invScale + 0.5f;

    float4 bgra = tex2D<float4>(texObj, srcX, srcY);
    float gray = 0.114f * bgra.x + 0.587f * bgra.y + 0.299f * bgra.z;

    const size_t planeStride = static_cast<size_t>(outHeight) * static_cast<size_t>(outputStrideW);
    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(outputStrideW) + static_cast<size_t>(x);

    // NCHW: channel-major planes
    outputNCHW[idx] = gray;
    outputNCHW[planeStride + idx] = gray;
    outputNCHW[2 * planeStride + idx] = gray;
}

/**
 * @brief BGRA to grayscale with ROI extraction from device pointer
 */
__global__ void bgraToGrayROIDirectKernel(
    const uint8_t* __restrict__ input,
    size_t inputPitch,
    int inputWidth, int inputHeight,
    float* __restrict__ output,
    int roiX, int roiY,
    int outWidth, int outHeight,
    float invScale
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= outWidth || y >= outHeight) return;
    
    // Calculate source coordinates (nearest neighbor for direct access)
    int srcX = roiX + static_cast<int>(x * invScale);
    int srcY = roiY + static_cast<int>(y * invScale);
    
    // Clamp to input bounds
    srcX = min(max(srcX, 0), inputWidth - 1);
    srcY = min(max(srcY, 0), inputHeight - 1);
    
    // Read BGRA pixel
    const uint8_t* row = input + srcY * inputPitch;
    const uint8_t* pixel = row + srcX * 4;
    
    // Convert to grayscale
    float b = pixel[0] / 255.0f;
    float g = pixel[1] / 255.0f;
    float r = pixel[2] / 255.0f;
    
    float gray = 0.114f * b + 0.587f * g + 0.299f * r;
    output[y * outWidth + x] = gray;
}

/**
 * @brief BGRA ROI to grayscale uint8 (direct device pointer)
 */
__global__ void bgraToGrayU8ROIDirectKernel(
    const uint8_t* __restrict__ input,
    size_t inputPitch,
    int inputWidth, int inputHeight,
    uint8_t* __restrict__ output,
    size_t outputPitch,
    int roiX, int roiY,
    int roiWidth, int roiHeight
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= roiWidth || y >= roiHeight) return;

    int srcX = roiX + x;
    int srcY = roiY + y;

    if (srcX < 0 || srcX >= inputWidth || srcY < 0 || srcY >= inputHeight) return;

    const uint8_t* row = input + static_cast<size_t>(srcY) * inputPitch;
    const uint8_t* pixel = row + srcX * 4;

    float b = pixel[0];
    float g = pixel[1];
    float r = pixel[2];

    float gray = 0.114f * b + 0.587f * g + 0.299f * r;
    uint8_t val = static_cast<uint8_t>(fminf(fmaxf(gray, 0.0f), 255.0f) + 0.5f);

    output[static_cast<size_t>(y) * outputPitch + x] = val;
}

/**
 * @brief BGRA texture ROI to grayscale uint8
 */
__global__ void bgraTexToGrayU8ROIKernel(
    cudaTextureObject_t texObj,
    uint8_t* __restrict__ output,
    size_t outputPitch,
    int roiX, int roiY,
    int roiWidth, int roiHeight
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= roiWidth || y >= roiHeight) return;

    const float srcX = static_cast<float>(roiX + x) + 0.5f;
    const float srcY = static_cast<float>(roiY + y) + 0.5f;

    float4 bgra = tex2D<float4>(texObj, srcX, srcY);
    float gray = 0.114f * bgra.x + 0.587f * bgra.y + 0.299f * bgra.z;
    float scaled = gray * 255.0f;

    uint8_t val = static_cast<uint8_t>(fminf(fmaxf(scaled, 0.0f), 255.0f) + 0.5f);
    output[static_cast<size_t>(y) * outputPitch + x] = val;
}

/**
 * @brief Luma ROI diff row sums + update prev ROI
 */
__global__ void lumaDiffRowSumsUpdateKernel(
    const uint8_t* __restrict__ input,
    size_t inputPitch,
    int roiX, int roiY,
    int roiWidth, int roiHeight,
    int bandX0, int bandX1,
    uint8_t* __restrict__ prev,
    unsigned int* __restrict__ rowSums
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= roiWidth || y >= roiHeight) return;

    const int srcX = roiX + x;
    const int srcY = roiY + y;
    const uint8_t* row = input + static_cast<size_t>(srcY) * inputPitch;
    const uint8_t cur = row[srcX];

    const size_t prevIdx = static_cast<size_t>(y) * static_cast<size_t>(roiWidth) + static_cast<size_t>(x);
    const uint8_t old = prev[prevIdx];
    prev[prevIdx] = cur;

    if (x >= bandX0 && x < bandX1) {
        const unsigned int diff = static_cast<unsigned int>(cur > old ? cur - old : old - cur);
        atomicAdd(&rowSums[y], diff);
    }
}

/**
 * @brief Contrast enhancement kernel
 */
__global__ void contrastEnhanceKernel(
    float* __restrict__ data,
    int width, int height,
    float alpha, float beta
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = width * height;
    
    if (idx >= total) return;
    
    float val = data[idx];
    
    // Apply contrast: alpha * (val - 0.5) + 0.5 + beta
    val = __fmaf_rn(alpha, val - 0.5f, 0.5f + beta);
    
    // Clamp to [0, 1]
    data[idx] = fminf(fmaxf(val, 0.0f), 1.0f);
}

/**
 * @brief Normalization kernel
 */
__global__ void normalizeKernel(
    float* __restrict__ data,
    int width, int height,
    float mean, float invStd
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = width * height;
    
    if (idx >= total) return;
    
    data[idx] = (data[idx] - mean) * invStd;
}

/**
 * @brief Perceptual hash computation kernel
 * 
 * Downsamples to 8x8 and computes hash based on mean comparison
 */
__global__ void computeHashKernel(
    const float* __restrict__ input,
    int width, int height,
    uint64_t* hashOutput
) {
    __shared__ float block[64];  // 8x8 downsampled image
    __shared__ float mean;
    
    const int tid = threadIdx.x;
    
    if (tid < 64) {
        // Calculate block position for this thread
        int bx = tid % 8;
        int by = tid / 8;
        
        // Sample from corresponding region
        int srcX = (bx * width) / 8 + width / 16;
        int srcY = (by * height) / 8 + height / 16;
        
        srcX = min(max(srcX, 0), width - 1);
        srcY = min(max(srcY, 0), height - 1);
        
        block[tid] = input[srcY * width + srcX];
    }
    
    __syncthreads();
    
    // Compute mean (simple reduction)
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 64; i++) {
            sum += block[i];
        }
        mean = sum / 64.0f;
    }
    
    __syncthreads();

    // Compute hash bits
    if (tid < 64) {
        uint64_t bit = (block[tid] >= mean) ? (1ULL << tid) : 0ULL;
        atomicOr(reinterpret_cast<unsigned long long*>(hashOutput), bit);
    }
}

/**
 * @brief Perceptual hash for strided grayscale images
 */
__global__ void computeHashKernelStrided(
    const float* __restrict__ input,
    int inputStride,
    int width,
    int height,
    uint64_t* hashOutput
) {
    __shared__ float block[64];
    __shared__ float mean;

    const int tid = threadIdx.x;

    if (tid < 64) {
        int bx = tid % 8;
        int by = tid / 8;

        int srcX = (bx * width) / 8 + width / 16;
        int srcY = (by * height) / 8 + height / 16;

        srcX = min(max(srcX, 0), width - 1);
        srcY = min(max(srcY, 0), height - 1);

        block[tid] = input[srcY * inputStride + srcX];
    }

    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 64; i++) {
            sum += block[i];
        }
        mean = sum / 64.0f;
    }

    __syncthreads();

    if (tid < 64) {
        uint64_t bit = (block[tid] >= mean) ? (1ULL << tid) : 0ULL;
        atomicOr(reinterpret_cast<unsigned long long*>(hashOutput), bit);
    }
}

// =============================================================================
// Launcher Functions
// =============================================================================

cudaError_t launchBgraToGrayROI(
    cudaTextureObject_t texObj,
    float* output,
    const ROIParams& params,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(
        (params.outWidth + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (params.outHeight + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y
    );

    float invScale = 1.0f / params.scale;

    bgraToGrayROIKernel<<<grid, block, 0, stream>>>(
        texObj, output,
        params.roiX, params.roiY,
        params.outWidth, params.outHeight,
        invScale
    );

    return cudaGetLastError();
}

cudaError_t launchBgraToNCHW3ROIStrided(
    cudaTextureObject_t texObj,
    float* outputNCHW,
    int outputStrideW,
    const ROIParams& params,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(
        (params.outWidth + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (params.outHeight + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y
    );

    float invScale = 1.0f / params.scale;

    bgraToNCHW3ROIKernelStrided<<<grid, block, 0, stream>>>(
        texObj,
        outputNCHW,
        outputStrideW,
        params.roiX,
        params.roiY,
        params.outWidth,
        params.outHeight,
        invScale
    );

    return cudaGetLastError();
}

cudaError_t launchBgraToGrayROIDirect(
    const uint8_t* input,
    size_t inputPitch,
    int inputWidth,
    int inputHeight,
    float* output,
    const ROIParams& params,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(
        (params.outWidth + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (params.outHeight + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y
    );

    float invScale = 1.0f / params.scale;

    bgraToGrayROIDirectKernel<<<grid, block, 0, stream>>>(
        input, inputPitch,
        inputWidth, inputHeight,
        output,
        params.roiX, params.roiY,
        params.outWidth, params.outHeight,
        invScale
    );

    return cudaGetLastError();
}

cudaError_t launchContrastEnhance(
    float* data,
    int width,
    int height,
    const ContrastParams& params,
    cudaStream_t stream
) {
    int total = width * height;
    int blocks = (total + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    contrastEnhanceKernel<<<blocks, BLOCK_SIZE_1D, 0, stream>>>(
        data, width, height, params.alpha, params.beta
    );

    return cudaGetLastError();
}

cudaError_t launchNormalize(
    float* data,
    int width,
    int height,
    float mean,
    float std,
    cudaStream_t stream
) {
    int total = width * height;
    int blocks = (total + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
    float invStd = 1.0f / std;

    normalizeKernel<<<blocks, BLOCK_SIZE_1D, 0, stream>>>(
        data, width, height, mean, invStd
    );

    return cudaGetLastError();
}

cudaError_t launchBgraToGrayU8ROIDirect(
    const uint8_t* input,
    size_t inputPitch,
    int inputWidth,
    int inputHeight,
    uint8_t* output,
    size_t outputPitch,
    int roiX,
    int roiY,
    int roiWidth,
    int roiHeight,
    cudaStream_t stream
) {
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize(
        (roiWidth + blockSize.x - 1) / blockSize.x,
        (roiHeight + blockSize.y - 1) / blockSize.y
    );

    bgraToGrayU8ROIDirectKernel<<<gridSize, blockSize, 0, stream>>>(
        input, inputPitch, inputWidth, inputHeight,
        output, outputPitch, roiX, roiY, roiWidth, roiHeight
    );

    return cudaGetLastError();
}

cudaError_t launchBgraTexToGrayU8ROI(
    cudaTextureObject_t texObj,
    uint8_t* output,
    size_t outputPitch,
    int roiX,
    int roiY,
    int roiWidth,
    int roiHeight,
    cudaStream_t stream
) {
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize(
        (roiWidth + blockSize.x - 1) / blockSize.x,
        (roiHeight + blockSize.y - 1) / blockSize.y
    );

    bgraTexToGrayU8ROIKernel<<<gridSize, blockSize, 0, stream>>>(
        texObj, output, outputPitch, roiX, roiY, roiWidth, roiHeight
    );

    return cudaGetLastError();
}

cudaError_t launchComputeHash(
    const float* input,
    int width,
    int height,
    uint64_t* hashOutput,
    cudaStream_t stream
) {
    // Initialize hash to 0
    cudaMemsetAsync(hashOutput, 0, sizeof(uint64_t), stream);

    // Launch with single block of 64 threads
    computeHashKernel<<<1, 64, 0, stream>>>(
        input, width, height, hashOutput
    );

    return cudaGetLastError();
}

cudaError_t launchComputeHashStrided(
    const float* input,
    int inputStride,
    int width,
    int height,
    uint64_t* hashOutput,
    cudaStream_t stream
) {
    cudaMemsetAsync(hashOutput, 0, sizeof(uint64_t), stream);
    computeHashKernelStrided<<<1, 64, 0, stream>>>(
        input,
        inputStride,
        width,
        height,
        hashOutput
    );
    return cudaGetLastError();
}

/**
 * @brief YOLO preprocessing kernel from texture
 * Resize, BGRA->RGB, normalize to [0,1], output NCHW
 */
__global__ void yoloPreprocessKernel(
    cudaTextureObject_t texObj,
    float* __restrict__ output,
    int frameWidth, int frameHeight,
    int outWidth, int outHeight
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outWidth || y >= outHeight) return;

    // Calculate source coordinates (letterbox scaling)
    float scale = fminf(static_cast<float>(outWidth) / frameWidth,
                        static_cast<float>(outHeight) / frameHeight);
    int newW = static_cast<int>(frameWidth * scale);
    int newH = static_cast<int>(frameHeight * scale);
    int offsetX = (outWidth - newW) / 2;
    int offsetY = (outHeight - newH) / 2;

    float4 bgra;
    if (x >= offsetX && x < offsetX + newW && y >= offsetY && y < offsetY + newH) {
        // Map to source coordinates
        float srcX = static_cast<float>(x - offsetX) / scale + 0.5f;
        float srcY = static_cast<float>(y - offsetY) / scale + 0.5f;
        bgra = tex2D<float4>(texObj, srcX, srcY);
    } else {
        // Letterbox padding (gray 114/255 is YOLO standard)
        bgra = make_float4(0.447f, 0.447f, 0.447f, 1.0f);
    }

    // NCHW layout: [1, 3, H, W]
    const size_t planeSize = static_cast<size_t>(outWidth) * outHeight;
    const size_t idx = static_cast<size_t>(y) * outWidth + x;

    // BGRA -> RGB (z=R, y=G, x=B in D3D11 BGRA)
    output[idx]                 = bgra.z;  // R channel
    output[idx + planeSize]     = bgra.y;  // G channel
    output[idx + planeSize * 2] = bgra.x;  // B channel
}

/**
 * @brief YOLO preprocessing kernel from BGRA pointer
 */
__global__ void yoloPreprocessFromBGRAKernel(
    const uint8_t* __restrict__ input,
    int frameWidth, int frameHeight,
    float* __restrict__ output,
    int outWidth, int outHeight
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outWidth || y >= outHeight) return;

    // Letterbox scaling
    float scale = fminf(static_cast<float>(outWidth) / frameWidth,
                        static_cast<float>(outHeight) / frameHeight);
    int newW = static_cast<int>(frameWidth * scale);
    int newH = static_cast<int>(frameHeight * scale);
    int offsetX = (outWidth - newW) / 2;
    int offsetY = (outHeight - newH) / 2;

    float r, g, b;
    if (x >= offsetX && x < offsetX + newW && y >= offsetY && y < offsetY + newH) {
        int srcX = static_cast<int>((x - offsetX) / scale);
        int srcY = static_cast<int>((y - offsetY) / scale);
        srcX = min(srcX, frameWidth - 1);
        srcY = min(srcY, frameHeight - 1);

        size_t srcIdx = (static_cast<size_t>(srcY) * frameWidth + srcX) * 4;
        b = input[srcIdx + 0] / 255.0f;
        g = input[srcIdx + 1] / 255.0f;
        r = input[srcIdx + 2] / 255.0f;
    } else {
        r = g = b = 0.447f; // Letterbox padding
    }

    const size_t planeSize = static_cast<size_t>(outWidth) * outHeight;
    const size_t idx = static_cast<size_t>(y) * outWidth + x;

    output[idx]                 = r;
    output[idx + planeSize]     = g;
    output[idx + planeSize * 2] = b;
}

cudaError_t launchYOLOPreprocess(
    cudaTextureObject_t texObj,
    int frameWidth, int frameHeight,
    float* output,
    int outWidth, int outHeight,
    cudaStream_t stream
) {
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize(
        (outWidth + blockSize.x - 1) / blockSize.x,
        (outHeight + blockSize.y - 1) / blockSize.y
    );

    yoloPreprocessKernel<<<gridSize, blockSize, 0, stream>>>(
        texObj, output, frameWidth, frameHeight, outWidth, outHeight
    );

    return cudaGetLastError();
}

cudaError_t launchYOLOPreprocessFromBGRA(
    const uint8_t* input,
    int frameWidth, int frameHeight,
    float* output,
    int outWidth, int outHeight,
    cudaStream_t stream
) {
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize(
        (outWidth + blockSize.x - 1) / blockSize.x,
        (outHeight + blockSize.y - 1) / blockSize.y
    );

    yoloPreprocessFromBGRAKernel<<<gridSize, blockSize, 0, stream>>>(
        input, frameWidth, frameHeight, output, outWidth, outHeight
    );

    return cudaGetLastError();
}

cudaError_t launchLumaDiffRowSumsUpdate(
    const uint8_t* input,
    size_t inputPitch,
    int roiX,
    int roiY,
    int roiWidth,
    int roiHeight,
    int bandX0,
    int bandX1,
    uint8_t* prev,
    unsigned int* rowSums,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(
        (roiWidth + block.x - 1) / block.x,
        (roiHeight + block.y - 1) / block.y
    );

    lumaDiffRowSumsUpdateKernel<<<grid, block, 0, stream>>>(
        input,
        inputPitch,
        roiX,
        roiY,
        roiWidth,
        roiHeight,
        bandX0,
        bandX1,
        prev,
        rowSums
    );

    return cudaGetLastError();
}

} // namespace cuda
} // namespace trading_monitor
