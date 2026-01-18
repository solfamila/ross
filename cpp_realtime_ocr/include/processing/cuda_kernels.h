#pragma once
/**
 * @file cuda_kernels.h
 * @brief CUDA kernel declarations for image preprocessing
 * 
 * Provides GPU-accelerated preprocessing for OCR:
 * - BGRA to grayscale conversion with ROI extraction
 * - Contrast enhancement
 * - Normalization for TensorRT input
 * 
 * Compatible with CUDA 13.1.1
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace trading_monitor {
namespace cuda {

/**
 * @brief ROI extraction parameters
 */
struct ROIParams {
    int roiX;           ///< ROI top-left X coordinate
    int roiY;           ///< ROI top-left Y coordinate
    int roiWidth;       ///< ROI width in source pixels
    int roiHeight;      ///< ROI height in source pixels
    int outWidth;       ///< Output width (after scaling)
    int outHeight;      ///< Output height (after scaling)
    float scale;        ///< Upscaling factor (e.g., 2.0)
};

/**
 * @brief Contrast enhancement parameters
 */
struct ContrastParams {
    float alpha;        ///< Contrast multiplier (1.0 = no change, >1.0 = more contrast)
    float beta;         ///< Brightness offset (-1.0 to 1.0)
};

/**
 * @brief Extract ROI from BGRA texture and convert to grayscale
 * 
 * @param texObj     CUDA texture object bound to D3D11 texture
 * @param output     Output buffer (pre-allocated, size: outWidth * outHeight)
 * @param params     ROI and scaling parameters
 * @param stream     CUDA stream for async execution
 * @return cudaError_t 
 */
cudaError_t launchBgraToGrayROI(
    cudaTextureObject_t texObj,
    float* output,
    const ROIParams& params,
    cudaStream_t stream = 0
);

/**
 * @brief Extract ROI from BGRA texture and write to 3-channel NCHW buffer
 *
 * Writes grayscale replicated into 3 channels (RGB) in NCHW layout.
 * Supports a larger destination stride for padding (e.g. pad to width=320).
 *
 * @param texObj        CUDA texture object bound to D3D11 texture
 * @param outputNCHW    Output buffer in NCHW layout
 * @param outputStrideW Destination row stride in pixels (>= params.outWidth)
 * @param params        ROI and scaling parameters (params.outWidth is the active width)
 * @param stream        CUDA stream for async execution
 */
cudaError_t launchBgraToNCHW3ROIStrided(
    cudaTextureObject_t texObj,
    float* outputNCHW,
    int outputStrideW,
    const ROIParams& params,
    cudaStream_t stream = 0
);

/**
 * @brief Extract ROI from BGRA device pointer and convert to grayscale
 * 
 * Alternative when texture binding is not desired.
 * 
 * @param input      Input BGRA buffer (device memory)
 * @param inputPitch Pitch of input buffer in bytes
 * @param inputWidth Full input width
 * @param inputHeight Full input height
 * @param output     Output grayscale buffer
 * @param params     ROI and scaling parameters
 * @param stream     CUDA stream
 * @return cudaError_t 
 */
cudaError_t launchBgraToGrayROIDirect(
    const uint8_t* input,
    size_t inputPitch,
    int inputWidth,
    int inputHeight,
    float* output,
    const ROIParams& params,
    cudaStream_t stream = 0
);

/**
 * @brief Extract ROI from BGRA device pointer and convert to grayscale uint8
 *
 * @param input      Input BGRA buffer (device memory)
 * @param inputPitch Pitch of input buffer in bytes
 * @param inputWidth Full input width
 * @param inputHeight Full input height
 * @param output     Output grayscale buffer (uint8)
 * @param outputPitch Output pitch in bytes
 * @param roiX       ROI top-left X coordinate
 * @param roiY       ROI top-left Y coordinate
 * @param roiWidth   ROI width
 * @param roiHeight  ROI height
 * @param stream     CUDA stream
 * @return cudaError_t
 */
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
    cudaStream_t stream = 0
);

/**
 * @brief Extract ROI from BGRA texture and convert to grayscale uint8
 *
 * @param texObj      CUDA texture object bound to BGRA image
 * @param output      Output grayscale buffer (uint8)
 * @param outputPitch Output pitch in bytes
 * @param roiX        ROI top-left X coordinate
 * @param roiY        ROI top-left Y coordinate
 * @param roiWidth    ROI width
 * @param roiHeight   ROI height
 * @param stream      CUDA stream
 * @return cudaError_t
 */
cudaError_t launchBgraTexToGrayU8ROI(
    cudaTextureObject_t texObj,
    uint8_t* output,
    size_t outputPitch,
    int roiX,
    int roiY,
    int roiWidth,
    int roiHeight,
    cudaStream_t stream = 0
);

/**
 * @brief Apply contrast enhancement to grayscale image
 * 
 * Formula: output = clamp(alpha * (input - 0.5) + 0.5 + beta, 0, 1)
 * 
 * @param data       Input/output buffer (in-place operation)
 * @param width      Image width
 * @param height     Image height
 * @param params     Contrast parameters
 * @param stream     CUDA stream
 * @return cudaError_t 
 */
cudaError_t launchContrastEnhance(
    float* data,
    int width,
    int height,
    const ContrastParams& params,
    cudaStream_t stream = 0
);

/**
 * @brief Normalize image for TensorRT input
 * 
 * Applies channel-wise normalization: (x - mean) / std
 * 
 * @param data       Input/output buffer
 * @param width      Image width
 * @param height     Image height
 * @param mean       Mean value (default 0.5 for grayscale)
 * @param std        Std value (default 0.5 for grayscale)
 * @param stream     CUDA stream
 * @return cudaError_t 
 */
cudaError_t launchNormalize(
    float* data,
    int width,
    int height,
    float mean = 0.5f,
    float std = 0.5f,
    cudaStream_t stream = 0
);

/**
 * @brief Compute image hash for change detection
 * 
 * Computes a simple perceptual hash by downsampling and thresholding.
 * 
 * @param input      Grayscale input buffer
 * @param width      Image width
 * @param height     Image height
 * @param hashOutput Output hash value (single uint64_t on device)
 * @param stream     CUDA stream
 * @return cudaError_t 
 */
cudaError_t launchComputeHash(
    const float* input,
    int width,
    int height,
    uint64_t* hashOutput,
    cudaStream_t stream = 0
);

/**
 * @brief Compute perceptual hash for an image with row stride
 *
 * @param input       Input grayscale buffer (device)
 * @param inputStride Pitch/stride in pixels (not bytes)
 * @param width       Active width in pixels
 * @param height      Active height in pixels
 * @param hashOutput  Output hash (single uint64_t on device)
 * @param stream      CUDA stream
 */
cudaError_t launchComputeHashStrided(
    const float* input,
    int inputStride,
    int width,
    int height,
    uint64_t* hashOutput,
    cudaStream_t stream = 0
);

/**
 * @brief Preprocess frame for YOLO detection from texture
 *
 * Resizes frame to model input size (640x640), converts BGRA->RGB,
 * and normalizes to [0,1] in NCHW layout.
 *
 * @param texObj      CUDA texture object bound to BGRA frame
 * @param frameWidth  Input frame width
 * @param frameHeight Input frame height
 * @param output      Output buffer in NCHW layout [1, 3, 640, 640]
 * @param outWidth    Output width (typically 640)
 * @param outHeight   Output height (typically 640)
 * @param stream      CUDA stream
 * @return cudaError_t
 */
cudaError_t launchYOLOPreprocess(
    cudaTextureObject_t texObj,
    int frameWidth, int frameHeight,
    float* output,
    int outWidth, int outHeight,
    cudaStream_t stream = 0
);

/**
 * @brief Preprocess BGRA buffer for YOLO detection
 *
 * Same as launchYOLOPreprocess but takes raw BGRA pointer.
 *
 * @param input       Input BGRA buffer (device memory)
 * @param frameWidth  Input frame width
 * @param frameHeight Input frame height
 * @param output      Output buffer in NCHW layout
 * @param outWidth    Output width (typically 640)
 * @param outHeight   Output height (typically 640)
 * @param stream      CUDA stream
 * @return cudaError_t
 */
cudaError_t launchYOLOPreprocessFromBGRA(
    const uint8_t* input,
    int frameWidth, int frameHeight,
    float* output,
    int outWidth, int outHeight,
    cudaStream_t stream = 0
);

} // namespace cuda
} // namespace trading_monitor

