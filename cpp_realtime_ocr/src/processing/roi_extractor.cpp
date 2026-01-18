/**
 * @file roi_extractor.cpp
 * @brief ROI extraction and preprocessing pipeline
 *
 * Produces PaddleOCR-style SVTR input tensors:
 * - NCHW float
 * - 3 channels (grayscale replicated)
 * - height fixed to 48, width padded/strided to 320
 * - normalized to [-1, 1] via (x - 0.5) / 0.5
 */

#include "processing/roi_extractor.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace trading_monitor {

namespace {
constexpr int kModelH = 48;
constexpr int kModelW = 320;
constexpr int kModelC = 3;

static int clampInt(int v, int lo, int hi) {
	return (std::max)(lo, (std::min)(hi, v));
}

} // namespace

ROIExtractor::ROIExtractor() = default;

ROIExtractor::~ROIExtractor() {
	cleanup();
}

bool ROIExtractor::initialize(const std::vector<ROI>& rois, float upscaleFactor) {
	cleanup();

	m_rois = rois;
	m_upscaleFactor = upscaleFactor;
	if (m_preUpscaleFactor < 1.0f) m_preUpscaleFactor = 1.0f;

	// Note: current implementation matches the recognition model input (H=48, W=320).
	// The upscaleFactor is kept for compatibility but not applied (would require a 2-stage resize).

	m_buffers.clear();
	m_buffers.resize(m_rois.size());

	for (size_t i = 0; i < m_rois.size(); i++) {
		auto& buf = m_buffers[i];
		buf.width = kModelW;
		buf.height = kModelH;
		buf.activeWidth = 0;

		const size_t elems = static_cast<size_t>(kModelC) * buf.width * buf.height;
		cudaError_t err = cudaMalloc(&buf.data, elems * sizeof(float));
		if (err != cudaSuccess) {
			std::cerr << "ROIExtractor: cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
			cleanup();
			return false;
		}
	}

	return true;
}

bool ROIExtractor::initialize(size_t roiCount, float upscaleFactor) {
	cleanup();

	m_rois.clear();
	m_upscaleFactor = upscaleFactor;
	if (m_preUpscaleFactor < 1.0f) m_preUpscaleFactor = 1.0f;

	// Note: current implementation matches the recognition model input (H=48, W=320).
	// The upscaleFactor is kept for compatibility but not applied (would require a 2-stage resize).

	m_buffers.clear();
	m_buffers.resize(roiCount);

	for (size_t i = 0; i < roiCount; i++) {
		auto& buf = m_buffers[i];
		buf.width = kModelW;
		buf.height = kModelH;
		buf.activeWidth = 0;

		const size_t elems = static_cast<size_t>(kModelC) * buf.width * buf.height;
		cudaError_t err = cudaMalloc(&buf.data, elems * sizeof(float));
		if (err != cudaSuccess) {
			std::cerr << "ROIExtractor: cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
			cleanup();
			return false;
		}
	}

	return true;
}

float* ROIExtractor::extractROI(cudaTextureObject_t texObj, size_t roiIndex, cudaStream_t stream) {
	if (roiIndex >= m_rois.size()) {
		return nullptr;
	}
	return extractROI(texObj, m_rois[roiIndex], roiIndex, stream);
}

float* ROIExtractor::extractROI(
	cudaTextureObject_t texObj,
	const ROI& roi,
	size_t bufferIndex,
	cudaStream_t stream
) {
	if (bufferIndex >= m_buffers.size()) {
		return nullptr;
	}
	if (!texObj) {
		return nullptr;
	}

	ROIBuffer& buf = m_buffers[bufferIndex];
	if (!buf.data || buf.width <= 0 || buf.height <= 0) {
		return nullptr;
	}
	if (roi.w <= 0 || roi.h <= 0) {
		return nullptr;
	}

	// Clear to 0 so that after normalize (mean=0.5, std=0.5) padding becomes -1.
	const size_t elems = static_cast<size_t>(kModelC) * buf.width * buf.height;
	cudaMemsetAsync(buf.data, 0, elems * sizeof(float), stream);

	// Pre-upscale/zoom into ROI center if requested (helps small text).
	const float zoom = (std::max)(1.0f, m_preUpscaleFactor);
	ROI effective = roi;
	if (zoom > 1.0f) {
		const int effW = (std::max)(1, static_cast<int>(std::lround(static_cast<float>(roi.w) / zoom)));
		const int effH = (std::max)(1, static_cast<int>(std::lround(static_cast<float>(roi.h) / zoom)));
		const int dx = (roi.w - effW) / 2;
		const int dy = (roi.h - effH) / 2;
		effective.x = roi.x + dx;
		effective.y = roi.y + dy;
		effective.w = effW;
		effective.h = effH;
	}

	// Compute aspect-preserving resized width at fixed height.
	// Standard PaddleOCR approach: resize to height=48 then pad to width=320.
	const float scale = static_cast<float>(kModelH) / (std::max)(1.0f, static_cast<float>(effective.h));
	int resizedW = static_cast<int>(std::round(static_cast<float>(effective.w) * scale));
	resizedW = clampInt(resizedW, 1, kModelW);
	buf.activeWidth = resizedW;

	// Map ROI -> resizedW x 48 using isotropic scaling.
	// scale = outHeight / roi.h == outWidth / roi.w (approximately due to rounding)
	// To keep kernel mapping consistent, derive scale from width.
	const float scaleW = static_cast<float>(resizedW) / (std::max)(1.0f, static_cast<float>(effective.w));

	cuda::ROIParams params{};
	params.roiX = effective.x;
	params.roiY = effective.y;
	params.roiWidth = effective.w;
	params.roiHeight = effective.h;
	params.outWidth = resizedW;
	params.outHeight = kModelH;
	params.scale = scaleW;

	if (cuda::launchBgraToNCHW3ROIStrided(texObj, buf.data, kModelW, params, stream) != cudaSuccess) {
		return nullptr;
	}

	// Contrast + normalize, per-channel.
	const size_t planeStride = static_cast<size_t>(buf.width) * buf.height;
	for (int c = 0; c < kModelC; c++) {
		float* plane = buf.data + static_cast<size_t>(c) * planeStride;
		if (cuda::launchContrastEnhance(plane, buf.width, buf.height, m_contrastParams, stream) != cudaSuccess) {
			return nullptr;
		}
		if (cuda::launchNormalize(plane, buf.width, buf.height, 0.5f, 0.5f, stream) != cudaSuccess) {
			return nullptr;
		}
	}

	return buf.data;
}

void ROIExtractor::getOutputDimensions(size_t roiIndex, int& width, int& height) const {
	if (roiIndex >= m_buffers.size()) {
		width = 0;
		height = 0;
		return;
	}
	width = m_buffers[roiIndex].width;
	height = m_buffers[roiIndex].height;
}

int ROIExtractor::getActiveOutputWidth(size_t roiIndex) const {
	if (roiIndex >= m_buffers.size()) return 0;
	return m_buffers[roiIndex].activeWidth;
}

size_t ROIExtractor::getROICount() const {
	// In the dynamic ROI mode, m_rois may be empty while buffers are allocated.
	return m_buffers.size();
}

void ROIExtractor::setPreUpscaleFactor(float factor) {
	m_preUpscaleFactor = (std::max)(1.0f, factor);
}

void ROIExtractor::cleanup() {
	for (auto& b : m_buffers) {
		if (b.data) {
			cudaFree(b.data);
			b.data = nullptr;
		}
	}
	m_buffers.clear();
	m_rois.clear();
}

} // namespace trading_monitor

