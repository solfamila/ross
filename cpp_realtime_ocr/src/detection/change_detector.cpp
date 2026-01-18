/**
 * @file change_detector.cpp
 * @brief Frame change detection to skip unchanged content
 */

#include "detection/change_detector.h"
#include "processing/cuda_kernels.h"

#include <algorithm>
#include <iostream>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace trading_monitor {

ChangeDetector::ChangeDetector() = default;

ChangeDetector::~ChangeDetector() {
	if (m_deviceHash) {
		cudaFree(m_deviceHash);
		m_deviceHash = nullptr;
	}
}

void ChangeDetector::initialize(size_t numROIs, int threshold) {
	m_previousHashes.assign(numROIs, 0);
	m_valid.assign(numROIs, false);
	m_threshold = (std::max)(0, threshold);

	if (!m_deviceHash) {
		cudaError_t err = cudaMalloc(&m_deviceHash, sizeof(uint64_t));
		if (err != cudaSuccess) {
			std::cerr << "ChangeDetector: cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
			m_deviceHash = nullptr;
		}
	}
}

int ChangeDetector::hammingDistance(uint64_t a, uint64_t b) const {
	const uint64_t x = a ^ b;
#ifdef _MSC_VER
	return static_cast<int>(__popcnt64(x));
#else
	return static_cast<int>(__builtin_popcountll(x));
#endif
}

bool ChangeDetector::hasChanged(
	size_t roiIndex,
	const float* imageData,
	int width,
	int height,
	cudaStream_t stream
) {
	return hasChangedStrided(roiIndex, imageData, width, width, height, stream);
}

bool ChangeDetector::hasChangedStrided(
	size_t roiIndex,
	const float* imageData,
	int inputStride,
	int width,
	int height,
	cudaStream_t stream
) {
	if (!m_deviceHash || !imageData) return true;
	if (roiIndex >= m_previousHashes.size() || roiIndex >= m_valid.size()) return true;
	if (width <= 0 || height <= 0 || inputStride < width) return true;

	// If the ROI is too small, treat as changed so OCR gets a chance to run.
	if (width < 8 || height < 8) {
		m_valid[roiIndex] = false;
		return true;
	}

	cudaError_t err = cuda::launchComputeHashStrided(imageData, inputStride, width, height, m_deviceHash, stream);
	if (err != cudaSuccess) {
		return true;
	}

	uint64_t h = 0;
	cudaMemcpyAsync(&h, m_deviceHash, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	if (!m_valid[roiIndex]) {
		m_previousHashes[roiIndex] = h;
		m_valid[roiIndex] = true;
		return true;
	}

	const uint64_t prev = m_previousHashes[roiIndex];
	const int dist = hammingDistance(prev, h);
	const bool changed = dist >= m_threshold;
	if (changed) {
		m_previousHashes[roiIndex] = h;
	}
	return changed;
}

void ChangeDetector::invalidate(size_t roiIndex) {
	if (roiIndex < m_valid.size()) {
		m_valid[roiIndex] = false;
	}
}

void ChangeDetector::invalidateAll() {
	std::fill(m_valid.begin(), m_valid.end(), false);
}

uint64_t ChangeDetector::getCurrentHash(size_t roiIndex) const {
	if (roiIndex >= m_previousHashes.size()) return 0;
	return m_previousHashes[roiIndex];
}

} // namespace trading_monitor

