/**
 * @file ctc_decoder.cpp
 * @brief CTC decoding for OCR output
 *
 * Implements greedy CTC decoding for text recognition
 */

#include "ocr/ctc_decoder.h"
#include <fstream>
#include <algorithm>
#include <cmath>

namespace trading_monitor {

CTCDecoder::CTCDecoder() = default;
CTCDecoder::~CTCDecoder() = default;

bool CTCDecoder::loadDictionary(const std::string& dictPath) {
    std::ifstream file(dictPath);
    if (!file.good()) return false;

    m_dictionary.clear();
    m_dictionary.push_back("");  // Index 0 is typically blank token

    std::string line;
    while (std::getline(file, line)) {
        // Remove trailing whitespace
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        m_dictionary.push_back(line);
    }

    return !m_dictionary.empty();
}

void CTCDecoder::setDictionary(const std::vector<std::string>& chars) {
    m_dictionary = chars;
}

bool CTCDecoder::ensurePaddleOCRDictionarySize(int numClasses) {
    if (numClasses <= 0) return false;
    if (m_dictionary.empty()) return false;

    // Typical PaddleOCR:
    // - blank at index 0 (we already insert "" at 0 on load)
    // - optional space character as an extra class
    if (static_cast<int>(m_dictionary.size()) == numClasses) {
        return true;
    }

    if (static_cast<int>(m_dictionary.size()) == (numClasses - 1)) {
        m_dictionary.push_back(" ");
        return static_cast<int>(m_dictionary.size()) == numClasses;
    }

    return false;
}

CTCResult CTCDecoder::decode(
    const float* logits,
    int timesteps,
    int numClasses,
    int blankIndex
) const {
    CTCResult result;
    result.confidence = 0.0f;

    int lastIdx = -1;
    float confidenceSum = 0.0f;
    int charCount = 0;

    // Temporary buffer for softmax
    std::vector<float> softmaxProbs(numClasses);

    for (int t = 0; t < timesteps; t++) {
        const float* timestepLogits = logits + t * numClasses;

        // Heuristic: some exports already output probabilities (post-softmax).
        // If values are in [0,1] and sum is ~1, treat as probs and skip softmax.
        float minVal = timestepLogits[0];
        float maxVal = timestepLogits[0];
        float sumVal = timestepLogits[0];
        for (int c = 1; c < numClasses; c++) {
            const float v = timestepLogits[c];
            minVal = (std::min)(minVal, v);
            maxVal = (std::max)(maxVal, v);
            sumVal += v;
        }

        const bool looksLikeProbs = (minVal >= -1e-4f) && (maxVal <= 1.0f + 1e-3f) && (std::fabs(sumVal - 1.0f) < 1e-2f);

        if (looksLikeProbs) {
            for (int c = 0; c < numClasses; c++) {
                softmaxProbs[c] = timestepLogits[c];
            }
        } else {
            // Apply softmax to convert logits to probabilities
            float maxLogit = timestepLogits[0];
            for (int c = 1; c < numClasses; c++) {
                if (timestepLogits[c] > maxLogit) {
                    maxLogit = timestepLogits[c];
                }
            }

            float sumExp = 0.0f;
            for (int c = 0; c < numClasses; c++) {
                softmaxProbs[c] = std::exp(timestepLogits[c] - maxLogit);
                sumExp += softmaxProbs[c];
            }
            for (int c = 0; c < numClasses; c++) {
                softmaxProbs[c] /= sumExp;
            }
        }

        // Find argmax for this timestep
        int maxIdx = 0;
        float maxProb = softmaxProbs[0];
        for (int c = 1; c < numClasses; c++) {
            if (softmaxProbs[c] > maxProb) {
                maxProb = softmaxProbs[c];
                maxIdx = c;
            }
        }

        // Skip blank and repeated characters.
        // PaddleOCR uses blank index 0 by default.
        int effectiveBlank = blankIndex;
        if (effectiveBlank < 0) {
            effectiveBlank = numClasses - 1;
        }

        if (maxIdx != effectiveBlank && maxIdx != lastIdx) {
            // Map to character (index 0 in output = first char in dict)
            if (maxIdx < static_cast<int>(m_dictionary.size())) {
                result.text += m_dictionary[maxIdx];
                result.charConfidences.push_back(maxProb);
                confidenceSum += maxProb;
                charCount++;
            }
        }

        lastIdx = maxIdx;
    }

    // Compute average confidence
    if (charCount > 0) {
        result.confidence = confidenceSum / charCount;
    }

    return result;
}

size_t CTCDecoder::getDictionarySize() const {
    return m_dictionary.size();
}

} // namespace trading_monitor

