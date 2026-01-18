#pragma once
/**
 * @file ctc_decoder.h
 * @brief CTC decoding for OCR output
 */

#include <string>
#include <vector>

namespace trading_monitor {

/**
 * @brief CTC decoder result
 */
struct CTCResult {
    std::string text;           ///< Decoded text
    float confidence;           ///< Average confidence
    std::vector<float> charConfidences;  ///< Per-character confidence
};

/**
 * @brief CTC decoder for text recognition
 * 
 * Implements greedy CTC decoding with optional beam search.
 */
class CTCDecoder {
public:
    CTCDecoder();
    ~CTCDecoder();
    
    /**
     * @brief Load character dictionary
     * @param dictPath Path to dictionary file (one char per line)
     * @return true if successful
     */
    bool loadDictionary(const std::string& dictPath);
    
    /**
     * @brief Set dictionary from character list
     * @param chars Vector of characters (index = class ID)
     */
    void setDictionary(const std::vector<std::string>& chars);

    /**
     * @brief Ensure dictionary size matches PaddleOCR output classes.
     *
     * PaddleOCR recognition models often have:
     * - blank token at index 0
     * - optional space character as an extra class
     *
     * This helper appends a single-space token if the dictionary is exactly
     * one entry short (common when ppocr_keys_v1.txt is used with a model that
     * was trained with `use_space_char=True`).
     */
    bool ensurePaddleOCRDictionarySize(int numClasses);
    
    /**
     * @brief Decode output probabilities using greedy decoding
     * @param probs Output probabilities [timesteps x num_classes]
     * @param timesteps Number of timesteps
     * @param numClasses Number of character classes
     * @param blankIndex Index of blank token (default: 0)
     * @return Decoded result
     */
    CTCResult decode(
        const float* probs,
        int timesteps,
        int numClasses,
        int blankIndex = 0
    ) const;
    
    /**
     * @brief Get number of characters in dictionary
     */
    size_t getDictionarySize() const;
    
private:
    std::vector<std::string> m_dictionary;
    int m_blankIndex = 0;
};

} // namespace trading_monitor

