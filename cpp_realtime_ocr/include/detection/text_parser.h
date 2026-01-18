#pragma once
/**
 * @file text_parser.h
 * @brief Text parsing for trading position data
 * 
 * Ports functionality from realtime_ocr/parser.py
 */

#include "types.h"
#include <string>
#include <optional>
#include <vector>

namespace trading_monitor {

/**
 * @brief Parser for trading position data from OCR text
 */
class TextParser {
public:
    TextParser() = default;
    
    /**
     * @brief Parse a number from OCR text with error correction
     * 
     * Handles common OCR mistakes:
     * - O/o → 0
     * - l/I → 1
     * - S → 5
     * - Parentheses as negative
     * 
     * @param text Input text
     * @return Parsed number or nullopt if parsing fails
     */
    std::optional<double> parseNumber(const std::string& text) const;
    
    /**
     * @brief Parse an integer from OCR text
     * 
     * @param text Input text
     * @return Parsed integer or nullopt
     */
    std::optional<int> parseInteger(const std::string& text) const;
    
    /**
     * @brief Parse a position row from Lightspeed
     * 
     * Expected format: "shares avg_price unrealized_pnl realized_pnl"
     * Example: "1000 45.67 234.50 -12.30"
     * 
     * @param text Full row text
     * @param confidence OCR confidence score
     * @return Parsed position data
     */
    PositionData parsePositionRow(const std::string& text, double confidence = 0.0) const;
    
    /**
     * @brief Parse price from OCR text
     * 
     * @param text Price text
     * @return Parsed price or nullopt
     */
    std::optional<double> parsePrice(const std::string& text) const;
    
    /**
     * @brief Tokenize text by whitespace
     * 
     * @param text Input text
     * @return Vector of tokens
     */
    std::vector<std::string> tokenize(const std::string& text) const;
    
    /**
     * @brief Clean OCR text (remove common artifacts)
     * 
     * @param text Input text
     * @return Cleaned text
     */
    std::string cleanText(const std::string& text) const;
    
private:
    /**
     * @brief Apply OCR error corrections to text
     */
    std::string applyOCRCorrections(std::string text) const;
};

} // namespace trading_monitor

