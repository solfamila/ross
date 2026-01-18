/**
 * @file text_parser.cpp
 * @brief Text parsing implementation for trading position data
 * 
 * Ported from realtime_ocr/parser.py
 */

#include "detection/text_parser.h"
#include <algorithm>
#include <sstream>
#include <cctype>

namespace trading_monitor {

std::string TextParser::applyOCRCorrections(std::string text) const {
    // Common OCR error corrections
    for (char& c : text) {
        switch (c) {
            case 'O': case 'o': c = '0'; break;
            case 'l': case 'I': c = '1'; break;
            case 'S': c = '5'; break;
        }
    }
    return text;
}

std::string TextParser::cleanText(const std::string& text) const {
    std::string result;
    result.reserve(text.size());
    
    for (char c : text) {
        // Remove common artifacts
        if (c == ',' || c == '$' || c == ' ') {
            continue;
        }
        result.push_back(c);
    }
    
    return result;
}

std::vector<std::string> TextParser::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string token;
    
    while (iss >> token) {
        tokens.push_back(token);
    }
    
    return tokens;
}

std::optional<double> TextParser::parseNumber(const std::string& text) const {
    if (text.empty()) {
        return std::nullopt;
    }
    
    // Clean and apply corrections
    std::string cleaned = applyOCRCorrections(cleanText(text));
    
    if (cleaned.empty()) {
        return std::nullopt;
    }
    
    // Handle parentheses as negative
    bool isNegative = false;
    if (cleaned.front() == '(' && cleaned.back() == ')') {
        isNegative = true;
        cleaned = cleaned.substr(1, cleaned.length() - 2);
    } else if (cleaned.front() == '-') {
        isNegative = true;
        cleaned = cleaned.substr(1);
    }
    
    if (cleaned.empty()) {
        return std::nullopt;
    }
    
    try {
        double value = std::stod(cleaned);
        return isNegative ? -value : value;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::optional<int> TextParser::parseInteger(const std::string& text) const {
    auto value = parseNumber(text);
    if (value.has_value()) {
        return static_cast<int>(*value);
    }
    return std::nullopt;
}

std::optional<double> TextParser::parsePrice(const std::string& text) const {
    return parseNumber(text);
}

PositionData TextParser::parsePositionRow(const std::string& text, double confidence) const {
    PositionData result;
    result.rawText = text;
    result.confidence = confidence;
    
    auto tokens = tokenize(text);
    
    if (tokens.size() >= 1) {
        result.shares = parseInteger(tokens[0]);
    }
    if (tokens.size() >= 2) {
        result.avgPrice = parseNumber(tokens[1]);
    }
    if (tokens.size() >= 3) {
        result.unrealizedPnl = parseNumber(tokens[2]);
    }
    if (tokens.size() >= 4) {
        result.realizedPnl = parseNumber(tokens[3]);
    }
    
    return result;
}

} // namespace trading_monitor

