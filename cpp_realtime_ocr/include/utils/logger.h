#pragma once
/**
 * @file logger.h
 * @brief Logging utilities
 */

#include <string>

namespace trading_monitor {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

/**
 * @brief Set minimum log level
 */
void setLogLevel(LogLevel level);

/**
 * @brief Log a message
 */
void log(LogLevel level, const std::string& message);

/**
 * @brief Convenience logging functions
 */
void logDebug(const std::string& message);
void logInfo(const std::string& message);
void logWarning(const std::string& message);
void logError(const std::string& message);

} // namespace trading_monitor

