/**
 * @file logger.cpp
 * @brief Logging utilities
 */

#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace trading_monitor {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

LogLevel g_minLogLevel = LogLevel::INFO;

void setLogLevel(LogLevel level) {
    g_minLogLevel = level;
}

std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time), "%H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

void log(LogLevel level, const std::string& message) {
    if (level < g_minLogLevel) return;
    
    const char* levelStr = "";
    switch (level) {
        case LogLevel::DEBUG:   levelStr = "[DEBUG]"; break;
        case LogLevel::INFO:    levelStr = "[INFO]"; break;
        case LogLevel::WARNING: levelStr = "[WARN]"; break;
        case LogLevel::ERROR:   levelStr = "[ERROR]"; break;
    }
    
    std::cout << getTimestamp() << " " << levelStr << " " << message << std::endl;
}

void logDebug(const std::string& message) { log(LogLevel::DEBUG, message); }
void logInfo(const std::string& message) { log(LogLevel::INFO, message); }
void logWarning(const std::string& message) { log(LogLevel::WARNING, message); }
void logError(const std::string& message) { log(LogLevel::ERROR, message); }

} // namespace trading_monitor

