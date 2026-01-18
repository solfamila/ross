/**
 * @file timer.cpp
 * @brief Timer utility implementations
 * 
 * Note: Most timer functionality is header-only in utils/timer.h
 * This file contains any additional utility functions.
 */

#include "utils/timer.h"
#include "types.h"
#include <iostream>
#include <iomanip>

namespace trading_monitor {

/**
 * @brief Print pipeline stats in formatted table
 */
void printStats(const PipelineStats& stats) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "┌─────────────────────────────────────────┐\n";
    std::cout << "│ Pipeline Timing                         │\n";
    std::cout << "├─────────────────────────────────────────┤\n";
    std::cout << "│ Capture:        " << std::setw(8) << stats.captureMs << " ms          │\n";
    std::cout << "│ D3D11→CUDA:     " << std::setw(8) << stats.interopMs << " ms          │\n";
    std::cout << "│ Preprocess:     " << std::setw(8) << stats.preprocessMs << " ms          │\n";
    std::cout << "│ Inference:      " << std::setw(8) << stats.inferenceMs << " ms          │\n";
    std::cout << "│ Decode:         " << std::setw(8) << stats.decodeMs << " ms          │\n";
    std::cout << "│ Parse:          " << std::setw(8) << stats.parseMs << " ms          │\n";
    std::cout << "├─────────────────────────────────────────┤\n";
    std::cout << "│ TOTAL:          " << std::setw(8) << stats.totalMs << " ms          │\n";
    std::cout << "└─────────────────────────────────────────┘\n";
}

} // namespace trading_monitor

