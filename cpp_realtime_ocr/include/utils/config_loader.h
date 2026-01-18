#pragma once
/**
 * @file config_loader.h
 * @brief Configuration file loading utilities
 */

#include "types.h"
#include <string>
#include <vector>

namespace trading_monitor {

/**
 * @brief Load ROI configuration from JSON file
 * @param path Path to roi_config.json
 * @return Vector of ROI definitions
 */
std::vector<ROI> loadROIConfig(const std::string& path);

/**
 * @brief Load full pipeline configuration
 * @param path Path to config file
 * @return Pipeline configuration
 */
PipelineConfig loadPipelineConfig(const std::string& path);

/**
 * @brief Add or update a single ROI entry in the config JSON.
 *
 * Preserves other top-level fields (e.g., "meta"). If the file does not exist,
 * a minimal structure will be created.
 *
 * @param path Path to roi_config.json
 * @param roi ROI to insert/update (matched by name)
 * @param errorMessage Populated on failure
 * @return true on success
 */
bool upsertROIConfig(const std::string& path, const ROI& roi, std::string& errorMessage);

} // namespace trading_monitor

