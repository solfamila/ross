/**
 * @file config_loader.cpp
 * @brief Configuration file loading
 */

#include "types.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace trading_monitor {

using json = nlohmann::json;

static json roiToJson(const ROI& roi) {
    json r;
    r["name"] = roi.name;
    r["x"] = roi.x;
    r["y"] = roi.y;
    r["w"] = roi.w;
    r["h"] = roi.h;
    return r;
}

/**
 * @brief Load ROI configuration from JSON file
 * 
 * Expected format (matches roi_config.json):
 * {
 *   "meta": { ... },
 *   "rois": [
 *     { "name": "...", "x": 0, "y": 0, "w": 100, "h": 50 },
 *     ...
 *   ]
 * }
 */
std::vector<ROI> loadROIConfig(const std::string& path) {
    std::vector<ROI> rois;
    
    std::ifstream file(path);
    if (!file.good()) {
        std::cerr << "Cannot open config file: " << path << std::endl;
        return rois;
    }
    
    try {
        json config = json::parse(file);
        
        if (config.contains("rois")) {
            for (const auto& roi : config["rois"]) {
                ROI r;
                r.name = roi.value("name", "");
                r.x = roi.value("x", 0);
                r.y = roi.value("y", 0);
                r.w = roi.value("w", 0);
                r.h = roi.value("h", 0);
                rois.push_back(r);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing config: " << e.what() << std::endl;
    }
    
    return rois;
}

/**
 * @brief Load full pipeline configuration
 */
PipelineConfig loadPipelineConfig(const std::string& path) {
    PipelineConfig config;
    config.rois = loadROIConfig(path);
    
    // TODO: Load additional settings from config file
    // For now, use defaults from types.h
    
    return config;
}

bool upsertROIConfig(const std::string& path, const ROI& roi, std::string& errorMessage) {
    errorMessage.clear();
    if (roi.name.empty()) {
        errorMessage = "ROI name is empty";
        return false;
    }
    if (roi.w <= 0 || roi.h <= 0) {
        errorMessage = "ROI has non-positive size";
        return false;
    }

    json config = json::object();

    // Load existing file if present.
    {
        std::ifstream in(path);
        if (in.good()) {
            try {
                config = json::parse(in);
            } catch (const std::exception& e) {
                errorMessage = std::string("Error parsing config: ") + e.what();
                return false;
            }
        }
    }

    if (!config.is_object()) {
        config = json::object();
    }

    if (!config.contains("rois") || !config["rois"].is_array()) {
        config["rois"] = json::array();
    }

    bool updated = false;
    for (auto& r : config["rois"]) {
        if (r.is_object() && r.value("name", "") == roi.name) {
            r = roiToJson(roi);
            updated = true;
            break;
        }
    }
    if (!updated) {
        config["rois"].push_back(roiToJson(roi));
    }

    try {
        std::filesystem::path p(path);
        if (p.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(p.parent_path(), ec);
        }

        std::ofstream out(path, std::ios::trunc);
        if (!out.good()) {
            errorMessage = "Cannot open config for writing: " + path;
            return false;
        }
        out << config.dump(2) << std::endl;
    } catch (const std::exception& e) {
        errorMessage = std::string("Error writing config: ") + e.what();
        return false;
    }

    return true;
}

} // namespace trading_monitor

