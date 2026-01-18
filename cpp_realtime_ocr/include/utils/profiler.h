#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trading_monitor {

struct FrameProfileRow {
    uint64_t frameNumber = 0;
    double totalMs = 0.0;
    double grayMs = 0.0;
    double findMs = 0.0;
    double trackMs = 0.0;
    double roiBuildMs = 0.0;
    double matchMs = 0.0;
    double stateMs = 0.0;
    bool armed = false;
    bool triggered = false;
    float triggerScore = 0.0f;
};

class Profiler {
public:
    explicit Profiler(bool enabled) : m_enabled(enabled) {}

    void beginFrame(uint64_t frameNumber);
    void markGray();
    void markFind();
    void markTrack();
    void markRoiBuild();
    void markMatch(float score);
    void markState(bool armed, bool triggered);
    void endFrame();

    void noteEvent(const std::string& name, uint64_t frameNumber, float score) {
        (void)name; (void)frameNumber; (void)score;
    }

    void flush(const std::string& jsonPath, const std::string& summaryPath);

private:
    bool m_enabled = false;
    uint64_t m_frameNumber = 0;
    int64_t m_t0 = 0;
    int64_t m_last = 0;
    std::vector<FrameProfileRow> m_rows;
};

} // namespace trading_monitor