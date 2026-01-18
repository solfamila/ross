#pragma once

#include <cstdint>
#include <string>

namespace trading_monitor::detect {

struct EntryTriggerConfig {
    float armThreshold = 0.80f;
    float triggerThreshold = 0.80f;
    int armConfirmFrames = 2;
    int triggerConfirmFrames = 2;
    double delayCompensationMs = 0.0;
};

struct EntryEvent {
    bool fired = false;
    std::string targetSymbol;
    uint64_t absentLastFrame = 0;
    uint64_t presentFirstFrame = 0;
    double absentLastTimeMs = 0.0;
    double presentFirstTimeMs = 0.0;
    double absentLastTimeEstMs = 0.0;
    double presentFirstTimeEstMs = 0.0;
};

struct EntryTriggerState {
    bool armed = false;
    bool triggered = false;
    int armCount = 0;
    int presentCount = 0;
    uint64_t lastAbsentFrame = 0;
    double lastAbsentTimeMs = 0.0;
    uint64_t presentFirstFrame = 0;
    double presentFirstTimeMs = 0.0;
};

class EntryTrigger {
public:
    explicit EntryTrigger(EntryTriggerConfig cfg) : m_cfg(cfg) {}
    void reset() { m_state = EntryTriggerState{}; }
    void setTargetSymbol(const std::string& s) { m_target = s; }
    const EntryTriggerState& state() const { return m_state; }

    EntryEvent update(uint64_t frameIdx, double frameTimeMs,
                     float armScoreOrder, float armScoreQuote,
                     float triggerScore);

private:
    EntryTriggerConfig m_cfg;
    std::string m_target;
    EntryTriggerState m_state;
};

} // namespace trading_monitor::detect
