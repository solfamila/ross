#include "detection/entry_trigger.h"

namespace trading_monitor::detect {

EntryEvent EntryTrigger::update(uint64_t frameIdx, double frameTimeMs,
                                float armScoreOrder, float armScoreQuote,
                                float triggerScore) {
    EntryEvent ev{};
    ev.targetSymbol = m_target;

    auto markAbsent = [&](){
        m_state.lastAbsentFrame = frameIdx;
        m_state.lastAbsentTimeMs = frameTimeMs;
    };

    if (!m_state.armed) {
        const bool armHit = (armScoreOrder >= m_cfg.armThreshold) || (armScoreQuote >= m_cfg.armThreshold);
        if (armHit) {
            if (++m_state.armCount >= m_cfg.armConfirmFrames) {
                m_state.armed = true;
                m_state.presentCount = 0;
                markAbsent();
            }
        } else {
            m_state.armCount = 0;
        }
        return ev;
    }

    if (m_state.triggered) return ev;

    const bool present = (triggerScore >= m_cfg.triggerThreshold);
    if (present) {
        if (++m_state.presentCount == 1) {
            m_state.presentFirstFrame = frameIdx;
            m_state.presentFirstTimeMs = frameTimeMs;
        }
        if (m_state.presentCount >= m_cfg.triggerConfirmFrames) {
            m_state.triggered = true;
            ev.fired = true;
            ev.absentLastFrame = m_state.lastAbsentFrame;
            ev.absentLastTimeMs = m_state.lastAbsentTimeMs;
            ev.presentFirstFrame = m_state.presentFirstFrame;
            ev.presentFirstTimeMs = m_state.presentFirstTimeMs;
            ev.absentLastTimeEstMs = ev.absentLastTimeMs - m_cfg.delayCompensationMs;
            ev.presentFirstTimeEstMs = ev.presentFirstTimeMs - m_cfg.delayCompensationMs;
        }
    } else {
        m_state.presentCount = 0;
        markAbsent();
    }
    return ev;
}

} // namespace trading_monitor::detect