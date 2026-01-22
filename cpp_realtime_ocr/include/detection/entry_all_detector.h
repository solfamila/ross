#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace trading_monitor::detect {

struct EntryAllEvent {
	uint64_t frame = 0;
	double t_ms = 0.0;
	int row = -1;
	std::string symbol;
	float ocr_score = -1.0f;
};

struct EntryAllConfig {
	int maxRows = 3;
	int rowStride = 0; // 0 = auto (roiH / maxRows)
	int bandH = 32;
	int symbolX0 = 0;
	int symbolX1 = 90;
	double diffThreshold = 2.5;
	float ocrThreshold = 0.50f;
	bool debug = false;
	int debugMax = 5;
};

class EntryAllDetector {
public:
	explicit EntryAllDetector(EntryAllConfig cfg) : m_cfg(cfg) { reset(); }
	void reset();

	std::vector<EntryAllEvent> update(uint64_t frameIdx, double frameTimeMs,
									 const std::vector<uint8_t>& frameGray, int frameW, int frameH,
									 int roiX, int roiY, int roiW, int roiH);

private:
	EntryAllConfig m_cfg;

	struct RowState {
		bool active = false;
		std::string symbol;
		std::vector<uint8_t> prevBand;
		int prevW = 0;
		int prevH = 0;
	};
	std::vector<RowState> m_rows;
	int m_debugRemaining = 0;

	static bool isTickerLike(const std::string& s);
};

} // namespace trading_monitor::detect
