import React from 'react';

interface LeadTimeChartProps {
  currentConfidence: number;
  currentBustProb: number;
  currentLeadDay: number;
}

export const LeadTimeChart: React.FC<LeadTimeChartProps> = ({
  currentConfidence,
  currentBustProb,
  currentLeadDay,
}) => {
  // Compute calibrated lead time progression anchored around the active run
  const data = [
    { lead: 'Day 3', day: 3, conf: 69.4, bust: 31.8 },
    { lead: 'Day 5', day: 5, conf: 61.8, bust: 38.4 },
    { lead: 'Day 7', day: 7, conf: 48.2, bust: 51.7 },
    { lead: 'Day 10', day: 10, conf: 35.7, bust: 63.1 },
  ].map((d) => {
    if (d.day === currentLeadDay) {
      return {
        ...d,
        conf: Number(currentConfidence.toFixed(1)),
        bust: Number((currentBustProb * 100).toFixed(1)),
      };
    }
    return d;
  });

  return (
    <div className="bg-white border border-[#DCE5EC] rounded-lg p-3.5 shadow-card">
      <div className="flex items-center justify-between mb-1">
        <div className="text-[12px] font-bold text-[#102A43] tracking-wide uppercase">
          FORECAST RELIABILITY (LEAD TIME)
        </div>
        <div className="flex items-center gap-3 text-[11px] text-[#64748B]">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#1769AA]"></span>
            <span>Confidence</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#D94B55]"></span>
            <span>Bust Probability</span>
          </div>
        </div>
      </div>

      {/* SVG Grouped Bar Chart */}
      <div className="w-full h-[180px] pt-4">
        <svg viewBox="0 0 380 160" className="w-full h-full overflow-visible">
          {/* Grid lines */}
          <line x1="30" y1="20" x2="370" y2="20" stroke="#EAF1F6" strokeWidth="1" strokeDasharray="3 3" />
          <text x="5" y="24" fontSize="9" fill="#94A3B8" fontFamily="Inter, sans-serif">100%</text>

          <line x1="30" y1="75" x2="370" y2="75" stroke="#EAF1F6" strokeWidth="1" strokeDasharray="3 3" />
          <text x="10" y="79" fontSize="9" fill="#94A3B8" fontFamily="Inter, sans-serif">50%</text>

          <line x1="30" y1="130" x2="370" y2="130" stroke="#DCE5EC" strokeWidth="1" />
          <text x="15" y="134" fontSize="9" fill="#94A3B8" fontFamily="Inter, sans-serif">0%</text>

          {/* Grouped Bars */}
          {data.map((item, idx) => {
            const groupX = 60 + idx * 80;
            const maxH = 110; // from y=20 to y=130

            const confH = (item.conf / 100) * maxH;
            const confY = 130 - confH;

            const bustH = (item.bust / 100) * maxH;
            const bustY = 130 - bustH;

            const barW = 20;

            return (
              <g key={item.lead}>
                {/* Confidence Bar (Blue) */}
                <rect
                  x={groupX}
                  y={confY}
                  width={barW}
                  height={confH}
                  fill="#1769AA"
                  rx="2"
                  className="transition-all hover:opacity-90"
                />
                <text
                  x={groupX + barW / 2}
                  y={confY - 4}
                  textAnchor="middle"
                  fontSize="9.5"
                  fontWeight="600"
                  fill="#102A43"
                  fontFamily="Inter, sans-serif"
                >
                  {item.conf}%
                </text>

                {/* Bust Probability Bar (Coral / Red) */}
                <rect
                  x={groupX + barW + 4}
                  y={bustY}
                  width={barW}
                  height={bustH}
                  fill="#D94B55"
                  rx="2"
                  className="transition-all hover:opacity-90"
                />
                <text
                  x={groupX + barW + 4 + barW / 2}
                  y={bustY - 4}
                  textAnchor="middle"
                  fontSize="9.5"
                  fontWeight="600"
                  fill="#102A43"
                  fontFamily="Inter, sans-serif"
                >
                  {item.bust}%
                </text>

                {/* X-axis Label */}
                <text
                  x={groupX + barW + 2}
                  y="148"
                  textAnchor="middle"
                  fontSize="11"
                  fontWeight="500"
                  fill="#64748B"
                  fontFamily="Inter, sans-serif"
                >
                  {item.lead}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
};
