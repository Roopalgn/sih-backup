import React from 'react';

interface LeadPoint {
  day: number;
  conf: number;    // 0–100
  bust: number;    // 0–100
  isReal: boolean;
}

interface LeadTimeChartProps {
  points: LeadPoint[];
  activeLead: number;
}

export const LeadTimeChart: React.FC<LeadTimeChartProps> = ({ points, activeLead }) => {
  if (!points.length) return null;

  const W = 360;
  const H = 140;
  const PAD = { top: 24, right: 16, bottom: 28, left: 32 };
  const chartW = W - PAD.left - PAD.right;
  const chartH = H - PAD.top - PAD.bottom;

  const xs = points.map((_, i) => PAD.left + (i / (points.length - 1 || 1)) * chartW);
  const toY = (pct: number) => PAD.top + chartH - (pct / 100) * chartH;

  const confPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${xs[i].toFixed(1)},${toY(p.conf).toFixed(1)}`).join(' ');
  const bustPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${xs[i].toFixed(1)},${toY(p.bust).toFixed(1)}`).join(' ');

  const gridLines = [0, 25, 50, 75, 100];

  return (
    <div className="lead-chart-card">
      <div className="lead-chart-header">
        <span className="panel-kicker">LEAD-TIME RELIABILITY</span>
        <div className="lead-chart-legend">
          <span><i style={{ background: '#7ac9f7' }} />Confidence</span>
          <span><i style={{ background: '#f08c6e' }} />Bust Prob</span>
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        {/* Grid lines */}
        {gridLines.map(pct => (
          <g key={pct}>
            <line
              x1={PAD.left} y1={toY(pct)}
              x2={PAD.left + chartW} y2={toY(pct)}
              stroke="#2a405d" strokeWidth="1" strokeDasharray={pct === 0 ? 'none' : '3 3'}
            />
            <text x={PAD.left - 5} y={toY(pct) + 3.5} textAnchor="end"
              fontSize="8" fill="#7191b3" fontFamily="DM Mono, monospace">
              {pct}
            </text>
          </g>
        ))}

        {/* Confidence line */}
        <path d={confPath} fill="none" stroke="#7ac9f7" strokeWidth="2" strokeLinejoin="round" />
        {/* Bust line */}
        <path d={bustPath} fill="none" stroke="#f08c6e" strokeWidth="2" strokeLinejoin="round" />

        {/* Points */}
        {points.map((p, i) => {
          const isActive = p.day === activeLead;
          const cx = xs[i];
          return (
            <g key={p.day}>
              {/* Active lead highlight bar */}
              {isActive && (
                <rect
                  x={cx - 12} y={PAD.top}
                  width={24} height={chartH}
                  fill="#6dafff10" rx="3"
                />
              )}
              {/* Confidence dot */}
              <circle cx={cx} cy={toY(p.conf)} r={isActive ? 5 : 3.5}
                fill={isActive ? '#7ac9f7' : '#3679bb'}
                stroke={isActive ? '#edf7ff' : 'none'} strokeWidth="1.5" />
              {/* Bust dot */}
              <circle cx={cx} cy={toY(p.bust)} r={isActive ? 5 : 3.5}
                fill={isActive ? '#f08c6e' : '#b34e38'}
                stroke={isActive ? '#edf7ff' : 'none'} strokeWidth="1.5" />
              {/* X label */}
              <text x={cx} y={H - 5} textAnchor="middle"
                fontSize="9" fill={isActive ? '#e8f3ff' : '#7191b3'}
                fontWeight={isActive ? '600' : '400'}
                fontFamily="DM Mono, monospace">
                D{String(p.day).padStart(2, '0')}
              </text>
              {/* Value labels on active point */}
              {isActive && (
                <>
                  <text x={cx} y={toY(p.conf) - 8} textAnchor="middle"
                    fontSize="9" fill="#a8dcf7" fontFamily="DM Mono, monospace">
                    {p.conf.toFixed(1)}%
                  </text>
                  <text x={cx} y={toY(p.bust) + 14} textAnchor="middle"
                    fontSize="9" fill="#f4a88a" fontFamily="DM Mono, monospace">
                    {p.bust.toFixed(1)}%
                  </text>
                </>
              )}
              {/* Dashed indicator for estimated points */}
              {!p.isReal && (
                <circle cx={cx} cy={toY(p.conf)} r={isActive ? 5 : 3.5}
                  fill="none" stroke="#4a6a8a" strokeWidth="1" strokeDasharray="2 2" />
              )}
            </g>
          );
        })}
      </svg>
      <div className="lead-chart-note">
        Values for Days 3, 5, 7, 10 from live model · other leads not in training scope
      </div>
    </div>
  );
};
