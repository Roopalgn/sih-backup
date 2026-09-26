import React from 'react';
import type { SubdivisionStat } from '../types';

interface RegionalRiskTableProps {
  stats: SubdivisionStat[];
}

export const RegionalRiskTable: React.FC<RegionalRiskTableProps> = ({ stats }) => {
  return (
    <div className="bg-white border border-[#DCE5EC] rounded-lg p-3.5 shadow-card">
      <div className="flex items-center justify-between mb-2">
        <div className="text-[12px] font-bold text-[#102A43] tracking-wide uppercase">
          REGIONAL RISK (TOP 5)
        </div>
        <button className="text-[11px] font-semibold text-[#1769AA] hover:underline flex items-center gap-0.5">
          View All &rarr;
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-[#DCE5EC] text-[10px] uppercase font-bold text-[#64748B]">
              <th className="py-1.5 px-1 font-semibold">Region</th>
              <th className="py-1.5 px-1 text-right font-semibold">Confidence</th>
              <th className="py-1.5 px-1 text-right font-semibold">Bust Probability</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#F4F7FA]">
            {stats.map((row) => (
              <tr key={row.name} className="hover:bg-[#F8FAFC] transition-colors">
                <td className="py-2 px-1 flex items-center gap-2">
                  <span
                    className="w-1 h-3.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: row.color }}
                  ></span>
                  <span className="text-xs font-semibold text-[#102A43]">{row.name}</span>
                </td>
                <td className="py-2 px-1 text-right text-xs font-mono font-medium text-[#102A43]">
                  {row.confidence}%
                </td>
                <td
                  className="py-2 px-1 text-right text-xs font-mono font-bold"
                  style={{ color: row.bust_prob > 0.5 ? '#D94B55' : row.bust_prob > 0.3 ? '#D99A28' : '#26966F' }}
                >
                  {Math.round(row.bust_prob * 100)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
