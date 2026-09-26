import React from 'react';
import { HelpCircle, AlertCircle, Cpu } from 'lucide-react';
import type { MeteoDriver } from '../types';

interface ExplainabilityPanelProps {
  drivers: MeteoDriver[];
  interpretation?: string;
  leadDay: number;
}

export const ExplainabilityPanel: React.FC<ExplainabilityPanelProps> = ({
  drivers,
  interpretation,
  leadDay,
}) => {
  const maxPct = drivers.length > 0 ? Math.max(...drivers.map((d) => d.attribution_pct), 1) : 1;

  return (
    <div className="bg-white rounded-lg border border-[#DCE5EC] p-4 flex flex-col justify-between h-full shadow-[0_1px_3px_rgba(16,42,67,0.05)]">
      <div>
        {/* Header */}
        <div className="flex items-center justify-between pb-3 mb-3 border-b border-[#DCE5EC]">
          <div>
            <h3 className="text-xs font-bold text-[#102A43] tracking-wider uppercase flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-[#1769AA]" />
              Why Is Confidence Low? (Day {leadDay})
            </h3>
            <p className="text-[11px] text-[#64748B]">
              Integrated Gradients attribution on dual-head UNet encoder
            </p>
          </div>
          <div className="flex items-center gap-1 text-[11px] text-[#64748B] bg-[#F4F7FA] px-2 py-0.5 rounded border border-[#DCE5EC]">
            <HelpCircle className="w-3 h-3 text-[#1769AA]" />
            <span>P90 Attribution</span>
          </div>
        </div>

        {/* Contribution Bars */}
        {drivers.length === 0 ? (
          <div className="py-8 text-center text-xs text-[#64748B]">
            Explanation unavailable for this forecast.
          </div>
        ) : (
          <div className="space-y-2.5 my-2">
            {drivers.map((driver, idx) => {
              const widthPct = Math.round((driver.attribution_pct / maxPct) * 100);
              const isTop = idx === 0;

              return (
                <div key={driver.channel_name || idx} className="space-y-1">
                  <div className="flex justify-between items-center text-xs">
                    <span
                      className={`truncate max-w-[210px] ${
                        isTop ? 'font-semibold text-[#102A43]' : 'text-[#334E68]'
                      }`}
                      title={driver.description || driver.channel_name}
                    >
                      {driver.channel_name}
                    </span>
                    <span
                      className={`font-mono text-[11px] ${
                        isTop ? 'font-bold text-[#D94B55]' : 'text-[#64748B]'
                      }`}
                    >
                      +{driver.attribution_pct.toFixed(1)}%
                    </span>
                  </div>

                  <div className="w-full bg-[#E2E8F0] h-2 rounded-full overflow-hidden flex">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        isTop
                          ? 'bg-[#D94B55]'
                          : idx === 1
                          ? 'bg-[#E53E3E]'
                          : idx === 2
                          ? 'bg-[#DD6B20]'
                          : 'bg-[#1769AA]'
                      }`}
                      style={{ width: `${Math.min(widthPct, 100)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Synoptic Meteorological Interpretation */}
      {interpretation && (
        <div className="mt-3 pt-3 border-t border-[#DCE5EC]">
          <div className="bg-[#FFFBEB] border border-[#FDE68A] rounded-md p-2.5 text-xs text-[#92400E] flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-[#D97706] shrink-0 mt-0.5" />
            <div className="leading-relaxed">
              <span className="font-semibold block text-[#78350F] mb-0.5">
                Synoptic Interpretation:
              </span>
              <p className="text-[11px] text-[#92400E]">{interpretation}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
