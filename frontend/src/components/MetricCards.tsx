import React from 'react';
import { BarChart3, AlertTriangle, Waves, Clock } from 'lucide-react';

interface MetricCardsProps {
  confidence: number;
  bustProb: number;
  expectedError: number;
  leadDay: number;
}

export const MetricCards: React.FC<MetricCardsProps> = ({
  confidence,
  bustProb,
  expectedError,
  leadDay,
}) => {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5 mb-4">
      {/* 1. FORECAST CONFIDENCE */}
      <div className="bg-white border border-[#DCE5EC] rounded-lg p-3.5 flex items-center justify-between shadow-card hover:shadow-md transition-shadow">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-[#EAF8F1] flex items-center justify-center flex-shrink-0 text-[#26966F]">
            <BarChart3 className="w-5 h-5" />
          </div>
          <div>
            <div className="text-[10px] font-bold text-[#64748B] uppercase tracking-wider">
              FORECAST CONFIDENCE
            </div>
            <div className="text-[26px] font-bold text-[#102A43] tracking-tight leading-tight mt-0.5 font-sans">
              {confidence.toFixed(1)}%
            </div>
            <div className="text-[11px] font-semibold flex items-center gap-1">
              <span className={confidence >= 70 ? 'text-[#26966F]' : (confidence >= 40 ? 'text-[#D99A28]' : 'text-[#D94B55]')}>
                {confidence >= 70 ? '● High Confidence' : (confidence >= 40 ? '▲ Moderate Confidence' : '▼ Low Confidence')}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 2. BUST PROBABILITY */}
      <div className="bg-white border border-[#DCE5EC] rounded-lg p-3.5 flex items-center justify-between shadow-card hover:shadow-md transition-shadow">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-[#FDE8E8] flex items-center justify-center flex-shrink-0 text-[#D94B55]">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div>
            <div className="text-[10px] font-bold text-[#64748B] uppercase tracking-wider">
              BUST PROBABILITY
            </div>
            <div className="text-[26px] font-bold text-[#102A43] tracking-tight leading-tight mt-0.5 font-sans">
              {(bustProb * 100).toFixed(1)}%
            </div>
            <div className="text-[11px] font-semibold flex items-center gap-1">
              <span className={bustProb >= 0.5 ? 'text-[#D94B55]' : (bustProb >= 0.3 ? 'text-[#D99A28]' : 'text-[#26966F]')}>
                {bustProb >= 0.5 ? '▲ P90 Bust Alert' : (bustProb >= 0.3 ? '▲ Elevated Risk' : '● Low Bust Risk')}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 3. EXPECTED ERROR */}
      <div className="bg-white border border-[#DCE5EC] rounded-lg p-3.5 flex items-center justify-between shadow-card hover:shadow-md transition-shadow">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-[#E8F2FA] flex items-center justify-center flex-shrink-0 text-[#1769AA]">
            <Waves className="w-5 h-5" />
          </div>
          <div>
            <div className="text-[10px] font-bold text-[#64748B] uppercase tracking-wider">
              EXPECTED ERROR
            </div>
            <div className="text-[26px] font-bold text-[#102A43] tracking-tight leading-tight mt-0.5 font-sans">
              {expectedError.toFixed(1)} mm
            </div>
            <div className="text-[11px] font-semibold text-[#1769AA] flex items-center gap-1">
              <span>Domain Spatial MAE</span>
            </div>
          </div>
        </div>
      </div>

      {/* 4. FORECAST HORIZON */}
      <div className="bg-white border border-[#DCE5EC] rounded-lg p-3.5 flex items-center justify-between shadow-card hover:shadow-md transition-shadow">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-[#EAEFF5] flex items-center justify-center flex-shrink-0 text-[#123B6D]">
            <Clock className="w-5 h-5" />
          </div>
          <div>
            <div className="text-[10px] font-bold text-[#64748B] uppercase tracking-wider">
              FORECAST HORIZON
            </div>
            <div className="text-[26px] font-bold text-[#102A43] tracking-tight leading-tight mt-0.5 font-sans">
              Day {leadDay}
            </div>
            <div className="text-[11px] font-semibold text-[#64748B]">
              {leadDay * 24}-hour lead time
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
