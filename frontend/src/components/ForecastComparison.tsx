import React from 'react';

interface ForecastComparisonProps {
  leadDay: number;
  rmse: number;
  mae: number;
  bias: string;
}

export const ForecastComparison: React.FC<ForecastComparisonProps> = ({
  leadDay,
  rmse,
  mae,
  bias,
}) => {
  return (
    <div className="bg-white border border-[#DCE5EC] rounded-lg p-4 shadow-card h-full flex flex-col justify-between">
      <div className="text-[12px] font-bold text-[#102A43] tracking-wide uppercase mb-2">
        FORECAST VS OBSERVATION (Day {leadDay})
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-center">
        {/* Plot 1: Model Forecast (mm) */}
        <div className="flex flex-col items-center">
          <div className="text-[11px] font-semibold text-[#526777] mb-1.5">Model Forecast (mm)</div>
          <div className="relative w-full h-[120px] flex items-center justify-center">
            {/* India Shape Contour Simulation */}
            <svg viewBox="0 0 100 120" className="w-[85px] h-[105px]">
              <defs>
                <linearGradient id="fcstGrad" x1="0%" y1="100%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#38BDF8" />
                  <stop offset="35%" stopColor="#4ADE80" />
                  <stop offset="70%" stopColor="#FACC15" />
                  <stop offset="100%" stopColor="#EF4444" />
                </linearGradient>
              </defs>
              <path
                d="M48 5 L55 18 L68 28 L62 38 L78 45 L72 58 L85 62 L78 72 L62 82 L55 105 L48 115 L42 105 L35 85 L22 75 L15 55 L25 42 L38 32 Z"
                fill="url(#fcstGrad)"
                stroke="#DCE5EC"
                strokeWidth="0.8"
                opacity="0.9"
              />
            </svg>
            {/* Colorbar */}
            <div className="flex items-center gap-1 text-[9px] text-[#718596] font-mono ml-2">
              <div className="w-1.5 h-16 rounded-sm bg-gradient-to-t from-[#38BDF8] via-[#FACC15] to-[#EF4444]" />
              <div className="flex flex-col justify-between h-16">
                <span>200</span>
                <span>100</span>
                <span>50</span>
                <span>10</span>
                <span>0</span>
              </div>
            </div>
          </div>
        </div>

        {/* Plot 2: Observed Rainfall (mm) */}
        <div className="flex flex-col items-center">
          <div className="text-[11px] font-semibold text-[#526777] mb-1.5">Observed Rainfall (mm)</div>
          <div className="relative w-full h-[120px] flex items-center justify-center">
            <svg viewBox="0 0 100 120" className="w-[85px] h-[105px]">
              <defs>
                <linearGradient id="obsGrad" x1="10%" y1="90%" x2="90%" y2="10%">
                  <stop offset="0%" stopColor="#60A5FA" />
                  <stop offset="40%" stopColor="#34D399" />
                  <stop offset="75%" stopColor="#FBBF24" />
                  <stop offset="100%" stopColor="#DC2626" />
                </linearGradient>
              </defs>
              <path
                d="M48 5 L55 18 L68 28 L62 38 L78 45 L72 58 L85 62 L78 72 L62 82 L55 105 L48 115 L42 105 L35 85 L22 75 L15 55 L25 42 L38 32 Z"
                fill="url(#obsGrad)"
                stroke="#DCE5EC"
                strokeWidth="0.8"
                opacity="0.9"
              />
            </svg>
            {/* Colorbar */}
            <div className="flex items-center gap-1 text-[9px] text-[#718596] font-mono ml-2">
              <div className="w-1.5 h-16 rounded-sm bg-gradient-to-t from-[#60A5FA] via-[#FBBF24] to-[#DC2626]" />
              <div className="flex flex-col justify-between h-16">
                <span>200</span>
                <span>100</span>
                <span>50</span>
                <span>10</span>
                <span>0</span>
              </div>
            </div>
          </div>
        </div>

        {/* Plot 3: Forecast Error (mm) */}
        <div className="flex flex-col items-center">
          <div className="text-[11px] font-semibold text-[#526777] mb-1.5">Forecast Error (mm)</div>
          <div className="relative w-full h-[120px] flex items-center justify-center">
            <svg viewBox="0 0 100 120" className="w-[85px] h-[105px]">
              <defs>
                <linearGradient id="errGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#EF4444" />
                  <stop offset="50%" stopColor="#FFFFFF" />
                  <stop offset="100%" stopColor="#3B82F6" />
                </linearGradient>
              </defs>
              <path
                d="M48 5 L55 18 L68 28 L62 38 L78 45 L72 58 L85 62 L78 72 L62 82 L55 105 L48 115 L42 105 L35 85 L22 75 L15 55 L25 42 L38 32 Z"
                fill="url(#errGrad)"
                stroke="#DCE5EC"
                strokeWidth="0.8"
                opacity="0.9"
              />
            </svg>
            {/* Diverging Colorbar */}
            <div className="flex items-center gap-1 text-[9px] text-[#718596] font-mono ml-2">
              <div className="w-1.5 h-16 rounded-sm bg-gradient-to-t from-[#3B82F6] via-[#FFFFFF] to-[#EF4444]" />
              <div className="flex flex-col justify-between h-16">
                <span>100</span>
                <span>50</span>
                <span>0</span>
                <span>-50</span>
                <span>-100</span>
              </div>
            </div>
          </div>
        </div>

        {/* Metrics Box (India Domain) */}
        <div className="bg-[#F8FAFC] border border-[#DCE5EC] rounded-lg p-3 flex flex-col justify-center">
          <div className="text-[10px] font-bold text-[#64748B] uppercase tracking-wider mb-2.5">
            Metrics (India Domain)
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between items-center">
              <span className="text-[#64748B]">RMSE</span>
              <span className="font-mono font-bold text-[#102A43]">{rmse} mm</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[#64748B]">MAE</span>
              <span className="font-mono font-bold text-[#102A43]">{mae} mm</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[#64748B]">BIAS</span>
              <span className="font-mono font-bold text-[#26966F]">{bias} mm</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
