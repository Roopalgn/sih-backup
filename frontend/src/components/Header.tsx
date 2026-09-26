import React from 'react';
import { LayoutGrid, TrendingUp, CheckCircle2, ScanSearch, Database } from 'lucide-react';
import type { NavTab } from '../types';

interface HeaderProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  apiOnline: boolean;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, onTabChange, apiOnline }) => {
  const currentDateStr = new Date().toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  });
  const currentTimeStr = new Date().toLocaleTimeString('en-GB', {
    hour: '2-digit',
    minute: '2-digit',
  });

  return (
    <header className="bg-white border-b border-[#DCE5EC] px-6 py-3 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-30 shadow-[0_1px_3px_rgba(16,42,67,0.04)]">
      {/* Left: Emblem & Institutional Title */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3">
          {/* Government of India Emblem SVG */}
          <div className="w-10 h-10 rounded-full bg-[#F4F7FA] border border-[#DCE5EC] flex items-center justify-center p-1.5 flex-shrink-0">
            <svg viewBox="0 0 24 24" fill="none" stroke="#123B6D" strokeWidth="1.8" className="w-full h-full">
              <circle cx="12" cy="12" r="9" />
              <path d="M12 3v18" />
              <path d="M3 12h18" />
              <circle cx="12" cy="12" r="4" />
            </svg>
          </div>
          <div className="border-r border-[#DCE5EC] pr-4">
            <div className="text-[11px] font-bold text-[#123B6D] tracking-wide leading-tight">Ministry of Earth Sciences</div>
            <div className="text-[10px] text-[#64748B] leading-tight">Government of India</div>
            <div className="text-[10px] text-[#94A3B8] leading-tight">पृथ्वी विज्ञान मंत्रालय</div>
          </div>
        </div>

        <div>
          <div className="text-[20px] font-extrabold text-[#102A43] tracking-tight leading-none">NCMRWF</div>
          <div className="text-[12px] font-semibold text-[#526777] leading-tight mt-0.5">National Centre for Medium Range Weather Forecasting</div>
          <div className="text-[11px] text-[#94A3B8] leading-tight">AI-Based Forecast Bust Detection System &nbsp;|&nbsp; SIH #26079</div>
        </div>
      </div>

      {/* Center: Navigation Pill Tabs */}
      <nav className="flex items-center bg-[#F4F7FA] border border-[#DCE5EC] p-1 rounded-lg gap-1">
        <button
          onClick={() => onTabChange('overview')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
            activeTab === 'overview'
              ? 'bg-[#1769AA] text-white shadow-sm'
              : 'text-[#526777] hover:text-[#102A43] hover:bg-white/60'
          }`}
        >
          <LayoutGrid className="w-3.5 h-3.5" />
          <span>Overview</span>
        </button>

        <button
          onClick={() => onTabChange('forecast')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
            activeTab === 'forecast'
              ? 'bg-[#1769AA] text-white shadow-sm'
              : 'text-[#526777] hover:text-[#102A43] hover:bg-white/60'
          }`}
        >
          <TrendingUp className="w-3.5 h-3.5" />
          <span>Forecast</span>
        </button>

        <button
          onClick={() => onTabChange('error_analysis')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
            activeTab === 'error_analysis'
              ? 'bg-[#1769AA] text-white shadow-sm'
              : 'text-[#526777] hover:text-[#102A43] hover:bg-white/60'
          }`}
        >
          <CheckCircle2 className="w-3.5 h-3.5" />
          <span>Error Analysis</span>
        </button>

        <button
          onClick={() => onTabChange('explainability')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
            activeTab === 'explainability'
              ? 'bg-[#1769AA] text-white shadow-sm'
              : 'text-[#526777] hover:text-[#102A43] hover:bg-white/60'
          }`}
        >
          <ScanSearch className="w-3.5 h-3.5" />
          <span>Explainability</span>
        </button>
      </nav>

      {/* Right: Model Info & System Status */}
      <div className="flex items-center gap-5">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-[#1769AA]" />
          <div>
            <div className="text-[10px] text-[#94A3B8] uppercase font-bold tracking-wider">Model</div>
            <div className="text-[12px] font-bold text-[#102A43] leading-tight">GEFSv12</div>
            <div className="text-[10px] text-[#64748B] font-mono leading-tight">0.25&deg; Grid</div>
          </div>
        </div>

        <div className="border-l border-[#DCE5EC] pl-4">
          <div className="text-[10px] text-[#94A3B8] uppercase font-bold tracking-wider">System Status</div>
          <div className="text-[12px] font-bold text-[#26966F] flex items-center gap-1.5 leading-tight">
            <span className={`w-2 h-2 rounded-full ${apiOnline ? 'bg-[#26966F] animate-pulse' : 'bg-[#D99A28]'}`}></span>
            {apiOnline ? 'Ready' : 'Cache Mode'}
          </div>
          <div className="text-[10px] text-[#94A3B8] leading-tight mt-0.5">
            Last updated<br />{currentDateStr}, {currentTimeStr} IST
          </div>
        </div>
      </div>
    </header>
  );
};
