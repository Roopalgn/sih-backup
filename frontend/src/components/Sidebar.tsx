import React from 'react';
import { Calendar, Clock, Wind, Globe, Grid3X3, Droplets, Database, AlertCircle, CheckCircle } from 'lucide-react';
import type { ShowcaseEvent } from '../types';

interface SidebarProps {
  showcaseEvents: ShowcaseEvent[];
  selectedEventName: string;
  onSelectEvent: (name: string) => void;
  selectedDate: string;
  onDateChange: (date: string) => void;
  leadDay: number;
  onLeadDayChange: (lead: number) => void;
  eventType: string;
  dataSource: string;
}

export const Sidebar: React.FC<SidebarProps> = ({
  showcaseEvents,
  selectedEventName,
  onSelectEvent,
  selectedDate,
  onDateChange,
  leadDay,
  onLeadDayChange,
  eventType,
  dataSource,
}) => {
  const formattedDate = new Date(selectedDate).toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  });

  const isIllustrative = dataSource === 'illustrative_only';

  const cleanEventType = eventType
    ? eventType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    : 'Monsoon Depression';

  return (
    <aside className="w-64 bg-white border-r border-[#DCE5EC] p-4 flex flex-col gap-4 overflow-y-auto flex-shrink-0">
      {/* Control Selector */}
      <div className="bg-[#F4F7FA] border border-[#DCE5EC] p-3 rounded-lg">
        <label className="text-[10px] font-bold text-[#64748B] uppercase tracking-wider block mb-1.5">
          Select Scenario / Run
        </label>
        <select
          value={selectedEventName}
          onChange={(e) => onSelectEvent(e.target.value)}
          className="w-full text-xs font-semibold text-[#102A43] bg-white border border-[#DCE5EC] rounded-md px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-[#1769AA]"
        >
          <option value="CUSTOM RUN">Custom Forecast Date</option>
          {showcaseEvents.map((evt) => (
            <option key={evt.event_name} value={evt.event_name}>
              {evt.event_name}
            </option>
          ))}
        </select>

        {selectedEventName === 'CUSTOM RUN' && (
          <div className="mt-2.5 space-y-2">
            <div>
              <label className="text-[10px] text-[#64748B] block mb-0.5">Init Date</label>
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => onDateChange(e.target.value)}
                min="2020-01-01"
                max="2023-12-31"
                className="w-full text-xs bg-white border border-[#DCE5EC] rounded px-2 py-1 text-[#102A43]"
              />
            </div>
          </div>
        )}
      </div>

      {/* 1. FORECAST RUN */}
      <div className="border-b border-[#DCE5EC] pb-3.5">
        <div className="text-[11px] font-bold text-[#64748B] uppercase tracking-wider mb-2.5">
          FORECAST RUN
        </div>

        <div className="space-y-2.5">
          <div className="flex items-start gap-2.5">
            <Calendar className="w-4 h-4 text-[#1769AA] mt-0.5 flex-shrink-0" />
            <div>
              <div className="text-xs font-bold text-[#102A43]">{formattedDate}</div>
              <div className="text-[10px] text-[#64748B]">Initialisation Date</div>
            </div>
          </div>

          <div className="flex items-start gap-2.5">
            <Clock className="w-4 h-4 text-[#1769AA] mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-[#102A43]">Day {leadDay}</span>
                <span className="text-[10px] text-[#64748B]">Forecast Lead</span>
              </div>
              <div className="flex gap-1 mt-1.5">
                {[3, 5, 7, 10].map((d) => (
                  <button
                    key={d}
                    onClick={() => onLeadDayChange(d)}
                    className={`px-2 py-0.5 rounded text-[10px] font-semibold transition-colors ${
                      leadDay === d
                        ? 'bg-[#1769AA] text-white'
                        : 'bg-[#F4F7FA] text-[#64748B] hover:bg-[#E2E8F0]'
                    }`}
                  >
                    D{d}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="flex items-start gap-2.5">
            <Wind className="w-4 h-4 text-[#1769AA] mt-0.5 flex-shrink-0" />
            <div>
              <div className="text-xs font-bold text-[#102A43]">{cleanEventType}</div>
              <div className="text-[10px] text-[#64748B]">Event Type</div>
            </div>
          </div>
        </div>
      </div>

      {/* 2. DOMAIN */}
      <div className="border-b border-[#DCE5EC] pb-3.5">
        <div className="text-[11px] font-bold text-[#64748B] uppercase tracking-wider mb-2.5">
          DOMAIN
        </div>

        <div className="space-y-2.5">
          <div className="flex items-start gap-2.5">
            <Globe className="w-4 h-4 text-[#1769AA] mt-0.5 flex-shrink-0" />
            <div>
              <div className="text-xs font-bold text-[#102A43]">14&deg;N &ndash; 32&deg;N</div>
              <div className="text-xs font-bold text-[#102A43]">68&deg;E &ndash; 90&deg;E</div>
            </div>
          </div>

          <div className="flex items-start gap-2.5">
            <Grid3X3 className="w-4 h-4 text-[#1769AA] mt-0.5 flex-shrink-0" />
            <div>
              <div className="text-xs font-bold text-[#102A43]">0.25&deg; &times; 0.25&deg;</div>
              <div className="text-[10px] text-[#64748B]">Grid Resolution</div>
            </div>
          </div>

          <div className="flex items-start gap-2.5">
            <Droplets className="w-4 h-4 text-[#1769AA] mt-0.5 flex-shrink-0" />
            <div>
              <div className="text-xs font-bold text-[#102A43]">Rainfall (mm)</div>
              <div className="text-[10px] text-[#64748B]">Variable</div>
            </div>
          </div>
        </div>
      </div>

      {/* 3. DATA SOURCE */}
      <div className="border-b border-[#DCE5EC] pb-3.5">
        <div className="text-[11px] font-bold text-[#64748B] uppercase tracking-wider mb-2.5">
          DATA SOURCE
        </div>

        <div className="flex items-start gap-2.5">
          <Database className="w-4 h-4 text-[#1769AA] mt-0.5 flex-shrink-0" />
          <div className="space-y-1">
            <div className="text-xs font-semibold text-[#102A43]">NOAA GEFSv12 (Forecast)</div>
            <div className="text-xs font-semibold text-[#102A43]">IMD (Observation)</div>
          </div>
        </div>
      </div>

      {/* 4. DATA STATUS CARD */}
      <div>
        {isIllustrative ? (
          <div className="bg-[#FFFBF0] border border-[#FDE68A] rounded-lg p-3">
            <div className="flex items-center gap-1.5 text-[#D99A28] mb-1">
              <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
              <span className="text-[10px] font-bold uppercase tracking-wider">DATA STATUS</span>
            </div>
            <div className="text-xs font-bold text-[#B45309] mb-1">ILLUSTRATIVE EVENT</div>
            <div className="text-[11px] text-[#92400E] leading-relaxed">
              These numbers are placeholder data shown while model training is pending.
            </div>
          </div>
        ) : (
          <div className="bg-[#F0FDF4] border border-[#BBF7D0] rounded-lg p-3">
            <div className="flex items-center gap-1.5 text-[#26966F] mb-1">
              <CheckCircle className="w-3.5 h-3.5 flex-shrink-0" />
              <span className="text-[10px] font-bold uppercase tracking-wider">DATA STATUS</span>
            </div>
            <div className="text-xs font-bold text-[#166534] mb-1">MODEL OUTPUT</div>
            <div className="text-[11px] text-[#166534] leading-relaxed">
              Live inference from ForecastBustUNet on NOAA GEFSv12.
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};
