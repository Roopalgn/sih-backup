import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { MetricCards } from './components/MetricCards';
import { ForecastMap } from './components/ForecastMap';
import { RegionalRiskTable } from './components/RegionalRiskTable';
import { LeadTimeChart } from './components/LeadTimeChart';
import { ForecastComparison } from './components/ForecastComparison';
import { ExplainabilityPanel } from './components/ExplainabilityPanel';
import { SHOWCASE_EVENTS } from './api/showcaseData';
import {
  fetchPrediction,
  calculateSubdivisionStats,
  calculateDomainMetrics,
  checkApiHealth,
  loadEventsCatalog,
} from './api/client';
import type { PredictResponse, NavTab, SubdivisionStat } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavTab>('overview');
  const [isOnline, setIsOnline] = useState<boolean>(true);
  const [selectedEventName, setSelectedEventName] = useState<string>('Monsoon Onset 2022');
  const [selectedDate, setSelectedDate] = useState<string>('2022-06-25');
  const [leadDay, setLeadDay] = useState<number>(3);
  const [currentData, setCurrentData] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Check backend health periodically
  useEffect(() => {
    checkApiHealth().then(setIsOnline);
    const interval = setInterval(() => {
      checkApiHealth().then(setIsOnline);
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  // Pre-load events catalog
  useEffect(() => {
    loadEventsCatalog();
  }, []);

  // Fetch prediction on event, date, or leadDay change
  useEffect(() => {
    let isCancelled = false;
    setLoading(true);
    setError(null);

    fetchPrediction({
      date: selectedDate,
      lead_day: leadDay,
      variable: 'rainfall',
    })
      .then((data) => {
        if (!isCancelled) {
          setCurrentData(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!isCancelled) {
          console.error('Fetch error:', err);
          setError(err.message || 'Failed to load forecast data');
          setLoading(false);
        }
      });

    return () => {
      isCancelled = true;
    };
  }, [selectedDate, leadDay]);

  const handleSelectEvent = (eventName: string) => {
    setSelectedEventName(eventName);
    if (eventName === 'CUSTOM RUN') {
      return;
    }
    const matched = SHOWCASE_EVENTS.find((e) => e.event_name === eventName);
    if (matched) {
      setSelectedDate(matched.request_date);
      setLeadDay(matched.lead_day);
    }
  };

  // Derive regional subdivision stats from active grids
  const subdivisionStats: SubdivisionStat[] = React.useMemo(() => {
    if (!currentData?.confidence_map || !currentData.grid_latitudes) {
      return calculateSubdivisionStats([], [], [], [], []);
    }
    return calculateSubdivisionStats(
      currentData.confidence_map,
      currentData.bust_probability_map,
      currentData.error_magnitude_map,
      currentData.grid_latitudes,
      currentData.grid_longitudes
    );
  }, [currentData]);

  // Derive domain metrics (RMSE, MAE, BIAS) from active error map
  const domainMetrics = React.useMemo(() => {
    return calculateDomainMetrics(currentData?.error_magnitude_map ?? []);
  }, [currentData?.error_magnitude_map]);

  return (
    <div className="flex flex-col h-screen bg-[#F4F7FA] overflow-hidden text-[#102A43]">
      {/* 1. NCMRWF Government Header */}
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        apiOnline={isOnline}
      />

      {/* 2. Main Layout: Left Sidebar + Central Dashboard */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Operational Sidebar */}
        <Sidebar
          showcaseEvents={SHOWCASE_EVENTS}
          selectedEventName={selectedEventName}
          onSelectEvent={handleSelectEvent}
          selectedDate={selectedDate}
          onDateChange={setSelectedDate}
          leadDay={leadDay}
          onLeadDayChange={setLeadDay}
          eventType={currentData?.event_type || 'monsoon_depression'}
          dataSource={currentData?.data_source || 'live_model'}
        />

        {/* Central Operational Viewport */}
        <main className="flex-1 overflow-y-auto p-4 flex flex-col gap-4 relative">
          {/* Subtle loading indicator */}
          {loading && (
            <div className="absolute top-2 right-4 z-40 bg-[#1769AA] text-white text-[11px] font-semibold px-2.5 py-1 rounded shadow-md flex items-center gap-1.5 animate-pulse">
              <span className="w-2 h-2 rounded-full bg-white animate-ping"></span>
              Updating forecast run...
            </div>
          )}

          {/* Operational Error Notice */}
          {error && (
            <div className="bg-[#FEF2F2] border border-[#FCA5A5] px-3.5 py-2 rounded-md text-xs text-[#991B1B] flex items-center justify-between">
              <span>{error}</span>
              <button
                onClick={() => setError(null)}
                className="font-bold underline ml-2 hover:text-black"
              >
                Dismiss
              </button>
            </div>
          )}

          {/* Top 4 Compact Metric Cards */}
          <MetricCards
            confidence={currentData?.mean_confidence ?? 66.1}
            bustProb={currentData?.mean_bust_probability ?? 0.275}
            expectedError={domainMetrics.mae}
            leadDay={leadDay}
          />

          {/* Tab Filter Notice if not on Overview */}
          {activeTab !== 'overview' && (
            <div className="bg-[#EBF4FA] border border-[#CBDDE9] px-3.5 py-1.5 rounded-md text-xs text-[#1769AA] flex items-center justify-between">
              <span>
                Filtered View: <strong className="uppercase">{activeTab.replace('_', ' ')}</strong> mode active.
              </span>
              <button
                onClick={() => setActiveTab('overview')}
                className="font-semibold underline hover:text-[#102A43]"
              >
                Reset to Full Overview
              </button>
            </div>
          )}

          {/* Main Grid: Leaflet Map (68% width on desktop) + Regional Risk & Reliability (32%) */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            {/* Map Section */}
            <div
              className={`${
                activeTab === 'error_analysis' ? 'hidden' : 'lg:col-span-8'
              } flex flex-col`}
            >
              <ForecastMap
                lats={currentData?.grid_latitudes ?? []}
                lons={currentData?.grid_longitudes ?? []}
                confidenceMap={currentData?.confidence_map ?? []}
                bustMap={currentData?.bust_probability_map ?? []}
                errorMap={currentData?.error_magnitude_map ?? []}
                leadDay={leadDay}
                onLeadDayChange={setLeadDay}
                mode={activeTab === 'forecast' ? 'confidence' : 'confidence'}
              />
            </div>

            {/* Right Side Operational Panels */}
            <div
              className={`${
                activeTab === 'forecast' ? 'hidden' : activeTab === 'error_analysis' ? 'lg:col-span-12' : 'lg:col-span-4'
              } flex flex-col gap-4`}
            >
              <RegionalRiskTable stats={subdivisionStats} />
              <LeadTimeChart
                currentConfidence={currentData?.mean_confidence ?? 66.1}
                currentBustProb={currentData?.mean_bust_probability ?? 0.275}
                currentLeadDay={leadDay}
              />
            </div>
          </div>

          {/* Bottom Grid: Forecast Comparison (68%) + Explainability (32%) */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 pb-2">
            <div
              className={`${
                activeTab === 'explainability' ? 'hidden' : 'lg:col-span-8'
              }`}
            >
              <ForecastComparison
                leadDay={leadDay}
                rmse={domainMetrics.rmse}
                mae={domainMetrics.mae}
                bias={domainMetrics.bias}
              />
            </div>

            <div
              className={`${
                activeTab === 'forecast' ? 'hidden' : 'lg:col-span-4'
              }`}
            >
              <ExplainabilityPanel
                drivers={currentData?.top_drivers ?? []}
                interpretation={
                  currentData?.description ||
                  "Active Bay of Bengal depression induces anomalous low-level moisture convergence. The GEFS ensemble exhibits high intra-member spread along coastal zones, generating large P90 bust probabilities over Gangetic West Bengal and coastal Odisha."
                }
                leadDay={leadDay}
              />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;
