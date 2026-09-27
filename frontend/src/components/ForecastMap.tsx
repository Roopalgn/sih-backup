import React, { useEffect, useRef } from 'react';
import L from 'leaflet';

interface ForecastMapProps {
  lats: number[];
  lons: number[];
  confidenceMap: number[][];
  bustMap: number[][];
  errorMap: number[][];
  mode?: 'confidence' | 'bust' | 'error';
  dataSource?: string;
}

export const ForecastMap: React.FC<ForecastMapProps> = ({
  lats,
  lons,
  confidenceMap,
  bustMap,
  errorMap,
  mode = 'confidence',
  dataSource = 'live_model',
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const layerGroupRef = useRef<L.LayerGroup | null>(null);
  const selectedMode = mode;

  // Initialize Map Once
  useEffect(() => {
    if (!mapContainerRef.current) return;
    if (mapInstanceRef.current) return;

    // Centered on Central/Northern India
    const map = L.map(mapContainerRef.current, {
      center: [23.5, 79.5],
      zoom: 5.2,
      minZoom: 4,
      maxZoom: 10,
      zoomControl: true,
      attributionControl: true,
    });

    // Legitimate, watermark-free OpenStreetMap basemap with complete geographic context and state boundaries
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 18,
      subdomains: ['a', 'b', 'c'],
      attribution: '&copy; OpenStreetMap contributors',
    }).addTo(map);

    const layerGroup = L.layerGroup().addTo(map);
    mapInstanceRef.current = map;
    layerGroupRef.current = layerGroup;

    // Fit to actual domain bounds (14°N–32°N, 68°E–90°E) with appropriate operational padding
    map.fitBounds([
      [14.0, 68.0],
      [32.0, 90.0],
    ], { padding: [15, 15] });

    // Handle container resizing smoothly
    const resizeObserver = new ResizeObserver(() => {
      map.invalidateSize();
    });
    resizeObserver.observe(mapContainerRef.current);

    return () => {
      resizeObserver.disconnect();
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update Grid Overlay
  useEffect(() => {
    if (!layerGroupRef.current || !lats || !lons || !confidenceMap) return;

    const layerGroup = layerGroupRef.current;
    layerGroup.clearLayers();

    const H = lats.length;
    const W = lons.length;
    if (H === 0 || W === 0) return;

    const errors = errorMap.flat().filter(Number.isFinite).sort((a, b) => a - b);
    const errorQuantile = (q: number) => errors[Math.min(errors.length - 1, Math.floor((errors.length - 1) * q))] ?? 0;
    const errP25 = errorQuantile(0.25);
    const errP50 = errorQuantile(0.50);
    const errP75 = errorQuantile(0.75);

    const step = 2; // Render at 0.50° resolution for high performance
    const cellDeg = 0.25 * step;

    for (let i = 0; i < H; i += step) {
      const lat = lats[i];
      for (let j = 0; j < W; j += step) {
        const lon = lons[j];
        const confVal = confidenceMap[i]?.[j];
        const bustVal = bustMap[i]?.[j];
        const errVal = errorMap?.[i]?.[j];
        // Unsupported or masked grid cells stay transparent instead of being
        // painted with plausible-looking default predictions.
        if (typeof confVal !== 'number' || !Number.isFinite(confVal) ||
            typeof bustVal !== 'number' || !Number.isFinite(bustVal) ||
            typeof errVal !== 'number' || !Number.isFinite(errVal)) continue;

        let fillColor = '#26966F';
        let fillOpacity = 0.65;

        if (selectedMode === 'confidence') {
          if (confVal >= 80) fillColor = '#45bcae';
          else if (confVal >= 40) fillColor = '#dfb263';
          else fillColor = '#ed776e';
        } else if (selectedMode === 'bust') {
          if (bustVal >= 0.6) fillColor = '#ed776e';
          else if (bustVal >= 0.35) fillColor = '#dfb263';
          else fillColor = '#45bcae';
        } else {
          if (errVal <= 0) fillColor = '#315f72';
          else if (errVal >= errP75) fillColor = '#ed776e';
          else if (errVal >= errP50) fillColor = '#dfb263';
          else if (errVal >= errP25) fillColor = '#66aaa7';
          else fillColor = '#315f72';
        }

        const rect = L.rectangle(
          [
            [lat - cellDeg / 2, lon - cellDeg / 2],
            [lat + cellDeg / 2, lon + cellDeg / 2],
          ],
          {
            color: 'transparent',
            weight: 0,
            fillColor,
            fillOpacity,
          }
        );

        rect.bindTooltip(
          `
          <div style="font-family:'DM Sans',sans-serif;font-size:11px;color:#e5eff0;padding:2px;">
            <div style="font-weight:700;color:#8ed1cd;margin-bottom:2px;">
              ${lat.toFixed(2)}&deg;N, ${lon.toFixed(2)}&deg;E
            </div>
            <div>Confidence: <strong style="color:${confVal >= 80 ? '#26966F' : (confVal >= 40 ? '#D99A28' : '#D94B55')}">${confVal.toFixed(1)}%</strong></div>
            <div>Bust Probability: <strong>${(bustVal * 100).toFixed(1)}%</strong></div>
            <div>Predicted Error: <strong>${errVal.toFixed(1)} mm/d</strong></div>
          </div>
          `,
          { sticky: true, className: 'custom-map-tooltip' }
        );

        layerGroup.addLayer(rect);
      }
    }

    // Add Subdivision reference boxes
    const subBoxes = [
      { name: 'Odisha', bounds: [[17.0, 82.0], [22.0, 88.0]] },
      { name: 'Gangetic West Bengal', bounds: [[21.0, 85.0], [25.0, 90.0]] },
      { name: 'Konkan & Goa', bounds: [[14.0, 72.0], [20.0, 76.0]] },
      { name: 'Northwest India', bounds: [[26.0, 68.0], [32.0, 78.0]] },
    ];

    subBoxes.forEach((sub) => {
      const bRect = L.rectangle(sub.bounds as L.LatLngBoundsExpression, {
        color: '#123B6D',
        weight: 1.2,
        dashArray: '3, 4',
        fill: false,
      });
      bRect.bindTooltip(`${sub.name} (IMD Subdivision)`, { sticky: true });
      layerGroup.addLayer(bRect);
    });

  }, [lats, lons, confidenceMap, bustMap, errorMap, selectedMode]);

  return (
    <div className="map-widget">
      <div className="relative w-full h-[440px]">
        <div ref={mapContainerRef} className="w-full h-full" />

        {/* Map Provenance Badge directly on map */}
        <div className="absolute top-3 left-14 z-[1000] pointer-events-none">
          {dataSource === 'live_model' ? (
            <div className="bg-[#E6F4EA]/95 border border-[#CEEAD6] text-[#137333] px-3 py-1 rounded-full text-[11px] font-bold shadow-sm flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#137333]"></span>
              <span>📍 Live model output (ForecastBustUNet)</span>
            </div>
          ) : dataSource === 'precomputed_cache' ? (
            <div className="bg-[#E8F0FE]/95 border border-[#D2E3FC] text-[#1A73E8] px-3 py-1 rounded-full text-[11px] font-bold shadow-sm flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#1A73E8]"></span>
              <span>Saved output · model lineage unverified</span>
            </div>
          ) : (
            <div className="bg-[#FEF7E0]/95 border border-[#FDD663] text-[#B06000] px-3 py-1 rounded-full text-[11px] font-bold shadow-sm flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#B06000]"></span>
              <span>⚠️ Illustrative pattern — NOT model output</span>
            </div>
          )}
        </div>

        {/* The parent dashboard shows the confidence legend beside the map. */}
        <div className="absolute bottom-4 left-4 z-[1000] bg-white/95 backdrop-blur-sm border border-[#DCE5EC] rounded-lg p-2.5 px-3.5 shadow-md">
          <div className="text-[11px] font-bold text-[#102A43] mb-1.5">
            {selectedMode === 'confidence' ? 'Forecast Confidence' : selectedMode === 'bust' ? 'Bust Probability' : 'Predicted Error'}
          </div>
          {selectedMode === 'confidence' ? (
            <div className="space-y-1 text-[11px] text-[#526777]">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-sm bg-[#26966F]"></span>
                <span>High (&ge; 80%)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-sm bg-[#D99A28]"></span>
                <span>Moderate (40 &ndash; 80%)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-sm bg-[#D94B55]"></span>
                <span>Low (&le; 40%)</span>
              </div>
            </div>
          ) : selectedMode === 'bust' ? (
            <div className="space-y-1 text-[11px] text-[#526777]">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-sm bg-[#D94B55]"></span>
                <span>High Risk (&ge; 60%)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-sm bg-[#D99A28]"></span>
                <span>Moderate Risk (35 &ndash; 60%)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-sm bg-[#26966F]"></span>
                <span>Low Risk (&lt; 35%)</span>
              </div>
            </div>
          ) : <div className="space-y-1 text-[11px] text-[#526777]"><div>Relative error classes</div><div>Units: mm/day</div></div>}
        </div>
      </div>

      {/* Per-Map Provenance Caption */}
      <div className="px-4 py-2 bg-[#F8FAFC] border-t border-[#DCE5EC] text-[11px] text-[#64748B] flex items-center justify-between">
        <div>
          {dataSource === 'live_model' ? (
            <span>📍 <strong>Map source: Live model output</strong> — real GEFS forecast processed by ForecastBustUNet.</span>
          ) : dataSource === 'precomputed_cache' ? (
            <span>📦 <strong>Map source: Saved output</strong> — model lineage is not bundled for independent verification.</span>
          ) : (
            <span className="text-[#B06000]">⚠️ <strong>Map source: Illustrative pattern</strong> — NOT model output. Generated locally for layout only.</span>
          )}
        </div>
        <div className="text-[10px] text-[#94A3B8]">Domain: 14&deg;N&ndash;32&deg;N, 68&deg;E&ndash;90&deg;E</div>
      </div>
    </div>
  );
};
