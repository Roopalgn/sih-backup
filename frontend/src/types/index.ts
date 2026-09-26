export interface MeteoDriver {
  rank: number;
  channel_name: string;
  attribution_pct: number;
  description: string;
}

export interface BustRegion {
  name: string;
  lat_center: number;
  lon_center: number;
  mean_bust_prob: number;
  area_fraction: number;
}

export interface PredictRequest {
  date: string;
  lead_day: number;
  variable: string;
  use_cache?: boolean;
}

export interface PredictResponse {
  request_date: string;
  lead_day: number;
  variable: string;
  grid_latitudes: number[];
  grid_longitudes: number[];
  bust_probability_map: number[][];
  error_magnitude_map: number[][];
  confidence_map: number[][];
  mean_bust_probability: number;
  mean_confidence: number;
  high_bust_regions: BustRegion[];
  top_drivers: MeteoDriver[];
  event_type?: string | null;
  data_source: 'live_model' | 'precomputed_cache' | 'illustrative_only' | string;
  from_cache?: boolean;
  event_name?: string;
  description?: string;
}

export interface ShowcaseEvent {
  event_name: string;
  request_date: string;
  lead_day: number;
  variable: string;
  description: string;
  event_type: string;
  data_source?: string;
  mean_confidence?: number;
  mean_bust_probability?: number;
  grid_latitudes?: number[];
  grid_longitudes?: number[];
  confidence_map?: number[][];
  bust_probability_map?: number[][];
  error_magnitude_map?: number[][];
  top_drivers?: MeteoDriver[];
  high_bust_regions?: BustRegion[];
}

export interface SubdivisionStat {
  name: string;
  code: string;
  confidence: number;
  bust_prob: number;
  expected_error: number;
  high_risk_frac: number;
  status: 'NORMAL' | 'HIGH WATCH' | 'CRITICAL BUST';
  color: string;
}

export type NavTab = 'overview' | 'forecast' | 'error_analysis' | 'explainability';
