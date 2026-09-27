import React, { useEffect, useState } from 'react';
import { Activity, AlertTriangle, CalendarDays, Check, ChevronDown, CircleHelp, CloudRain, Database, Download, Gauge, Layers3, MapPin, Radio, RefreshCw, Satellite, Search, ShieldAlert, Sparkles, Wind } from 'lucide-react';
import { ForecastMap } from './components/ForecastMap';
import { SHOWCASE_EVENTS } from './api/showcaseData';
import { fetchPrediction } from './api/client';
import type { PredictResponse, ShowcaseEvent } from './types';

const LEADS = [1,2,3,4,5,6,7,8,9,10];
const DATA_LEADS = new Set([3,5,7,10]);
const shortDate = (value: string) => new Date(`${value}T00:00:00Z`).toLocaleDateString('en-IN',{day:'2-digit',month:'short',year:'numeric',timeZone:'UTC'});

export const App: React.FC = () => {
  const [event, setEvent] = useState<ShowcaseEvent>(SHOWCASE_EVENTS.find(e => e.request_date === '2022-06-25') ?? SHOWCASE_EVENTS[0]);
  const [date, setDate] = useState(event.request_date);
  const [lead, setLead] = useState(event.lead_day);
  const [requestState, setRequestState] = useState<{key:string;data?:PredictResponse;error?:string}|null>(null);
  const [online, setOnline] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [reviewState, setReviewState] = useState<{key:string;value:boolean}|null>(null);
  const [mapMode, setMapMode] = useState<'confidence'|'bust'|'error'>('confidence');

  const requestKey=`${date}|${lead}|${refresh}`;
  const requestResult=requestState?.key===requestKey?requestState:null;
  const data=requestResult?.data??null;
  const error=requestResult?.error??'';
  const loading=requestResult===null;
  const reviewKey=`${date}:${lead}`;
  const reviewed=reviewState?.key===reviewKey?reviewState.value:localStorage.getItem(`drishti-review:${reviewKey}`)==='saved';

  useEffect(() => { let alive=true; fetch('/api/v1/info').then(r=>{if(alive)setOnline(r.ok)}).catch(()=>{if(alive)setOnline(false)}); return ()=>{alive=false}; },[]);
  useEffect(() => {
    let alive=true;
    fetchPrediction({date,lead_day:lead,variable:'rainfall',use_cache:true})
      .then(value=>{if(alive)setRequestState({key:requestKey,data:value})})
      .catch(reason=>{if(alive)setRequestState({key:requestKey,error:reason instanceof Error?reason.message:'No matching forecast is available.'})});
    return ()=>{alive=false};
  },[date,lead,refresh,requestKey]);

  const label=data?.data_source==='illustrative_only'?'Illustrative demo':data?.data_source==='precomputed_cache'?'Cached result':data?.data_source==='live_model'?'Computed from model':'Unavailable';
  const mapReady=Boolean(data?.grid_latitudes?.length && data?.grid_longitudes?.length && data?.confidence_map?.length);
  const hasMetrics=Boolean(data && typeof data.mean_bust_probability==='number' && Number.isFinite(data.mean_bust_probability) && typeof data.mean_confidence==='number' && Number.isFinite(data.mean_confidence));
  const selectEvent=(next:ShowcaseEvent)=>{setEvent(next);setDate(next.request_date);setLead(next.lead_day)};
  const exportBriefing=()=>{
    if(!data)return;
    const report={product:'DRISHTI — Forecast Trust Intelligence',problem_statement:'SIH26079',provenance:label,request:{initialization_date:date,lead_day:lead,variable:'rainfall'},forecast:{mean_bust_probability:data.mean_bust_probability,mean_confidence_index:data.mean_confidence,regions:data.high_bust_regions},limitations:'Cached forecast output. This project package does not include the training checkpoint or source gridded data needed to independently verify model lineage or predictive skill.',generated_at:new Date().toISOString()};
    const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'})); const a=document.createElement('a');a.href=url;a.download=`drishti-${date}-day${lead}.json`;a.click();URL.revokeObjectURL(url);
  };
  const toggleReview=()=>{const next=!reviewed;localStorage.setItem(`drishti-review:${reviewKey}`,next?'saved':'');setReviewState({key:reviewKey,value:next})};

  return <div className="drishti-shell">
    <header className="topbar">
      <a className="brand" href="#home" aria-label="DRISHTI home"><span className="brand-mark"><svg viewBox="0 0 40 40" fill="none" aria-hidden="true"><ellipse cx="20" cy="20" rx="17" ry="9" stroke="currentColor"/><ellipse cx="20" cy="20" rx="9" ry="17" stroke="currentColor" transform="rotate(35 20 20)"/><circle cx="20" cy="20" r="4" fill="currentColor"/></svg></span><span><strong>DRISHTI</strong><small>FORECAST TRUST INTELLIGENCE</small></span></a>
      <div className="top-context"><span className="context-label">EARTH SYSTEMS / 01</span><span className="context-divider"/><span className="context-event"><span className="live-dot"/>MONSOON OBSERVATORY</span></div>
      <div className="top-actions"><span className="prototype-tag">SIH26079 · PROTOTYPE</span><span className={`connection ${online?'connected':''}`}><i/>{online?'API READY':'OFFLINE DEMO'}</span><button className="icon-button" aria-label="Focus scenario selector" onClick={()=>document.getElementById('scenario-select')?.focus()}><Search size={17}/></button><button className="export-button" onClick={exportBriefing} disabled={!data}><Download size={15}/> Export briefing</button></div>
    </header>
    <div className="workspace">
      <aside className="rail"><button className="rail-item selected"><Layers3 size={18}/><span>Watch</span></button><button className="rail-item" onClick={()=>document.getElementById('evidence')?.scrollIntoView({behavior:'smooth'})}><Activity size={18}/><span>Evidence</span></button><button className="rail-item" onClick={()=>document.getElementById('about')?.scrollIntoView({behavior:'smooth'})}><Database size={18}/><span>Data</span></button><div className="rail-spacer"/><div className="rail-footer"><span>MOES</span><small>NCMRWF<br/>CHALLENGE</small></div></aside>
      <main className="main-view" id="home">
        <section className="page-heading"><div><div className="eyebrow"><span className="eyebrow-line"/>THE FORECAST INTELLIGENCE ATLAS</div><h1>Clarity in an<br className="title-break"/> <em>uncertain atmosphere.</em></h1><p>A closer look at where forecasts hold — and where they may break. Explore rainfall reliability across India.</p></div><div className="heading-date"><CalendarDays size={15}/><span>{shortDate(date)}</span><ChevronDown size={14}/></div></section>
        <section className="scenario-strip"><div className="scenario-select"><div className="scenario-icon"><CloudRain size={17}/></div><div className="scenario-content"><span className="micro-label">HISTORICAL CASE</span><select id="scenario-select" aria-label="Choose historical case" value={event.event_name} onChange={e=>{const next=SHOWCASE_EVENTS.find(item=>item.event_name===e.target.value);if(next)selectEvent(next)}}>{SHOWCASE_EVENTS.map(item=><option key={item.event_name} value={item.event_name}>{item.event_name}</option>)}</select></div></div><div className="scenario-meta"><span><MapPin size={14}/> India · 4 monitored regions</span><span><Satellite size={14}/> NOAA GEFSv12</span><span className="provenance-chip"><i/>{label}</span></div></section>
        <section className="metric-row" aria-label="Selected forecast summary">
          <article className="metric-card risk"><div className="metric-top"><span className="metric-icon"><ShieldAlert size={16}/></span><span className="metric-label">MEAN BUST PROBABILITY</span><CircleHelp size={14} className="muted-icon"/></div><div className="metric-value">{hasMetrics?`${(data!.mean_bust_probability*100).toFixed(1)}%`:loading?'—':'Unavailable'}</div><div className="metric-foot"><span className="risk-indicator"><i/>{hasMetrics?data!.mean_bust_probability>=.5?'Elevated review signal':'Model estimate':'No prediction'}</span><span>Day {lead}</span></div></article>
          <article className="metric-card confidence"><div className="metric-top"><span className="metric-icon"><Gauge size={16}/></span><span className="metric-label">CONFIDENCE INDEX</span><CircleHelp size={14} className="muted-icon"/></div><div className="metric-value">{hasMetrics?<>{data!.mean_confidence.toFixed(1)}<small>%</small></>:loading?'—':'Unavailable'}</div><div className="metric-foot"><span>Composite index · 0–100</span><span>Day {lead}</span></div></article>
          <article className="metric-card error-metric"><div className="metric-top"><span className="metric-icon"><Activity size={16}/></span><span className="metric-label">PREDICTED ERROR</span><CircleHelp size={14} className="muted-icon"/></div><div className="metric-value">{mapReady?'Grid available':loading?'—':'Unavailable'}</div><div className="metric-foot"><span>Absolute rainfall error · mm/day</span><span>{data?.error_magnitude_map?.length?`${data.error_magnitude_map.length} grid rows`:'No field'}</span></div></article>
          <article className="metric-card support"><div className="metric-top"><span className="metric-icon"><Radio size={16}/></span><span className="metric-label">EVIDENCE STATUS</span><CircleHelp size={14} className="muted-icon"/></div><div className="metric-value evidence-value">{data?'Historical inference':'Awaiting data'}</div><div className="metric-foot"><span>{data?.data_source==='precomputed_cache'?'Saved output · source model unverified':'Coverage and support metadata'}</span><span className="mini-dot"/></div></article>
        </section>
        <div className="atlas-section-label"><span>01 / THE SPATIAL PICTURE</span><span>INDIA · RAINFALL INTELLIGENCE</span></div><section className="analysis-grid">
          <div className="map-card"><div className="map-heading"><div><div className="panel-kicker">SPATIAL OUTLOOK <span> / </span> INDIA</div><h2>The reliability landscape</h2><p>Rainfall forecast · 0.25° native grid · domain view</p></div><div className="map-tools"><div className="layer-switch"><button className={mapMode==='confidence'?'active':''} onClick={()=>setMapMode('confidence')}>Confidence</button><button className={mapMode==='bust'?'active':''} onClick={()=>setMapMode('bust')}>Bust risk</button><button className={mapMode==='error'?'active':''} onClick={()=>setMapMode('error')}>Error</button></div></div></div>
            <div className="map-stage">{mapReady?<ForecastMap lats={data!.grid_latitudes} lons={data!.grid_longitudes} confidenceMap={data!.confidence_map} bustMap={data!.bust_probability_map} errorMap={data!.error_magnitude_map} mode={mapMode} dataSource={data!.data_source}/>:<div className="map-empty"><div className="map-empty-mark"><MapPin size={21}/></div><strong>{loading?'Reading forecast fields':'Map unavailable'}</strong><span>{loading?'Loading the exact selected date and lead…':error||'No spatial prediction accompanies this case.'}</span>{!loading&&<button className="subtle-button" onClick={()=>setRefresh(current=>current+1)}><RefreshCw size={14}/> Try again</button>}</div>}</div>
            <div className="map-legend"><div className="legend-title">{mapMode==='confidence'?'FORECAST CONFIDENCE INDEX':mapMode==='bust'?'FORECAST BUST PROBABILITY':'PREDICTED ABSOLUTE ERROR · MM/DAY'}</div><div className="legend-scale"><span>{mapMode==='bust'?'LOW':'LOWER'}</span><div className={`gradient-bar ${mapMode}`}/><span>{mapMode==='bust'?'HIGH':'HIGHER'}</span></div><span className="legend-note">{mapMode==='confidence'?'Predicted reliability indicator · not a calibrated probability':mapMode==='bust'?'Predicted threshold exceedance · calibration status unverified':'Relative color classes across the selected grid; values remain at native resolution'}</span></div>
          </div>
          <aside className="evidence-card" id="evidence"><div className="evidence-head"><div><div className="panel-kicker">SIGNAL REVIEW</div><h2>Behind the forecast</h2></div><span className="evidence-count"><Sparkles size={13}/> MODEL</span></div><div className="signal-box"><div className="signal-line"><span className="signal-symbol"><AlertTriangle size={16}/></span><div><strong>{data?.high_bust_regions?.length?`${data.high_bust_regions.length} flagged areas`:'Awaiting spatial signal'}</strong><small>{data?'Ranked connected high-risk regions':'Select a case with an available prediction'}</small></div></div>{data?.high_bust_regions?.slice(0,3).map((region,index)=><div className="region-row" key={`${region.name}-${index}`}><span className="region-rank">0{index+1}</span><span className="region-name">{region.name}</span><span className="region-risk">{(region.mean_bust_prob*100).toFixed(0)}%</span></div>)}</div><div className="evidence-section"><div className="section-title"><span>MODEL ATTRIBUTION</span><span className="not-causality">SENSITIVITY · NOT CAUSALITY</span></div>{data?.top_drivers?.length?data.top_drivers.slice(0,3).map(driver=><div className="driver-row" key={driver.channel_name}><div className="driver-label"><span>{driver.channel_name}</span><span>{driver.attribution_pct.toFixed(1)}%</span></div><div className="driver-track"><i style={{width:`${Math.min(driver.attribution_pct,100)}%`}}/></div></div>):<div className="empty-copy">No attribution evidence is available for this request.</div>}</div><div className="evidence-callout"><div className="callout-heading"><Wind size={15}/> HOW TO READ THIS</div><p>{data?.description||'A bust probability estimates the chance of exceeding the training-period error threshold. This project does not include the checkpoint or source grids needed to independently verify the cached model output.'}</p></div><button className="review-button" onClick={toggleReview} disabled={!data}><Check size={15}/>{reviewed?'Saved for forecaster review':'Mark for forecaster review'}<span>{reviewed?'✓':'↗'}</span></button></aside>
        </section>
        <section className="horizon-card"><div className="horizon-top"><div><div className="panel-kicker">RELIABILITY HORIZON <span> / </span> FORECAST LEAD</div><h2>A view through time</h2><p>The prototype is configured for four lead days. Each selected date and lead still needs an exact forecast result; unsupported requests stay unavailable.</p></div><span className="availability"><i/>{[...DATA_LEADS].length} LEADS IN PROTOTYPE</span></div><div className="horizon-track">{LEADS.map(day=>{const supported=DATA_LEADS.has(day);const active=lead===day;const failed=active&&!loading&&Boolean(error);return <button key={day} disabled={!supported} aria-pressed={active} title={supported?`Request exact forecast for day ${day}`:`Day ${day} is not implemented in the current data pipeline`} className={`horizon-step ${supported?'available':'unavailable'} ${active?'active':''} ${failed?'failed':''}`} onClick={()=>setLead(day)}><span className="step-day">D{String(day).padStart(2,'0')}</span><span className="step-state">{failed?'NO OUTPUT':active?'SELECTED':supported?'CONFIGURED':'NO DATA'}</span></button>})}</div><div className="horizon-foot"><span><i className="state-selected"/> Selected horizon</span><span><i className="state-available"/> Lead configured in prototype</span><span><i className="state-empty"/> Lead unsupported</span><span className="horizon-warning"><AlertTriangle size={13}/> Exact date & lead matching</span></div></section>
        <section className="bottom-grid" id="about"><article className="about-card"><div className="about-icon"><Database size={17}/></div><div><div className="panel-kicker">DATA & SCOPE</div><h3>Rainfall · 4 IMD subdivisions · 2021–22</h3><p>NOAA GEFSv12 control forecast compared with IMD gridded rainfall. Current project scope documents Days 3, 5, 7 and 10 only.</p></div></article><article className="about-card"><div className="about-icon amber"><CircleHelp size={17}/></div><div><div className="panel-kicker">PROTOTYPE LIMITATION</div><h3>Results need independent verification</h3><p>Cached maps are shown as saved outputs. This project package has no training checkpoint or raw gridded input data; predictive skill and lineage cannot be confirmed here.</p></div></article></section>
        <footer className="page-footer"><span>DRISHTI <i>·</i> SIH26079</span><span>Prototype for the NCMRWF problem statement · Ministry of Earth Sciences</span><button><Database size={13}/> Operational prototype</button></footer>
      </main>
    </div>
  </div>;
};

export default App;
