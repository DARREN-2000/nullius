import { useEffect, useState } from 'react';
import './index.css';

interface StudyResult {
  served: boolean;
  backend?: string;
  triage?: string;
  probability?: number;
  refusalReason?: string;
  detail?: string;
  latencyMs?: number;
  traceId?: string;
}

interface Study {
  id: string;
  title?: string;
  site: string;
  modality?: string;
  pseudonym?: string;
  truth?: boolean;
  pngInput?: string;
  pngOverlay?: string;
  features?: any;
  quality?: string;
  steps?: string[];
  header?: any;
  result: StudyResult;
}

interface Payload {
  studies: Study[];
  featureLabels?: any;
  featureMean?: any;
  operatingPoint?: number;
}

interface Report {
  p95Latency?: string;
  latencyDelta?: string;
  gatePassRate?: string;
  passRateDelta?: string;
}

export default function App() {
  const [payload, setPayload] = useState<Payload | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [activeStudy, setActiveStudy] = useState<Study | null>(null);
  const [imageTab, setImageTab] = useState<'input' | 'overlay'>('overlay');

  useEffect(() => {
    fetch('payload.json')
      .then(r => r.json())
      .then((data: Payload) => {
        setPayload(data);
        if (data.studies && data.studies.length > 0) {
          setActiveStudy(data.studies[0]);
        }
      })
      .catch(e => console.error('Failed to load payload.json', e));

    fetch('report.json')
      .then(r => r.json())
      .then(data => setReport(data))
      .catch(e => console.error('Failed to load report.json', e));
  }, []);

  const kpiP95 = report?.p95Latency || '218ms';
  const kpiP95Delta = report?.latencyDelta || 'vs baseline';
  const kpiPassRate = report?.gatePassRate || '67.4%';
  const kpiPassRateDelta = report?.passRateDelta || '+9%';
  
  const totalStudies = payload?.studies?.length || 0;
  const refusedCount = payload?.studies?.filter(s => s.result.served === false).length || 0;
  const refusalRate = totalStudies > 0 ? ((refusedCount / totalStudies) * 100).toFixed(1) + '%' : '0%';

  return (
    <div className="cp-shell">
      <header className="cp-card cp-card-strong cp-animate">
        <div className="header-content">
          <span className="cp-label">CLINICAL DECISION SUPPORT</span>
          <h1>Nullius — Clinical Copilot Dashboard</h1>
          <p>Deterministic gate verification for every clinical evaluation. Nullius in verba.</p>
        </div>
        <div className="status-pills">
          <div className="cp-pill">
            <div className="cp-dot cp-dot-green"></div>
            All gates active
          </div>
          <div className="cp-pill">
            <div className="cp-dot cp-dot-cyan"></div>
            ONNX runtime
          </div>
          <div className="cp-pill">
            <div className="cp-dot cp-dot-amber"></div>
            NLI judge enabled
          </div>
        </div>
      </header>

      <section className="cp-grid cp-grid-4 cp-animate cp-animate-delay-1">
        <div className="cp-card">
          <span className="cp-label">P95 LATENCY</span>
          <div className="kpi-value">
            {kpiP95} <span className="kpi-delta">{kpiP95Delta}</span>
          </div>
        </div>
        <div className="cp-card">
          <span className="cp-label">GATE PASS RATE</span>
          <div className="kpi-value">
            {kpiPassRate} <span className="kpi-delta">{kpiPassRateDelta}</span>
          </div>
        </div>
        <div className="cp-card">
          <span className="cp-label">STUDIES PROCESSED</span>
          <div className="kpi-value">{totalStudies}</div>
        </div>
        <div className="cp-card">
          <span className="cp-label">REFUSAL RATE</span>
          <div className="kpi-value">{refusalRate}</div>
        </div>
      </section>

      <main className="cp-grid cp-grid-main cp-animate cp-animate-delay-2">
        <article className="cp-card">
          <div style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between' }}>
            <div>
              <span className="cp-label">LIVE PLAYGROUND</span>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Clinical Copilot API</h2>
            </div>
            <p className="cp-label" style={{ color: '#6b7280', marginBottom: 0 }}>POST /copilot/ask</p>
          </div>

          <div className="playground-form" style={{ marginTop: 0 }}>
            <div style={{ display: 'grid', gap: '0.75rem', gridTemplateColumns: '1fr 1fr' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                <label className="cp-label" style={{ color: '#4b5563', marginBottom: 0 }}>Role <span style={{ color: 'var(--danger)' }}>*</span></label>
                <select className="cp-select" defaultValue="clinician">
                  <option value="clinician">Clinician</option>
                  <option value="nurse">Nurse</option>
                  <option value="radiologist">Radiologist</option>
                </select>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                <label className="cp-label" style={{ color: '#4b5563', marginBottom: 0 }}>Patient ID <span style={{ color: 'var(--danger)' }}>*</span></label>
                <input type="text" className="cp-input" placeholder="e.g. P-10943" />
              </div>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              <label className="cp-label" style={{ color: '#4b5563', marginBottom: 0 }}>Clinical Question <span style={{ color: 'var(--danger)' }}>*</span></label>
              <textarea className="cp-textarea" placeholder="Ask a question about the study..."></textarea>
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '0.5rem' }}>
              <button className="cp-button" style={{ minWidth: '14rem' }}>Run Copilot</button>
            </div>
          </div>

          {activeStudy && (
            <div className="study-detail cp-animate" style={{ marginTop: '2rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <h3>Study Evaluation: {activeStudy.id}</h3>
                <span className="cp-label">Trace: {activeStudy.result.traceId || 'N/A'}</span>
              </div>
              
              <div className={`verdict-box ${activeStudy.result.served ? 'served' : 'refused'}`}>
                <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', fontSize: '0.85rem', marginBottom: '0.5rem' }}>
                  <span><strong>Verdict:</strong> {activeStudy.result.served ? 'Served' : 'Refused'}</span>
                  <span><strong>Latency:</strong> {activeStudy.result.latencyMs ? `${activeStudy.result.latencyMs}ms` : 'N/A'}</span>
                  {activeStudy.result.probability !== undefined && payload?.operatingPoint !== undefined && (
                    <span><strong>Score:</strong> {(activeStudy.result.probability * 100).toFixed(1)}% (OP: {(payload.operatingPoint * 100).toFixed(1)}%)</span>
                  )}
                </div>
                {activeStudy.result.refusalReason && (
                  <p style={{ fontSize: '0.85rem', color: 'var(--danger)' }}>
                    Reason: {activeStudy.result.refusalReason}
                  </p>
                )}
                {activeStudy.result.detail && (
                  <p style={{ fontSize: '0.85rem', marginTop: '0.25rem' }}>
                    {activeStudy.result.detail}
                  </p>
                )}
              </div>

              {(activeStudy.pngInput || activeStudy.pngOverlay) && (
               <div className="image-viewer">
                 <div className="image-tabs">
                   {activeStudy.pngInput && (
                     <button 
                       className={`image-tab ${imageTab === 'input' ? 'active' : ''}`}
                       onClick={() => setImageTab('input')}
                     >
                       Input
                     </button>
                   )}
                   {activeStudy.pngOverlay && (
                     <button 
                       className={`image-tab ${imageTab === 'overlay' ? 'active' : ''}`}
                       onClick={() => setImageTab('overlay')}
                     >
                       Segmentation
                     </button>
                   )}
                 </div>
                 <img 
                   src={`data:image/png;base64,${imageTab === 'input' ? activeStudy.pngInput : activeStudy.pngOverlay}`} 
                   className="large-image" 
                   alt="Study detail" 
                 />
               </div>
              )}
            </div>
          )}
        </article>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div className="cp-card cp-animate cp-animate-delay-3">
            <span className="cp-label">USAGE TREND</span>
            <h3 style={{ fontSize: '1.125rem', fontWeight: 600, marginBottom: '0.75rem' }}>24h Request Volume</h3>
            <div className="cp-chart"></div>
          </div>

          <div className="cp-card cp-animate cp-animate-delay-4">
            <span className="cp-label">OPERATIONAL ACTIVITY</span>
            <ul className="gate-activity-list" style={{ marginTop: '0.75rem' }}>
              <li>
                <span>Quality check threshold updated</span>
              </li>
              <li>
                <span>Model routing fell back to CPU</span>
              </li>
              <li>
                <span>NLI judge confidence lowered</span>
              </li>
            </ul>
          </div>

          <div className="cp-card cp-animate cp-animate-delay-5">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <div>
                <span className="cp-label">STUDY LOGS</span>
                <h3 style={{ fontSize: '1.125rem', fontWeight: 600 }}>Recent Evaluations</h3>
              </div>
              <button className="cp-button" style={{ padding: '0.5rem 0.75rem', fontSize: '0.75rem' }}>Refresh</button>
            </div>

            <div className="study-list" style={{ maxHeight: '300px' }}>
              {payload?.studies?.map(study => {
                const status = study.result.served ? 'served' : (study.result.refusalReason ? 'refused' : 'review');
                return (
                  <div 
                    key={study.id} 
                    className={`study-item ${activeStudy?.id === study.id ? 'active' : ''}`}
                    onClick={() => setActiveStudy(study)}
                    style={{ padding: '0.5rem', gap: '0.75rem' }}
                  >
                    <div className="study-info">
                      <div style={{ display: 'flex', gap: '0.5rem', fontSize: '0.75rem', color: '#666', flexWrap: 'wrap' }}>
                        <span className="study-id" style={{ color: 'var(--fg)' }}>{study.id}</span>
                        <span>{study.result.latencyMs ? `${study.result.latencyMs}ms` : ''}</span>
                        <span>{study.result.served ? 'cache hit' : 'cache miss'}</span>
                      </div>
                      <div className="study-site" style={{ marginTop: '0.25rem' }}>Status: {status}</div>
                    </div>
                  </div>
                );
              })}
              {(!payload?.studies || payload.studies.length === 0) && (
                <div style={{ padding: '1rem', textAlign: 'center', color: '#888' }}>No logs available</div>
              )}
            </div>
          </div>
        </div>
      </main>

      <footer className="cp-animate cp-animate-delay-6">
        nullius in verba — Royal Society, 1660
      </footer>
    </div>
  );
}
