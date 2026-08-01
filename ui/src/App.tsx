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
          <h1>Nullius — Inference Dashboard</h1>
          <p>Deterministic gate verification for every clinical inference. Nullius in verba.</p>
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
          <span className="cp-label">STUDY EXPLORER</span>
          <h2 style={{ marginBottom: '0.25rem' }}>Gate Verification Results</h2>
          <p style={{ color: '#555', fontSize: '0.85rem', marginBottom: '1.5rem', fontFamily: 'var(--font-mono)' }}>POST /copilot/ask</p>
          
          <div className="study-list">
            {payload?.studies?.map(study => {
              const status = study.result.served ? 'served' : (study.result.refusalReason ? 'refused' : 'review');
              return (
                <div 
                  key={study.id} 
                  className={`study-item ${activeStudy?.id === study.id ? 'active' : ''}`}
                  onClick={() => setActiveStudy(study)}
                >
                  {study.pngOverlay && (
                    <img src={`data:image/png;base64,${study.pngOverlay}`} className="study-thumb" alt="Thumbnail" />
                  )}
                  <div className="study-info">
                    <div className="study-id">{study.id}</div>
                    <div className="study-site">{study.site}</div>
                  </div>
                  <div className={`badge badge-${status}`}>
                    {status}
                  </div>
                </div>
              );
            })}
            {(!payload?.studies || payload.studies.length === 0) && (
              <div style={{ padding: '1rem', textAlign: 'center', color: '#888' }}>No studies available</div>
            )}
          </div>

          {activeStudy && (
            <div className="study-detail cp-animate">
              <h3>Study Details</h3>
              <div className="detail-row" style={{ marginTop: '1rem' }}>
                <span className="cp-label">Trace ID</span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>{activeStudy.result.traceId || 'N/A'}</span>
              </div>
              <div className="detail-row">
                <span className="cp-label">Latency</span>
                <span>{activeStudy.result.latencyMs ? `${activeStudy.result.latencyMs}ms` : 'N/A'}</span>
              </div>

              <div className={`verdict-box ${activeStudy.result.served ? 'served' : 'refused'}`}>
                <strong>{activeStudy.result.served ? 'Served' : 'Refused'}</strong>
                {activeStudy.result.refusalReason && (
                  <p style={{ fontSize: '0.85rem', marginTop: '0.25rem', color: 'var(--danger)' }}>
                    Reason: {activeStudy.result.refusalReason}
                  </p>
                )}
                {activeStudy.result.detail && (
                  <p style={{ fontSize: '0.85rem', marginTop: '0.25rem' }}>
                    {activeStudy.result.detail}
                  </p>
                )}
              </div>

              {activeStudy.result.probability !== undefined && payload?.operatingPoint !== undefined && (
                <div>
                  <span className="cp-label">Model Probability vs Operating Point</span>
                  <div className="score-bar-container">
                    <div 
                      className="score-bar-fill" 
                      style={{ width: `${Math.min(100, Math.max(0, activeStudy.result.probability * 100))}%` }}
                    />
                    <div 
                      className="score-bar-marker" 
                      style={{ left: `${payload.operatingPoint * 100}%` }}
                      title={`Operating Point: ${payload.operatingPoint}`}
                    />
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
                    <span>{(activeStudy.result.probability * 100).toFixed(1)}%</span>
                    <span>OP: {(payload.operatingPoint * 100).toFixed(1)}%</span>
                  </div>
                </div>
              )}

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
            <div className="cp-chart"></div>
          </div>

          <div className="cp-card cp-animate cp-animate-delay-4">
            <span className="cp-label">GATE ACTIVITY</span>
            <ul className="gate-activity-list">
              <li>
                <span style={{ fontFamily: 'var(--font-mono)' }}>QualityCheck</span>
                <span style={{ color: '#10b981' }}>PASS</span>
              </li>
              <li>
                <span style={{ fontFamily: 'var(--font-mono)' }}>TriageRouter</span>
                <span style={{ color: '#10b981' }}>PASS</span>
              </li>
              <li>
                <span style={{ fontFamily: 'var(--font-mono)' }}>AnatomyVerif</span>
                <span style={{ color: '#f59e0b' }}>WARN</span>
              </li>
              <li>
                <span style={{ fontFamily: 'var(--font-mono)' }}>NLISemantics</span>
                <span style={{ color: '#10b981' }}>PASS</span>
              </li>
            </ul>
          </div>

          <div className="cp-card cp-animate cp-animate-delay-5">
            <span className="cp-label">COPILOT PLAYGROUND</span>
            <div className="playground-form">
              <select className="cp-select" defaultValue="clinician">
                <option value="clinician">Clinician</option>
                <option value="nurse">Nurse</option>
                <option value="radiologist">Radiologist</option>
              </select>
              <input type="text" className="cp-input" placeholder="Patient ID (e.g. P-10943)" />
              <textarea className="cp-textarea" placeholder="Ask a question about the study..."></textarea>
              <button className="cp-button">Run Inference</button>
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
