import { useState, useEffect } from 'react';
import './index.css';

function App() {
  const [data, setData] = useState<any>(null);
  const [currentStudyIdx, setCurrentStudyIdx] = useState(0);
  const [imgMode, setImgMode] = useState<'overlay' | 'input'>('overlay');
  
  // Playground state
  const [role, setRole] = useState('clinician');
  const [patientId, setPatientId] = useState('pat-001');
  const [question, setQuestion] = useState('');
  const [pgResult, setPgResult] = useState<string>('');

  useEffect(() => {
    fetch('payload.json')
      .then(r => r.json())
      .then(d => setData(d))
      .catch(e => console.error("Error loading payload:", e));
  }, []);

  const handleAsk = async () => {
    if (!question) return;
    setPgResult('Asking local Copilot (http://127.0.0.1:8080)...');
    try {
      const res = await fetch('http://127.0.0.1:8080/copilot/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-User': 'playground', 'X-Role': role },
        body: JSON.stringify({ question, patient_id: patientId })
      });
      const data = await res.json();
      setPgResult(JSON.stringify(data, null, 2));
    } catch(e: any) {
      setPgResult('Error: ' + e.message + '\n\nIs the API running locally on port 8080?');
    }
  };

  if (!data) return <div style={{padding: '5rem', textAlign: 'center'}}>Loading Inference Control Plane...</div>;

  const study = data.studies[currentStudyIdx];
  const r = study.result;

  const renderVerdict = () => {
    if (r.served) {
      return (
        <div className="verdict served">
          <div className="stat-label">SERVED • {r.backend}</div>
          <div style={{fontSize: '1.25rem', fontWeight: 600, marginTop: '0.25rem'}}>{r.triage}</div>
          <div style={{fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '0.5rem'}}>
            Score {r.probability.toFixed(3)} against operating point {data.operatingPoint.toFixed(3)}
          </div>
          <div style={{height: '6px', background: 'var(--bg-dark)', borderRadius: '99px', marginTop: '1rem', overflow: 'hidden'}}>
            <div style={{height: '100%', background: 'var(--accent)', width: `${r.probability * 100}%`}}></div>
          </div>
        </div>
      );
    }
    const isReview = r.refusalReason === 'indeterminate_needs_review';
    return (
      <div className="verdict refused">
        <div className="stat-label">REFUSED • {r.refusalReason}</div>
        <div style={{fontSize: '1.25rem', fontWeight: 600, marginTop: '0.25rem'}}>
          {isReview ? 'Referred to a human reader' : 'No score issued'}
        </div>
        <div style={{fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '0.5rem'}}>
          {r.detail}
        </div>
      </div>
    );
  };

  return (
    <div>
      <header className="header">
        <div className="container">
          <div className="brand">
            <span className="brand-dot"></span>
            NULLIUS
          </div>
          <nav className="nav">
            <a href="#control-plane">Control Plane</a>
            <a href="#playground">Playground</a>
          </nav>
        </div>
      </header>

      <div className="hero">
        <div className="container">
          <div className="eyebrow">Clinical Inference Control Plane</div>
          <h1>Take nobody's word for it.</h1>
          <p>
            Nullius in verba. A clinical decision support platform whose central design claim is not that the model is clever, but that nothing reaches a clinician unless it survives a gate that can be switched off and measured.
          </p>
          <div className="stats-bar">
            <div className="stat-pill">Runtime Deps <span>0</span></div>
            <div className="stat-pill">ONNX Backend <span>Active</span></div>
            <div className="stat-pill">Tests <span>Passing</span></div>
          </div>
        </div>
      </div>

      <section id="control-plane" className="section">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Inference Control Plane</h2>
            <p className="section-desc">Every study below was decoded, de-identified, preprocessed, scored and traced when this page was generated. Select one to see the gate ladder it passed through.</p>
          </div>

          <div className="explorer">
            <div className="glass study-list">
              {data.studies.map((s: any, i: number) => {
                const isServed = s.result.served;
                const isReview = s.result.refusalReason === 'indeterminate_needs_review';
                const cls = isServed ? 'served' : (isReview ? 'review' : 'refused');
                const text = isServed ? 'served' : (isReview ? 'review' : 'refused');
                return (
                  <button key={s.id} className={`study-btn ${currentStudyIdx === i ? 'active' : ''}`} onClick={() => setCurrentStudyIdx(i)}>
                    <img src={`data:image/png;base64,${s.pngOverlay}`} alt="Thumb" />
                    <div className="study-meta">
                      <div className="study-id">{s.id}</div>
                      <div className="study-site">{s.site}</div>
                    </div>
                    <span className={`badge ${cls}`}>{text}</span>
                  </button>
                );
              })}
            </div>

            <div className="glass detail-panel">
              <div className="detail-header">
                <div>
                  <div style={{fontFamily: 'var(--font-mono)', fontSize: '0.875rem', color: 'var(--text-secondary)'}}>
                    {study.id} • {study.modality} • {study.site}
                  </div>
                  <h3 style={{fontSize: '1.5rem', margin: '0.5rem 0'}}>{study.title}</h3>
                  <div style={{fontSize: '0.875rem', color: 'var(--text-secondary)'}}>
                    Pseudonym: {study.pseudonym} • Synthetic truth: {study.truth}
                  </div>
                </div>
                <div style={{textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: '0.875rem', color: 'var(--text-secondary)'}}>
                  {r.latencyMs.toFixed(1)} ms e2e<br />
                  trace: {r.traceId}
                </div>
              </div>

              <div className="viewer-grid">
                <div>
                  <div className="frame-viewer">
                    <img src={`data:image/png;base64,${imgMode === 'overlay' ? study.pngOverlay : study.pngInput}`} alt="Lesion" />
                  </div>
                  <div className="img-tabs">
                    <button className={`img-tab ${imgMode === 'input' ? 'active' : ''}`} onClick={() => setImgMode('input')}>Input</button>
                    <button className={`img-tab ${imgMode === 'overlay' ? 'active' : ''}`} onClick={() => setImgMode('overlay')}>Segmentation</button>
                  </div>
                </div>
                
                <div>
                  <h4 style={{marginBottom: '1rem'}}>Gate Ladder</h4>
                  {renderVerdict()}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="playground" className="section">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Copilot Playground</h2>
            <p className="section-desc">Test the clinical copilot locally. Run <code>python -m nullius.api --nli</code> on your machine, then ask questions here. The site will connect to <code>http://127.0.0.1:8080</code>.</p>
          </div>
          
          <div className="glass card">
            <div style={{display: 'flex', gap: '1rem', marginBottom: '1rem'}}>
              <select value={role} onChange={e => setRole(e.target.value)} style={{padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border-color)', background: 'rgba(0,0,0,0.2)', color: 'var(--text-primary)'}}>
                <option value="clinician">Role: Clinician</option>
                <option value="nurse">Role: Nurse</option>
                <option value="radiologist">Role: Radiologist</option>
              </select>
              <input type="text" value={patientId} onChange={e => setPatientId(e.target.value)} placeholder="Patient ID (e.g. pat-001)" style={{padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border-color)', background: 'rgba(0,0,0,0.2)', color: 'var(--text-primary)', width: '200px'}} />
            </div>
            <div style={{display: 'flex', gap: '1rem', marginBottom: '1rem'}}>
              <input type="text" className="playground-input" value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask a clinical question about this patient..." onKeyDown={e => e.key === 'Enter' && handleAsk()} />
              <button className="btn" onClick={handleAsk}>Ask</button>
            </div>
            {pgResult && (
              <pre style={{marginTop: '1rem'}}>{pgResult}</pre>
            )}
          </div>
        </div>
      </section>

      <footer className="section" style={{textAlign: 'center', color: 'var(--text-secondary)'}}>
        <div className="container">
          <p>nullius in verba — Royal Society, 1660</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
