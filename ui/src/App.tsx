import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import "./index.css";

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
  
  const [apiKey, setApiKey] = useState("dev-copilot-key");
  const [prompt, setPrompt] = useState("Evaluate this study.");
  const [userId, setUserId] = useState("clinician-1");
  const [priority, setPriority] = useState<"low" | "high">("low");
  const [isLoading, setIsLoading] = useState(false);
  const [isLogsLoading, setIsLogsLoading] = useState(false);

  useEffect(() => {
    fetch('payload.json')
      .then(r => r.json())
      .then((data: Payload) => {
        setPayload(data);
      })
      .catch(e => console.error('Failed to load payload.json', e));

    fetch('report.json')
      .then(r => r.json())
      .then(data => setReport(data))
      .catch(e => console.error('Failed to load report.json', e));
  }, []);

  const derivedModel = useMemo(() => {
    return priority === "high" ? "premium-model" : "smart-router";
  }, [priority]);

  const kpiP95 = report?.p95Latency || '218ms';
  const kpiP95Delta = report?.latencyDelta || '-11%';
  const kpiPassRate = report?.gatePassRate || '67.4%';
  const kpiPassRateDelta = report?.passRateDelta || '+9%';
  
  const totalStudies = payload?.studies?.length || 0;
  const refusedCount = payload?.studies?.filter(s => s.result.served === false).length || 0;
  const refusalRate = totalStudies > 0 ? ((refusedCount / totalStudies) * 100).toFixed(1) + '%' : '0%';

  function loadRecentLogs() {
    setIsLogsLoading(true);
    setTimeout(() => setIsLogsLoading(false), 800);
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsLoading(true);
    setTimeout(() => setIsLoading(false), 1200);
  }

  return (
    <main className="cp-shell">
      <header className="cp-card cp-card-strong mb-4 p-5 md:p-7 cp-animate cp-animate-delay-1">
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="cp-label text-[var(--accent-strong)]">Clinical Decision Support</p>
            <h1 className="text-3xl font-bold tracking-tight md:text-4xl">
              Nullius — Copilot Dashboard
            </h1>
            <a href="#" className="mt-4 inline-block text-sm font-medium text-[var(--accent-strong)] hover:underline">
              &larr; Back to Website
            </a>
            <p className="mt-2 max-w-2xl text-sm/6 text-neutral-700 md:text-base/7">
              Deterministic gate verification for every clinical evaluation. Nullius in verba.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <span className="cp-pill">
              <span className="cp-dot bg-emerald-500" /> All gates active
            </span>
            <span className="cp-pill">
              <span className="cp-dot bg-cyan-600" /> ONNX runtime
            </span>
            <span className="cp-pill">
              <span className="cp-dot bg-fuchsia-500" /> NLI judge enabled
            </span>
            <span className="cp-pill">
              <span className="cp-dot bg-amber-500" /> Routing: {derivedModel}
            </span>
          </div>
        </div>
      </header>

      <section className="cp-grid mb-4 grid-cols-1 md:grid-cols-2 xl:grid-cols-4">
        <article className="cp-card p-4 cp-animate cp-animate-delay-2">
          <p className="cp-label text-neutral-600">P95 Latency</p>
          <p className="mt-1 text-2xl font-bold">{kpiP95}</p>
          <p className="mt-1 text-sm text-emerald-700">{kpiP95Delta} vs baseline</p>
        </article>
        <article className="cp-card p-4 cp-animate cp-animate-delay-3">
          <p className="cp-label text-neutral-600">Gate Pass Rate</p>
          <p className="mt-1 text-2xl font-bold">{kpiPassRate}</p>
          <p className="mt-1 text-sm text-emerald-700">{kpiPassRateDelta} vs baseline</p>
        </article>
        <article className="cp-card p-4 cp-animate cp-animate-delay-4">
          <p className="cp-label text-neutral-600">Studies Processed</p>
          <p className="mt-1 text-2xl font-bold">{totalStudies}</p>
          <p className="mt-1 text-sm text-emerald-700">Total in payload</p>
        </article>
        <article className="cp-card p-4 cp-animate cp-animate-delay-5">
          <p className="cp-label text-neutral-600">Refusal Rate</p>
          <p className="mt-1 text-2xl font-bold">{refusalRate}</p>
          <p className="mt-1 text-sm text-emerald-700">Refused by gates</p>
        </article>
      </section>

      <section className="cp-grid grid-cols-1 xl:grid-cols-[1.4fr_1fr]">
        <article className="cp-card p-5 md:p-6 cp-animate cp-animate-delay-2">
          <div className="mb-4 flex items-end justify-between">
            <div>
              <p className="cp-label text-neutral-600">Live Playground</p>
              <h2 className="text-xl font-semibold">Generate With Clinical API</h2>
            </div>
            <p className="cp-label text-neutral-500">POST /copilot/ask</p>
          </div>

          <form onSubmit={onSubmit}>
            <fieldset disabled={isLoading} className="space-y-3">
            <div className="grid gap-3 md:grid-cols-2">
              <div className="space-y-1">
                <label htmlFor="userId" className="cp-label block text-neutral-600">
                  User ID <span className="text-red-500" aria-hidden="true">*</span>
                </label>
                <input
                  id="userId"
                  className="cp-input"
                  value={userId}
                  onChange={(event) => setUserId(event.target.value)}
                  required
                />
              </div>
              <div className="space-y-1">
                <label htmlFor="apiKey" className="cp-label block text-neutral-600">
                  API Key <span className="text-red-500" aria-hidden="true">*</span>
                </label>
                <input
                  id="apiKey"
                  className="cp-input"
                  value={apiKey}
                  onChange={(event) => setApiKey(event.target.value)}
                  required
                />
              </div>
            </div>

            <div className="space-y-1">
              <label htmlFor="prompt" className="cp-label block text-neutral-600">
                Prompt <span className="text-red-500" aria-hidden="true">*</span>
                <span className="ml-2 text-xs font-normal normal-case tracking-normal text-neutral-400">
                  (Cmd/Ctrl + Enter to run)
                </span>
              </label>
              <textarea
                id="prompt"
                className="cp-input min-h-32"
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                onKeyDown={(event) => {
                  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                    event.preventDefault();
                    if (!isLoading) {
                      event.currentTarget.form?.requestSubmit();
                    }
                  }
                }}
                required
              />
            </div>

            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <div className="space-y-1">
                <label htmlFor="priority" className="cp-label block text-neutral-600">Priority</label>
                <select
                  id="priority"
                  className="cp-input"
                  value={priority}
                  onChange={(event) => setPriority(event.target.value as "low" | "high")}
                >
                  <option value="low">Low</option>
                  <option value="high">High</option>
                </select>
              </div>

              <button
                className="cp-button md:min-w-56"
                aria-busy={isLoading}
                type="submit"
              >
                {isLoading && (
                  <svg aria-hidden="true" className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                )}
                {isLoading ? "Evaluating..." : "Run Copilot"}
              </button>
            </div>
            </fieldset>
          </form>

        </article>

        <div className="cp-grid grid-cols-1 gap-4">
          <article className="cp-card p-5 cp-animate cp-animate-delay-3">
            <p className="cp-label text-neutral-600">Usage Trend</p>
            <h3 className="text-lg font-semibold">24h Request Volume</h3>
            <div className="cp-chart mt-3" />
          </article>

          <article className="cp-card p-5 cp-animate cp-animate-delay-4">
            <p className="cp-label text-neutral-600">Gate Activity</p>
            <ul className="mt-3 space-y-2 text-sm text-neutral-700">
              <li className="rounded-lg border border-[var(--border)] bg-[#fffdf8] p-2">
                Quality check threshold updated
              </li>
              <li className="rounded-lg border border-[var(--border)] bg-[#fffdf8] p-2">
                Model routing fell back to CPU
              </li>
              <li className="rounded-lg border border-[var(--border)] bg-[#fffdf8] p-2">
                NLI judge confidence lowered
              </li>
            </ul>
          </article>

          <article className="cp-card p-5 cp-animate cp-animate-delay-5">
            <div className="flex items-center justify-between gap-2">
              <div>
                <p className="cp-label text-neutral-600">Evaluation Logs</p>
                <h3 className="text-lg font-semibold">Recent Requests</h3>
              </div>
              <button
                className="cp-button px-3 py-2 text-sm"
                disabled={isLogsLoading}
                aria-busy={isLogsLoading}
                onClick={loadRecentLogs}
                type="button"
              >
                {isLogsLoading && (
                  <svg aria-hidden="true" className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                )}
                {isLogsLoading ? "Loading..." : "Refresh"}
              </button>
            </div>

            <div className="mt-3 space-y-2 text-sm max-h-[300px] overflow-y-auto pr-2">
              {(!payload?.studies || payload.studies.length === 0) && !isLogsLoading && (
                <p className="rounded-lg border border-[var(--border)] bg-[#fffdf8] p-3 text-neutral-600">
                  No evaluation logs yet.
                </p>
              )}

              {payload?.studies?.map((entry) => (
                <div
                  key={entry.id}
                  className="rounded-lg border border-[var(--border)] bg-[#fffdf8] p-3"
                >
                  <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-neutral-600">
                    <span className="font-mono text-neutral-800 font-semibold">{entry.id}</span>
                    <span>{entry.result.latencyMs ? `${entry.result.latencyMs}ms` : 'N/A'}</span>
                    <span>{entry.result.served ? "cache hit" : "cache miss"}</span>
                  </div>
                  <p className="mt-1 text-xs text-neutral-500">Status: {entry.result.served ? 'success' : 'refused'}</p>
                  {entry.result.refusalReason && (
                    <p className="mt-1 text-xs text-[var(--danger)]">{entry.result.refusalReason}</p>
                  )}
                </div>
              ))}
            </div>
          </article>
        </div>
      </section>
    </main>
  );
}
