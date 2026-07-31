"""Render out/report.json into a self-contained clinician + observability view.

No build step, no JS framework, no CDN: one HTML file with inline SVG sparklines,
which means it opens from disk, embeds anywhere, and cannot silently break when a
remote asset moves.
"""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"

INK = "#2C2C2B"
MUTED = "#7D7A75"
BORDER = "#E6E5E3"
SOFT = "#F9F8F7"
BLUE = "#2783DE"
GREEN = "#46A171"
ORANGE = "#D5803B"
RED = "#E56458"

SEVERITY_COLOR = {
    "critical_high": RED, "critical_low": RED, "high": ORANGE, "low": ORANGE, "normal": GREEN,
}
SEVERITY_LABEL = {
    "critical_high": "Critical high", "critical_low": "Critical low",
    "high": "High", "low": "Low", "normal": "Normal",
}
INTERACTION_COLOR = {"contraindicated": RED, "major": RED, "moderate": ORANGE, "minor": MUTED}


def esc(value) -> str:
    return html.escape(str(value))


def sparkline(series: list[dict], color: str, width: int = 132, height: int = 34) -> str:
    values = [p["value"] for p in series]
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    step = width / (len(values) - 1)
    pts = [(i * step, height - 4 - ((v - lo) / span) * (height - 10)) for i, v in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"0,{height} " + line + f" {width},{height}"
    last_x, last_y = pts[-1]
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="trend of last {len(values)} results">'
        f'<polygon points="{area}" fill="{color}" opacity="0.10"/>'
        f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3" fill="{color}"/></svg>'
    )


def _ablation_note(row: dict) -> str:
    """Plain-language reading of each ablation row, derived from the numbers."""
    if row["gate_removed"] is None:
        return "Baseline: nothing leaks, nothing over-refuses"
    leaked = row["total_leakage"]
    if leaked:
        arm = max(row["leakage_by_arm"].items(), key=lambda kv: kv[1])[0]
        return f"{leaked} answers leak, all from {arm}"
    if row["grounded_answers_served"] > 16:
        return "Off-topic answers start being served"
    return "No incremental catch on this gold set \u2014 redundant here"


def lab_row(finding: dict) -> str:
    color = SEVERITY_COLOR.get(finding["severity"], MUTED)
    slope = finding["trend_slope_per_30d"]
    arrow = {"rising": "\u2191", "falling": "\u2193"}.get(finding["trend"], "\u2192")
    trend_note = f"{arrow} {finding['trend']}"
    if slope is not None:
        trend_note += f" &nbsp;{slope:+g}/30d"
    flag = ' <span class="flag">adverse</span>' if finding["clinically_adverse_trend"] else ""
    return f"""<tr>
  <td><div class="lab-name">{esc(finding['name'])}</div><div class="muted small">LOINC {esc(finding['loinc'])}</div></td>
  <td class="num"><span class="value" style="color:{color}">{esc(finding['latest_value'])}</span>
      <span class="muted small">{esc(finding['unit'])}</span></td>
  <td><span class="pill" style="color:{color};border-color:{color}33;background:{color}14">{esc(SEVERITY_LABEL.get(finding['severity'], finding['severity']))}</span></td>
  <td class="muted small">{esc(finding['reference'])}</td>
  <td class="small">{trend_note}{flag}</td>
  <td class="spark-cell">{sparkline(finding['series'], color)}</td>
</tr>"""


def build(report: dict) -> str:
    patient = report["patient"]
    demographics = patient["patient"]
    priority = patient["priority"]
    labs = patient["labs"]
    interactions = patient["interactions"]
    timeline = patient["timeline"]
    answers = report["answers"]
    ev = report["evaluation"]
    g, c = ev["grounded"]["summary"], ev["control"]["summary"]

    featured = next((a for a in answers if not a["refused"] and a["patient_id"]), answers[0])
    refused = next((a for a in answers if a["refused"]), None)

    banner_color = {"urgent": RED, "attention": ORANGE, "routine": GREEN}[priority["level"]]
    age = datetime.now().year - int(str(demographics["birth_date"])[:4])

    lab_rows = "\n".join(lab_row(f) for f in labs["findings"][:8])

    interaction_cards = "\n".join(
        f"""<div class="card interaction">
  <div class="row"><span class="pill" style="color:{INTERACTION_COLOR.get(i['severity'], MUTED)};border-color:{INTERACTION_COLOR.get(i['severity'], MUTED)}33;background:{INTERACTION_COLOR.get(i['severity'], MUTED)}14">{esc(i['severity'])}</span>
    <strong>{esc(' + '.join(i['drugs']))}{esc(' \u00b7 ' + i['condition'] if i.get('condition') else '')}</strong></div>
  <p class="effect">{esc(i['effect'])}</p>
  <p class="muted small"><strong>Mechanism.</strong> {esc(i['mechanism'])}</p>
  <p class="small"><strong>Action.</strong> {esc(i['action'])} <span class="muted">\u2014 {esc(i['source'])}</span></p>
</div>"""
        for i in interactions[:4]
    )

    dot = {"critical": RED, "abnormal": ORANGE, "attention": ORANGE, "info": BLUE}
    timeline_items = "\n".join(
        f"""<li><span class="dot" style="background:{dot.get(e['severity'], BLUE)}"></span>
  <span class="when">{esc(e['at'])}</span>
  <span class="what"><strong>{esc(e['title'])}</strong><br><span class="muted small">{esc(e['detail'])}</span></span></li>"""
        for e in timeline[:11]
    )

    evidence_items = "\n".join(
        f"""<li><span class="cite">[{n}]</span> <strong>{esc(e['doc_title'])}</strong> \u2014 {esc(e['section'])}
  <span class="muted small">{esc(e['source'])} \u00b7 {esc(e['chunk_id'])} \u00b7 BM25 {e['score']:.2f}</span></li>"""
        for n, e in enumerate(featured["evidence"], start=1)
    )

    check_rows = "\n".join(
        f"""<tr><td class="small">{esc(ch['sentence'][:150])}{'\u2026' if len(ch['sentence']) > 150 else ''}</td>
  <td class="num small">{esc(ch['citations'])}</td>
  <td class="num small" style="color:{GREEN if ch['grounded'] else RED}">{ch['support']:.2f}</td></tr>"""
        for ch in featured["sentence_checks"][:5]
    )

    metric_rows = "\n".join(
        f"""<tr><td>{esc(label)}</td><td class="num"><strong>{g[key]}</strong></td></tr>"""
        for key, label in [
            ("recall_at_k", "Retrieval recall@4 (labelled docs)"),
            ("precision_at_k", "Retrieval precision@4"),
            ("mrr", "Mean reciprocal rank"),
            ("answer_rate_on_answerable", "Answer rate on answerable cases"),
            ("mean_groundedness_on_answers", "Mean groundedness of answers"),
            ("mean_citation_coverage_on_answers", "Citation coverage of answers"),
            ("behaviour_accuracy", "Correct behaviour (answer vs refuse)"),
            ("unsafe_answers", "Unsafe answers served"),
            ("over_refusals", "Over-refusals (cost of the gates)"),
        ]
    )

    arms = report["evaluation"]["arms"]
    arm_notes = {
        "ungrounded-control": "Confident, uncited, from parametric memory",
        "numeric-tamper": "Real sentences, thresholds and doses shifted",
        "polarity-tamper": "Real sentences, recommendation negated",
    }
    redteam_rows = "\n".join(
        f"""<tr><td><div class="lab-name">{esc(name)}</div>
      <div class="muted small">{esc(arm_notes.get(name, ''))}</div></td>
  <td class="num"><strong>{arm['summary']['blocked']}/{arm['summary']['cases']}</strong></td>
  <td class="num"><span class="pill" style="color:{GREEN};border-color:{GREEN}33;background:{GREEN}14">{arm['summary']['leakage_rate']}</span></td>
  <td class="small muted">{esc(', '.join(f"{k} x{v}" for k, v in list(arm['summary']['refusal_reasons'].items())[:2]))}</td></tr>"""
        for name, arm in arms.items()
    )

    ablation_rows = "\n".join(
        f"""<tr>
  <td><span class="{'lab-name' if row['gate_removed'] is None else ''}">{esc(row['config'])}</span></td>
  <td class="num" style="color:{RED if row['total_leakage'] else GREEN}"><strong>{row['total_leakage']}</strong></td>
  <td class="num">{row['grounded_answers_served']}</td>
  <td class="num">{row['grounded_over_refusals']}</td>
  <td class="small muted">{esc(_ablation_note(row))}</td>
</tr>"""
        for row in report.get("ablation", [])
    )

    traces = report["traces"][:6]
    max_dur = max((t["durationMs"] for t in traces), default=1) or 1
    trace_rows = "\n".join(
        f"""<tr><td class="small mono">{esc(t['root'])}</td>
  <td class="num small">{t['spanCount']}</td>
  <td><div class="bar"><div style="width:{max(3, t['durationMs'] / max_dur * 100):.0f}%"></div></div></td>
  <td class="num small">{t['durationMs']:.2f} ms</td></tr>"""
        for t in traces
    )

    audit_rows = "\n".join(
        f"""<tr><td class="small mono">{esc(a['at'])}</td><td class="small">{esc(a['actor'])} <span class="muted">({esc(a['actor_role'])})</span></td>
  <td class="small">{esc(a['action'])}</td><td class="small muted">{esc(a['detail'])}</td>
  <td class="small mono muted">{esc((a['trace_id'] or '')[:12])}</td></tr>"""
        for a in report["audit"][:6]
    )

    gaps = ", ".join(g2["name"] for g2 in labs["monitoring_gaps"]) or "none"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nullius \u2014 Clinical Decision Support</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #fff; color: {INK};
    font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 40px 32px 72px; }}
  h1 {{ font-size: 30px; line-height: 1.2; margin: 0 0 6px; letter-spacing: -0.02em; }}
  h2 {{ font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }}
  h3 {{ font-size: 15px; text-transform: uppercase; letter-spacing: 0.08em; color: {MUTED};
    margin: 0 0 16px; font-weight: 600; }}
  p {{ margin: 0 0 10px; }}
  .muted {{ color: {MUTED}; }}
  .small {{ font-size: 14px; }}
  .mono {{ font-family: Menlo, Consolas, monospace; font-size: 13px; }}
  section {{ margin-top: 48px; }}
  header .sub {{ color: {MUTED}; max-width: 720px; }}
  .kicker {{ display: inline-block; font-size: 13px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: {BLUE}; margin-bottom: 12px; }}
  .banner {{ margin-top: 28px; border: 1px solid {banner_color}33; background: {banner_color}0f;
    border-radius: 12px; padding: 20px 24px; }}
  .banner .level {{ display: flex; align-items: center; gap: 10px; font-weight: 700;
    color: {banner_color}; text-transform: uppercase; letter-spacing: 0.08em; font-size: 14px; }}
  .banner ul {{ margin: 12px 0 0; padding-left: 20px; }}
  .banner li {{ margin-bottom: 4px; }}
  .grid {{ display: grid; gap: 16px; }}
  .cols-4 {{ grid-template-columns: repeat(4, 1fr); }}
  .cols-2 {{ grid-template-columns: 1.1fr 0.9fr; }}
  .stat {{ border: 1px solid {BORDER}; border-radius: 10px; padding: 16px 18px; background: #fff; }}
  .stat .n {{ font-size: 26px; font-weight: 650; letter-spacing: -0.02em; }}
  .stat .l {{ font-size: 13px; color: {MUTED}; }}
  .card {{ border: 1px solid {BORDER}; border-radius: 10px; padding: 18px 20px; background: #fff; }}
  .card.soft {{ background: {SOFT}; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align: left; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em;
    color: {MUTED}; font-weight: 600; padding: 0 12px 8px 0; border-bottom: 1px solid {BORDER}; }}
  td {{ padding: 12px 12px 12px 0; border-bottom: 1px solid {BORDER}; vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  .num {{ text-align: right; white-space: nowrap; }}
  th.num {{ text-align: right; }}
  th + th, td + td {{ padding-left: 18px; }}
  th:last-child, td:last-child {{ padding-right: 0; }}
  .value {{ font-size: 17px; font-weight: 650; }}
  .lab-name {{ font-weight: 600; }}
  .pill {{ display: inline-block; border: 1px solid; border-radius: 999px; padding: 2px 10px;
    font-size: 12px; font-weight: 600; white-space: nowrap; }}
  .flag {{ color: {ORANGE}; font-size: 12px; font-weight: 600; margin-left: 6px; }}
  .spark-cell {{ width: 140px; text-align: right; }}
  .spark {{ display: block; margin-left: auto; }}
  .row {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }}
  .interaction {{ margin-bottom: 12px; }}
  .interaction .effect {{ font-weight: 550; }}
  .interaction p:last-child {{ margin-bottom: 0; }}
  ul.timeline {{ list-style: none; margin: 0; padding: 0; }}
  ul.timeline li {{ display: flex; gap: 12px; padding: 10px 0; border-bottom: 1px solid {BORDER}; }}
  ul.timeline li:last-child {{ border-bottom: none; }}
  .dot {{ width: 8px; height: 8px; border-radius: 50%; margin-top: 8px; flex: 0 0 8px; }}
  .when {{ flex: 0 0 96px; color: {MUTED}; font-size: 13px; font-variant-numeric: tabular-nums; margin-top: 2px; }}
  .what {{ flex: 1; }}
  .answer {{ border-left: 3px solid {BLUE}; padding-left: 18px; margin: 0 0 18px; }}
  .answer p {{ margin-bottom: 12px; }}
  .cite {{ color: {BLUE}; font-weight: 700; }}
  ul.evidence {{ list-style: none; margin: 0; padding: 0; }}
  ul.evidence li {{ padding: 9px 0; border-bottom: 1px solid {BORDER}; }}
  ul.evidence li:last-child {{ border-bottom: none; }}
  .refusal {{ border: 1px solid {ORANGE}33; background: {ORANGE}0f; border-radius: 10px; padding: 18px 20px; }}
  .bar {{ background: {BORDER}; border-radius: 4px; height: 8px; width: 100%; overflow: hidden; }}
  .bar div {{ background: {BLUE}; height: 100%; border-radius: 4px; }}
  .meta {{ display: flex; gap: 24px; flex-wrap: wrap; font-size: 13px; color: {MUTED}; margin-top: 14px; }}
  footer {{ margin-top: 56px; padding-top: 20px; border-top: 1px solid {BORDER};
    color: {MUTED}; font-size: 13px; }}
  @media (max-width: 860px) {{
    .wrap {{ padding: 28px 20px 56px; }}
    .cols-4 {{ grid-template-columns: repeat(2, 1fr); }}
    .cols-2 {{ grid-template-columns: 1fr; }}
    .spark-cell {{ display: none; }}
    .when {{ flex: 0 0 82px; }}
  }}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <span class="kicker">Nullius \u00b7 clinical decision support</span>
    <h1>{esc(demographics['given'])} {esc(demographics['family'])}, {age}</h1>
    <p class="sub">{esc(', '.join(c2['display'] for c2 in patient['conditions']))} \u00b7
      MRN {esc(demographics['mrn'])} \u00b7 {len(patient['encounters'])} encounters \u00b7
      {len(labs['findings'])} distinct analytes reviewed</p>

    <div class="banner">
      <div class="level">{esc(priority['level'])} \u00b7 {priority['reason_count']} findings need attention</div>
      <ul>{''.join(f'<li>{esc(r)}</li>' for r in priority['reasons'][:4])}</ul>
    </div>
  </header>

  <section>
    <div class="grid cols-4">
      <div class="stat"><div class="n" style="color:{RED}">{len(labs['critical_values'])}</div><div class="l">Critical values</div></div>
      <div class="stat"><div class="n" style="color:{ORANGE}">{len(labs['adverse_trends'])}</div><div class="l">Adverse trends</div></div>
      <div class="stat"><div class="n" style="color:{RED}">{sum(1 for i in interactions if i['severity'] in ('major', 'contraindicated'))}</div><div class="l">Major interactions</div></div>
      <div class="stat"><div class="n">{len(labs['monitoring_gaps'])}</div><div class="l">Monitoring gaps</div></div>
    </div>
  </section>

  <section>
    <h2>Lab intelligence</h2>
    <p class="muted small">Reference ranges, deltas and least-squares trends computed deterministically \u2014 no model in this path. Missing from the monitoring panel: {esc(gaps)}.</p>
    <table>
      <thead><tr><th>Analyte</th><th class="num">Latest</th><th>Flag</th><th>Reference</th><th>Trend</th><th class="num">History</th></tr></thead>
      <tbody>
{lab_rows}
      </tbody>
    </table>
  </section>

  <section>
    <h2>Medication safety</h2>
    <p class="muted small">Curated rule table with stated mechanism and source. An LLM is never asked whether two drugs interact.</p>
{interaction_cards}
  </section>

  <section class="grid cols-2">
    <div>
      <h2>Copilot answer</h2>
      <p class="muted small">\u201c{esc(featured['question'])}\u201d</p>
      <div class="answer"><p>{esc(featured['answer'])}</p></div>
      <h3>Evidence actually used</h3>
      <ul class="evidence">
{evidence_items}
      </ul>
      <h3 style="margin-top:24px">Per-sentence verification</h3>
      <table>
        <thead><tr><th>Sentence</th><th class="num">Cites</th><th class="num">Support</th></tr></thead>
        <tbody>
{check_rows}
        </tbody>
      </table>
      <div class="meta">
        <span>groundedness <strong>{featured['groundedness']}</strong></span>
        <span>citation coverage <strong>{featured['citation_coverage']}</strong></span>
        <span>confidence <strong>{esc(featured['confidence'])}</strong></span>
        <span>trace <strong class="mono">{esc(featured['trace_id'][:12])}</strong></span>
      </div>
    </div>
    <div>
      <h2>Refusal</h2>
      <p class="muted small">Out-of-scope questions are declined rather than answered plausibly.</p>
      <div class="refusal">
        <p class="small muted">\u201c{esc(refused['question'] if refused else '')}\u201d</p>
        <p class="small"><strong>{esc(refused['answer'] if refused else '')}</strong></p>
        <p class="small muted" style="margin-bottom:0">reason: <span class="mono">{esc(refused['refusal_reason'] if refused else '')}</span></p>
      </div>
      <h3 style="margin-top:28px">Patient context injected</h3>
      <div class="card soft small">
        <p><strong>Critical:</strong> {esc(', '.join(featured['patient_context'].get('critical_values', [])) or 'none')}</p>
        <p><strong>Trends:</strong> {esc(', '.join(featured['patient_context'].get('adverse_trends', [])) or 'none')}</p>
        <p style="margin-bottom:0"><strong>Interactions:</strong> {esc('; '.join(featured['patient_context'].get('interactions', [])[:2]) or 'none')}</p>
      </div>
    </div>
  </section>

  <section>
    <h2>Patient timeline</h2>
    <p class="muted small">Encounters, diagnoses, prescriptions and abnormal results in one stream. Normal results are collapsed into per-encounter counts.</p>
    <ul class="timeline">
{timeline_items}
    </ul>
  </section>

  <section>
    <h2>Red team</h2>
    <p class="muted small">Three generators built to defeat the verification layer, each run against the full gold set. Every arm is untrustworthy by construction, so any answer that reaches a clinician is leakage \u2014 that is the number to watch, not the count of blocked attempts.</p>
    <table>
      <thead><tr><th>Arm</th><th class="num">Blocked</th><th class="num">Leakage rate</th><th>Caught by</th></tr></thead>
      <tbody>
{redteam_rows}
      </tbody>
    </table>
  </section>

  <section>
    <h2>Gate ablation</h2>
    <p class="muted small">One gate removed at a time, every arm re-run. This is what separates a safety claim from a safety story: a gate whose removal changes nothing is not protecting anything.</p>
    <table>
      <thead><tr><th>Configuration</th><th class="num">Leakage</th><th class="num">Answers served</th><th class="num">Over-refusals</th><th>Reading</th></tr></thead>
      <tbody>
{ablation_rows}
      </tbody>
    </table>
  </section>

  <section class="grid cols-2">
    <div>
      <h2>Evaluation</h2>
      <p class="muted small">{g['cases']} hand-labelled cases: answerable, unanswerable and adversarial. Both error types are reported separately \u2014 unsafe answers are the clinical risk, over-refusals are the price paid for avoiding them.</p>
      <table>
        <thead><tr><th>Metric</th><th class="num">Value</th></tr></thead>
        <tbody>
{metric_rows}
        </tbody>
      </table>
    </div>
    <div>
      <h2>Observability</h2>
      <p class="muted small">{len(report['traces'])} traces shown of the run; every span carries retrieval, token and quality attributes.</p>
      <table>
        <thead><tr><th>Root span</th><th class="num">Spans</th><th>Duration</th><th class="num"></th></tr></thead>
        <tbody>
{trace_rows}
        </tbody>
      </table>
      <h3 style="margin-top:28px">Audit trail</h3>
      <table>
        <thead><tr><th>When</th><th>Actor</th><th>Action</th><th>Detail</th><th>Trace</th></tr></thead>
        <tbody>
{audit_rows}
        </tbody>
      </table>
    </div>
  </section>

  <footer>
    Synthetic FHIR data \u2014 no real patient information. Decision support only: Nullius assists a licensed
    clinician and never diagnoses or prescribes. Cohort: {report['cohort']['patients']} patients,
    {report['cohort']['observations']} observations, {report['corpus']['chunks']} guideline chunks across
    {report['corpus']['documents']} documents. RBAC enforced: {report['rbac_enforced']}.
  </footer>
</div>
</body>
</html>
"""


def main() -> int:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    target = OUT / "dashboard.html"
    target.write_text(build(report), encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
