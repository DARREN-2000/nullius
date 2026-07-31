import re

with open('docs/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

playground_html = '''
<section id="playground"><div class="wrap">
  <div class="kicker">Playground</div>
  <h2>Copilot Playground</h2>
  <p class="lede">Test the clinical copilot locally. Run <code>python -m nullius.api --nli</code> on your machine, then ask questions here. The site will connect to <code>http://127.0.0.1:8080</code>.</p>
  
  <div class="card">
    <div style="display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap">
      <select id="pg-role" style="padding:8px;border-radius:6px;border:1px solid var(--line);background:var(--panel-2);color:var(--ink)">
        <option value="clinician">Role: Clinician</option>
        <option value="nurse">Role: Nurse</option>
        <option value="radiologist">Role: Radiologist</option>
      </select>
      <input type="text" id="pg-patient" placeholder="Patient ID (e.g. pat-001)" value="pat-001" style="padding:8px;border-radius:6px;border:1px solid var(--line);background:var(--panel-2);color:var(--ink);width:160px">
    </div>
    <div style="display:flex;gap:12px;margin-bottom:12px">
      <input type="text" id="pg-q" placeholder="Ask a clinical question about this patient..." style="flex:1;padding:12px;border-radius:6px;border:1px solid var(--line);background:var(--panel-2);color:var(--ink);font-size:15px">
      <button id="pg-btn" style="padding:0 24px;border-radius:6px;border:none;background:var(--accent);color:#000;font-weight:600;cursor:pointer">Ask</button>
    </div>
    <div id="pg-res" style="display:none;background:var(--panel-2);border:1px solid var(--line);border-radius:8px;padding:16px;font-family:var(--mono);font-size:13px;white-space:pre-wrap;color:var(--ink)"></div>
  </div>
</div></section>
'''

playground_js = '''
document.getElementById("pg-btn").onclick = async () => {
  const q = document.getElementById("pg-q").value;
  const pat = document.getElementById("pg-patient").value;
  const role = document.getElementById("pg-role").value;
  const res = document.getElementById("pg-res");
  if(!q) return;
  res.style.display = "block";
  res.innerHTML = "Asking local Copilot (http://127.0.0.1:8080)...";
  try {
    const r = await fetch("http://127.0.0.1:8080/copilot/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-User": "playground", "X-Role": role },
      body: JSON.stringify({ question: q, patient_id: pat })
    });
    const data = await r.json();
    res.innerHTML = esc(JSON.stringify(data, null, 2));
  } catch(e) {
    res.innerHTML = "Error: " + esc(e.message) + "\\n\\nIs the API running locally on port 8080?";
  }
};
'''

if '<a href="#playground">' not in html:
    html = html.replace('<a href="#limits">', '<a href="#playground">Playground</a>\n    <a href="#limits">')
if 'id="playground"' not in html:
    html = html.replace('<section id="limits">', playground_html + '\n<section id="limits">')
if 'pg-btn' not in html:
    html = html.replace('</script>', playground_js + '\n</script>', 1)

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('Patched docs/index.html')
