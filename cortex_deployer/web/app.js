async function api(path, opts) {
  const res = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...opts,
  });
  const text = await res.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch { data = { error: text }; }
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

const rowsEl = document.getElementById("rows");
const countEl = document.getElementById("count");
const statsEl = document.getElementById("stats");
const hostline = document.getElementById("hostline");
const fitline = document.getElementById("fitline");
const binwarn = document.getElementById("binwarn");
const deployDlg = document.getElementById("deploy-dlg");
const adoptDlg = document.getElementById("adopt-dlg");
const logDlg = document.getElementById("log-dlg");
const connectDlg = document.getElementById("connect-dlg");
const recipeSel = document.getElementById("recipe");

let recipes = [];

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function pill(backend) {
  if (backend.healthy) return '<span class="dot ok"></span>healthy';
  if (backend.state === "starting") return '<span class="dot warn"></span>starting';
  if (backend.state === "running") return '<span class="dot warn"></span>running';
  return '<span class="dot bad"></span>' + (backend.state || "stopped");
}

function render(backends) {
  countEl.textContent = backends.length + " total";
  const healthy = backends.filter((b) => b.healthy).length;
  statsEl.innerHTML = `
    <div class="stat"><b>${backends.length}</b><span>backends</span></div>
    <div class="stat"><b>${healthy}</b><span>healthy</span></div>
    <div class="stat"><b>${backends.filter((b) => b.state === "running" || b.healthy).length}</b><span>live</span></div>
  `;
  if (!backends.length) {
    rowsEl.innerHTML = '<tr><td colspan="6" class="empty">No backends yet. Click <b>Choose a Qwen build</b> — pick a quant; live VRAM marks one as recommended.</td></tr>';
    return;
  }
  rowsEl.innerHTML = backends.map((b) => `
    <tr>
      <td>
        <strong>${esc(b.served_name || b.name)}</strong>
        <div class="muted">${esc(b.kind || "managed")}${b.quant ? " · " + esc(b.quant) : ""}</div>
      </td>
      <td>${esc(b.engine || "—")}</td>
      <td>${pill(b)}</td>
      <td><code>${esc(b.base_url || "—")}</code></td>
      <td>${b.context_length || "—"}</td>
      <td class="actions">
        <button data-act="start" data-id="${b.id}">Start</button>
        <button data-act="stop" data-id="${b.id}">Stop</button>
        <button data-act="connect" data-id="${b.id}" data-name="${esc(b.served_name || b.name)}">Cortex</button>
        <button data-act="logs" data-id="${b.id}">Logs</button>
        <button class="danger" data-act="del" data-id="${b.id}">Remove</button>
      </td>
    </tr>
  `).join("");
}

function selectedRecipe() {
  return recipes.find((r) => r.file === recipeSel.value);
}

function paintRecipeNote() {
  const rec = selectedRecipe();
  const note = document.getElementById("recipe-note");
  if (!rec) { note.textContent = ""; return; }
  const bits = [rec.fit, rec.min_vram_mb ? `needs ≥${rec.min_vram_mb} MB` : "", rec.notes || ""];
  note.textContent = bits.filter(Boolean).join(" · ");
  if (rec.served_name && !document.getElementById("served").value) {
    document.getElementById("served").placeholder = rec.served_name;
  }
  const ctx = document.getElementById("ctx");
  if (ctx && rec.context_length && !ctx.dataset.touched) {
    ctx.value = rec.context_length;
  }
}

function paintChatModels(backends) {
  const sel = document.getElementById("chat-model");
  if (!sel) return;
  const prev = sel.value;
  const live = (backends || []).filter((b) => b.healthy || b.state === "running");
  const opts = live.length ? live : (backends || []);
  sel.innerHTML = opts.map((b) => {
    const id = b.served_name || b.name || b.id;
    return `<option value="${esc(id)}">${esc(id)}</option>`;
  }).join("") || '<option value="">(no backends)</option>';
  if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;
}

function paintLocal(models) {
  const el = document.getElementById("local-models");
  if (!el) return;
  if (!models || !models.length) {
    el.textContent = "none yet — download from Deploy model";
    return;
  }
  el.innerHTML = models.map((m) =>
    `<div><code>${esc(m.path)}</code> <span class="muted">${Math.round((m.bytes || 0) / 1048576)} MB</span></div>`
  ).join("");
}

async function refresh() {
  const [host, list, dls] = await Promise.all([
    api("/api/host"),
    api("/api/backends"),
    api("/api/downloads").catch(() => ({ local: [] })),
  ]);
  const gpus = (host.gpus || []).map((g) => `${g.name}${g.memory_mb ? " " + g.memory_mb + "MB" : ""}`).join(", ") || "no GPU detected";
  hostline.textContent = `${host.os} ${host.arch} · ${host.hostname} · ${gpus}`;
  const llama = (host.binaries || {}).llamacpp;
  if (binwarn) {
    if (llama) {
      binwarn.textContent = "";
    } else {
      binwarn.textContent = "llama-server not found yet — choosing a build installs the official CUDA binary automatically.";
    }
  }
  render(list.backends || []);
  paintChatModels(list.backends || []);
  paintLocal(dls.local || []);
}

async function loadRecipes() {
  const data = await api("/api/recommend");
  recipes = data.recipes || [];
  recipeSel.innerHTML = recipes.map((r) =>
    `<option value="${esc(r.file)}">${esc(r.fit)} · ${esc(r.name)} · ${esc(r.quant || r.engine)}</option>`
  ).join("");
  if (data.best) recipeSel.value = data.best;
  const vram = data.vram_mb ? `${data.vram_mb} MB NVIDIA` : (data.apple ? "Apple Silicon" : "no dedicated NVIDIA VRAM");
  fitline.textContent = `GPU fit: ${vram}. Recommended recipe: ${data.best || "none"}. A 16 GB card should use Q3, not Q4.`;
  paintRecipeNote();
}

document.getElementById("btn-refresh").onclick = () => refresh().catch(console.error);
document.getElementById("btn-deploy").onclick = () => { document.getElementById("deploy-err").textContent = ""; deployDlg.showModal(); };
document.getElementById("ctx").addEventListener("input", () => { document.getElementById("ctx").dataset.touched = "1"; });

async function openPicker() {
  const rec = await api("/api/recommend");
  const live = document.getElementById("pick-live");
  const gpu = rec.vram_mb ? `${rec.vram_mb} MB NVIDIA` : (rec.apple ? "Apple Silicon" : "no NVIDIA VRAM detected");
  live.textContent = `Live: ${gpu}. Recommended from catalog${rec.catalog_live ? " (live)" : ""}: ${rec.best || "none"}.`;
  const qwen = (rec.models || []).find((m) => m.id === "qwen3.8-27b") || { quants: [] };
  const box = document.getElementById("pick-rows");
  box.innerHTML = (qwen.quants || []).map((q) => {
    const recd = q.recipe === rec.best || q.id === qwen.recommended_quant;
    const skip = q.fit === "skip";
    return `<div class="pick-card ${recd ? "rec" : ""} ${skip ? "skip" : ""}">
      <div>
        <strong>${esc(q.quant)}</strong> ${recd ? '<span class="muted">recommended</span>' : ""}
        <div class="muted">${esc(q.fit)} · ≥${q.min_vram_mb || 0} MB · ${q.weight_gb || "?"} GB · ctx ${q.context_length || "—"}</div>
        <div class="muted">${esc(q.notes || "")}</div>
      </div>
      <button class="primary" data-recipe="${esc(q.recipe || "")}" ${skip ? "disabled" : ""}>${skip ? "won't fit" : "Use this"}</button>
    </div>`;
  }).join("") || '<p class="muted">No Qwen quants in catalog.</p>';
  document.getElementById("pick-dlg").showModal();
}

let lastRecipe = "";

function showHfBox(err) {
  const box = document.getElementById("hf-box");
  const msg = String(err || "");
  const hit = /403|429|rate limit|Hugging Face HTTP/i.test(msg);
  box.hidden = !hit;
  return hit;
}

async function runSetup(recipe) {
  const status = document.getElementById("setup-status");
  const btn = document.getElementById("btn-setup");
  lastRecipe = recipe || "";
  status.textContent = "starting " + (recipe || "recommended") + "…";
  btn.disabled = true;
  try {
    const job = await api("/api/setup", {
      method: "POST",
      body: JSON.stringify({ recipe: recipe || "" }),
    });
    const tick = async () => {
      const cur = await api("/api/setup/" + job.id);
      if (cur.state === "done") {
        status.textContent = "ready " + (cur.served_name || "") + " @ " + (cur.base_url || "");
        btn.disabled = false;
        await refresh();
        return;
      }
      if (cur.state === "error") {
        status.textContent = cur.error || "setup failed";
        showHfBox(cur.error);
        btn.disabled = false;
        return;
      }
      status.textContent = (cur.step || cur.state) + (cur.binary ? " · engine ready" : "");
      setTimeout(tick, 1500);
    };
    tick();
  } catch (e) {
    status.textContent = e.message;
    btn.disabled = false;
  }
}
document.getElementById("btn-setup").onclick = () => openPicker().catch((e) => {
  document.getElementById("setup-status").textContent = e.message;
  showHfBox(e.message);
});
document.getElementById("hf-save").onclick = async () => {
  const token = document.getElementById("hf-token").value.trim();
  const status = document.getElementById("setup-status");
  if (!token) { status.textContent = "paste an HF token first"; return; }
  try {
    await api("/api/settings", { method: "POST", body: JSON.stringify({ hf_token: token }) });
    document.getElementById("hf-box").hidden = true;
    status.textContent = "token saved — retrying…";
    runSetup(lastRecipe);
  } catch (e) {
    status.textContent = e.message;
  }
};
document.getElementById("pick-cancel").onclick = () => document.getElementById("pick-dlg").close();
document.getElementById("pick-rows").addEventListener("click", (ev) => {
  const btn = ev.target.closest("button[data-recipe]");
  if (!btn || btn.disabled) return;
  document.getElementById("pick-dlg").close();
  runSetup(btn.dataset.recipe);
});
document.getElementById("btn-adopt").onclick = () => { document.getElementById("adopt-err").textContent = ""; adoptDlg.showModal(); };
document.getElementById("deploy-cancel").onclick = () => deployDlg.close();
document.getElementById("adopt-cancel").onclick = () => adoptDlg.close();
document.getElementById("log-close").onclick = () => logDlg.close();
document.getElementById("connect-cancel").onclick = () => connectDlg.close();
recipeSel.onchange = paintRecipeNote;

document.getElementById("btn-dl").onclick = async () => {
  const rec = selectedRecipe();
  const status = document.getElementById("dl-status");
  if (!rec || !rec.download_repo) {
    status.textContent = "This recipe has no Hugging Face repo.";
    return;
  }
  status.textContent = "starting download…";
  try {
    const job = await api("/api/downloads", {
      method: "POST",
      body: JSON.stringify({
        repo: rec.download_repo,
        filename: rec.download_filename || "",
        glob: rec.download_glob || "",
      }),
    });
    const tick = async () => {
      const all = await api("/api/downloads");
      const cur = (all.jobs || []).find((j) => j.id === job.id);
      if (!cur) return;
      if (cur.state === "done") {
        status.textContent = "saved " + cur.path;
        if (cur.path) document.getElementById("model-path").value = cur.path;
        return;
      }
      if (cur.state === "error") {
        status.textContent = cur.error || "download failed";
        return;
      }
      status.textContent = `${cur.state} ${cur.filename || ""} ${cur.bytes || 0}/${cur.total || "?"}`;
      setTimeout(tick, 1500);
    };
    tick();
  } catch (e) {
    status.textContent = e.message;
  }
};

document.getElementById("deploy-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const err = document.getElementById("deploy-err");
  err.textContent = "";
  const rec = selectedRecipe();
  try {
    const recipeBody = {
      schema_version: "deployer.recipe.v1",
      name: rec.name,
      engine: rec.engine,
      executor: "process",
      quant: rec.quant || "",
      min_vram_mb: rec.min_vram_mb || 0,
      model: {
        id: document.getElementById("served").value || rec.served_name,
        served_name: document.getElementById("served").value || rec.served_name,
        source: { kind: "local_path", path: document.getElementById("model-path").value },
      },
      launch: {
        host: "127.0.0.1",
        port: Number(document.getElementById("port").value) || 0,
        context_length: Number(document.getElementById("ctx").value) || rec.context_length || 8192,
      },
    };
    await api("/api/backends", {
      method: "POST",
      body: JSON.stringify({
        kind: "recipe",
        recipe: recipeBody,
        model_path: document.getElementById("model-path").value,
        served_name: document.getElementById("served").value || rec.served_name,
        context_length: Number(document.getElementById("ctx").value) || undefined,
        port: Number(document.getElementById("port").value) || undefined,
        autostart: true,
      }),
    });
    deployDlg.close();
    await refresh();
  } catch (e) {
    err.textContent = e.message;
  }
};

document.getElementById("adopt-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const err = document.getElementById("adopt-err");
  err.textContent = "";
  try {
    await api("/api/backends", {
      method: "POST",
      body: JSON.stringify({
        kind: "adopt",
        model_id: document.getElementById("adopt-name").value,
        base_url: document.getElementById("adopt-url").value,
      }),
    });
    adoptDlg.close();
    await refresh();
  } catch (e) {
    err.textContent = e.message;
  }
};

document.getElementById("connect-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const err = document.getElementById("connect-err");
  err.textContent = "";
  const id = document.getElementById("connect-id").value;
  try {
    const gw = document.getElementById("connect-gw").value;
    if (gw) localStorage.setItem("cortex_deployer_gateway", gw);
    await api(`/api/backends/${id}/connect`, {
      method: "POST",
      body: JSON.stringify({
        gateway: gw,
        token: document.getElementById("connect-tok").value,
        model: document.getElementById("connect-model").value,
      }),
    });
    connectDlg.close();
    await refresh();
  } catch (e) {
    err.textContent = e.message;
  }
};

document.getElementById("chat-send").onclick = async () => {
  const out = document.getElementById("chat-out");
  out.textContent = "";
  const model = (document.getElementById("chat-model") || {}).value || "";
  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model,
        stream: true,
        messages: [{ role: "user", content: document.getElementById("chat-in").value }],
      }),
    });
    const ctype = res.headers.get("content-type") || "";
    if (!res.ok || !ctype.includes("event-stream")) {
      const text = await res.text();
      let data = {};
      try { data = text ? JSON.parse(text) : {}; } catch { data = { error: text }; }
      if (data.choices) {
        out.textContent = (((data.choices || [])[0] || {}).message || {}).content || text;
        return;
      }
      throw new Error(data.error || res.statusText || text);
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let acc = "";
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split("\n");
      buf = parts.pop();
      for (const line of parts) {
        const s = line.trim();
        if (!s.startsWith("data:")) continue;
        const payload = s.slice(5).trim();
        if (payload === "[DONE]") continue;
        try {
          const j = JSON.parse(payload);
          acc += (((j.choices || [])[0] || {}).delta || {}).content || "";
          out.textContent = acc;
        } catch { /* ignore keepalives */ }
      }
    }
    if (!acc) out.textContent = out.textContent || "(empty)";
  } catch (e) {
    out.textContent = e.message;
  }
};

rowsEl.addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-act]");
  if (!btn) return;
  const id = btn.dataset.id;
  const act = btn.dataset.act;
  try {
    if (act === "start") await api(`/api/backends/${id}/start`, { method: "POST", body: "{}" });
    if (act === "stop") await api(`/api/backends/${id}/stop`, { method: "POST", body: "{}" });
    if (act === "del") await api(`/api/backends/${id}`, { method: "DELETE" });
    if (act === "connect") {
      document.getElementById("connect-id").value = id;
      document.getElementById("connect-model").value = btn.dataset.name || "";
      document.getElementById("connect-err").textContent = "";
      const saved = localStorage.getItem("cortex_deployer_gateway");
      if (saved && !document.getElementById("connect-gw").value) {
        document.getElementById("connect-gw").value = saved;
      }
      connectDlg.showModal();
      return;
    }
    if (act === "logs") {
      const data = await api(`/api/backends/${id}/logs`);
      document.getElementById("log-body").textContent = data.log || "(empty)";
      logDlg.showModal();
    }
    await refresh();
  } catch (e) {
    alert(e.message);
  }
});

loadRecipes().then(refresh).catch((e) => { hostline.textContent = e.message; });
setInterval(() => refresh().catch(() => {}), 4000);
