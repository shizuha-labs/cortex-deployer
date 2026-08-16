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
const deployDlg = document.getElementById("deploy-dlg");
const adoptDlg = document.getElementById("adopt-dlg");
const logDlg = document.getElementById("log-dlg");
const recipeSel = document.getElementById("recipe");

let recipes = [];

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
    rowsEl.innerHTML = '<tr><td colspan="6" class="empty">No backends yet. Deploy a model or register an existing OpenAI URL.</td></tr>';
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
        <button data-act="logs" data-id="${b.id}">Logs</button>
        <button class="danger" data-act="del" data-id="${b.id}">Remove</button>
      </td>
    </tr>
  `).join("");
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

async function refresh() {
  const [host, list] = await Promise.all([api("/api/host"), api("/api/backends")]);
  const gpus = (host.gpus || []).map((g) => g.name).join(", ") || "no GPU detected";
  hostline.textContent = `${host.os} ${host.arch} · ${host.hostname} · ${gpus}`;
  render(list.backends || []);
}

async function loadRecipes() {
  const data = await api("/api/recipes");
  recipes = data.recipes || [];
  recipeSel.innerHTML = recipes.map((r) =>
    `<option value="${esc(r.file)}">${esc(r.file)} · ${esc(r.engine)}</option>`
  ).join("");
}

document.getElementById("btn-refresh").onclick = () => refresh().catch(console.error);
document.getElementById("btn-deploy").onclick = () => { document.getElementById("deploy-err").textContent = ""; deployDlg.showModal(); };
document.getElementById("btn-adopt").onclick = () => { document.getElementById("adopt-err").textContent = ""; adoptDlg.showModal(); };
document.getElementById("deploy-cancel").onclick = () => deployDlg.close();
document.getElementById("adopt-cancel").onclick = () => adoptDlg.close();
document.getElementById("log-close").onclick = () => logDlg.close();

document.getElementById("deploy-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const err = document.getElementById("deploy-err");
  err.textContent = "";
  const file = recipeSel.value;
  const rec = recipes.find((r) => r.file === file);
  try {
    const full = await fetch("/api/recipes").then((r) => r.json());
    const chosen = (full.recipes || []).find((r) => r.file === file) || rec;
    const recipeBody = {
      schema_version: "deployer.recipe.v1",
      name: chosen.name,
      engine: chosen.engine,
      executor: "process",
      quant: chosen.quant || "",
      model: {
        id: document.getElementById("served").value || chosen.served_name,
        served_name: document.getElementById("served").value || chosen.served_name,
        source: { kind: "local_path", path: document.getElementById("model-path").value },
      },
      launch: {
        host: "127.0.0.1",
        port: Number(document.getElementById("port").value) || 0,
        context_length: Number(document.getElementById("ctx").value) || chosen.context_length || 8192,
      },
    };
    await api("/api/backends", {
      method: "POST",
      body: JSON.stringify({
        kind: "recipe",
        recipe: recipeBody,
        model_path: document.getElementById("model-path").value,
        served_name: document.getElementById("served").value || chosen.served_name,
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

rowsEl.addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-act]");
  if (!btn) return;
  const id = btn.dataset.id;
  const act = btn.dataset.act;
  try {
    if (act === "start") await api(`/api/backends/${id}/start`, { method: "POST", body: "{}" });
    if (act === "stop") await api(`/api/backends/${id}/stop`, { method: "POST", body: "{}" });
    if (act === "del") await api(`/api/backends/${id}`, { method: "DELETE" });
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
