const canvas = document.getElementById("reserve-map");
const ctx = canvas.getContext("2d");
const metricsEl = document.getElementById("metrics");
const alertsEl = document.getElementById("alerts-list");
const cnpEl = document.getElementById("cnp-list");
const clockEl = document.getElementById("clock-info");
const timestampEl = document.getElementById("timestamp");
const droneCardsEl = document.getElementById("drone-cards");
const eventsEl = document.getElementById("events-info");

async function fetchState() {
  const response = await fetch(`state.json?ts=${Date.now()}`);
  if (!response.ok) {
    throw new Error("Failed to fetch state");
  }
  return response.json();
}

function updateTimestamp(state) {
  if (!state || !state.generated_at) {
    timestampEl.textContent = "Awaiting data…";
    return;
  }
  const clock = state.clock || {};
  const hour = clock.hour?.toString().padStart(2, "0");
  timestampEl.textContent = `Snapshot ${state.generated_at} · Sim Day ${clock.day} @ ${hour}:00`;
  clockEl.textContent = `Sim Day ${clock.day} · Hour ${hour}`;
}

function updateMetrics(state) {
  const metrics = state.metrics || {};
  const entries = [
    ["Alerts", metrics.alerts_total ?? 0],
    ["Dispatches", metrics.dispatch_total ?? 0],
    ["CNP Active", metrics.cnp_active ?? 0],
    ["Drones", metrics.drones_total ?? 0],
    ["Sensors", metrics.sensors_total ?? 0],
    ["Trackers", metrics.trackers_total ?? 0],
    ["Poachers", metrics.poachers_active ?? 0],
    ["Herds", metrics.herds_active ?? 0],
  ];
  metricsEl.innerHTML = entries
    .map(
      ([label, value]) => `
        <dt>${label}</dt>
        <dd>${value}</dd>
      `,
    )
    .join("");
}

function updateAlerts(state) {
  const alerts = (state.alerts && state.alerts.recent) || [];
  if (!alerts.length) {
    alertsEl.innerHTML = `<div class="item">No alerts yet.</div>`;
  } else {
    alertsEl.innerHTML = alerts
      .slice()
      .reverse()
      .map((alert) => {
        const coords = alert.pos ? `(${alert.pos[0]}, ${alert.pos[1]})` : "—";
        const confidence = alert.confidence ? `${Math.round(alert.confidence * 100)}%` : "n/a";
        return `
          <div class="item">
            <div class="title">
              <span>${alert.category || "unknown"} · ${alert.sensor || "sensor"}</span>
              <span class="meta">${alert.timestamp || ""}</span>
            </div>
            <div class="meta">ID: ${alert.id || "?"} · Pos: ${coords} · Confidence: ${confidence}</div>
          </div>
        `;
      })
      .join("");
  }

  const cnp = (state.alerts && state.alerts.cnp_pending) || [];
  if (!cnp.length) {
    cnpEl.innerHTML = `<div class="item">No open negotiations.</div>`;
  } else {
    cnpEl.innerHTML = cnp
      .map((entry) => {
        const coords = entry.pos ? `(${entry.pos[0]}, ${entry.pos[1]})` : "—";
        return `
          <div class="item">
            <div class="title">
              <span>${entry.category || "poacher"}</span>
              <span>${entry.received_proposals}/${entry.expected_proposals} proposals</span>
            </div>
            <div class="meta">Alert ${entry.alert_id} @ ${coords}</div>
          </div>
        `;
      })
      .join("");
  }
}

function updateDrones(state) {
  const drones = (state.agents && state.agents.drones) || [];
  if (!drones.length) {
    droneCardsEl.innerHTML = `<div class="item">No drones registered.</div>`;
    return;
  }
  droneCardsEl.innerHTML = drones
    .map((drone) => {
      const pct = Math.max(0, Math.min(100, drone.battery_pct ?? 0));
      const status = drone.status || "patrolling";
      const queue = drone.incident_queue || 0;
      const incident = drone.active_incident ? `Active: ${drone.active_incident.category}` : "Idle";
      return `
        <article class="drone-card">
          <div class="title">
            <strong>${drone.callsign || drone.jid}</strong>
          </div>
          <div class="status">${status} · queue ${queue}</div>
          <div class="battery-bar">
            <span style="width:${pct}%"></span>
          </div>
          <div class="meta">${pct}% battery • ${incident}</div>
        </article>
      `;
    })
    .join("");
}

function updateEvents(state) {
  const events = state.events || {};
  const poachers = events.poachers || [];
  const herds = events.herds || [];
  const sensors = (state.agents && state.agents.sensors) || [];
  const trackers = (state.agents && state.agents.trackers) || [];
  const ranger = state.agents && state.agents.ranger;

  const info = [];
  info.push(`<div class="item">Ranger fuel: ${Math.round((ranger?.fuel_level ?? 0) * 10) / 10}</div>`);
  info.push(`<div class="item">${sensors.length} sensors · ${trackers.length} trackers</div>`);
  info.push(`<div class="item">${poachers.length} poachers · ${herds.length} herds active</div>`);

  const poacherRows = poachers
    .slice(0, 4)
    .map((p) => `#${p.id.slice(-4)} @ (${p.position.join(", ")})`)
    .join(" • ");
  const herdRows = herds
    .slice(0, 4)
    .map((h) => `#${h.id.slice(-4)} @ (${h.center.join(", ")})`)
    .join(" • ");
  if (poacherRows) {
    info.push(`<div class="item">Poachers: ${poacherRows}</div>`);
  }
  if (herdRows) {
    info.push(`<div class="item">Herds: ${herdRows}</div>`);
  }
  eventsEl.innerHTML = info.join("");
}

function drawMap(state) {
  if (!ctx) {
    return;
  }
  const reserve = state.reserve || { width: 20, height: 20, no_fly: [] };
  const width = reserve.width || 20;
  const height = reserve.height || 20;
  const padding = 20;
  const cellSize = Math.min((canvas.width - padding * 2) / width, (canvas.height - padding * 2) / height);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#050b16";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  for (let x = 0; x <= width; x++) {
    const px = padding + x * cellSize;
    ctx.beginPath();
    ctx.moveTo(px, padding);
    ctx.lineTo(px, padding + height * cellSize);
    ctx.stroke();
  }
  for (let y = 0; y <= height; y++) {
    const py = padding + y * cellSize;
    ctx.beginPath();
    ctx.moveTo(padding, py);
    ctx.lineTo(padding + width * cellSize, py);
    ctx.stroke();
  }

  const drawCell = (x, y, color) => {
    ctx.fillStyle = color;
    ctx.fillRect(padding + x * cellSize, padding + y * cellSize, cellSize, cellSize);
  };

  (reserve.no_fly || []).forEach(([x, y]) => drawCell(x, y, "rgba(242,95,92,0.35)"));

  const drawPoint = (x, y, color, radius = cellSize * 0.3) => {
    const px = padding + x * cellSize + cellSize / 2;
    const py = padding + y * cellSize + cellSize / 2;
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  };

  const events = state.events || {};
  (events.poachers || []).forEach((p) => drawPoint(p.position[0], p.position[1], "rgba(242,95,92,0.9)", cellSize * 0.28));
  (events.herds || []).forEach((h) => drawPoint(h.center[0], h.center[1], "rgba(94,92,255,0.8)", cellSize * 0.32));

  const drones = (state.agents && state.agents.drones) || [];
  drones.forEach((d) => drawPoint(d.position[0], d.position[1], "rgba(48,188,237,0.9)"));

  const sensors = (state.agents && state.agents.sensors) || [];
  sensors.forEach((s) => drawPoint(s.position[0], s.position[1], "rgba(255,224,102,0.9)", cellSize * 0.25));

  const trackers = (state.agents && state.agents.trackers) || [];
  trackers.forEach((t) => drawPoint(t.position[0], t.position[1], "rgba(112,193,179,0.9)", cellSize * 0.22));

  const ranger = state.agents && state.agents.ranger;
  if (ranger) {
    drawPoint(ranger.position[0], ranger.position[1], "rgba(255,255,255,0.9)", cellSize * 0.3);
  }
}

async function refresh() {
  try {
    const state = await fetchState();
    drawMap(state);
    updateTimestamp(state);
    updateMetrics(state);
    updateAlerts(state);
    updateDrones(state);
    updateEvents(state);
  } catch (err) {
    console.error(err);
  } finally {
    setTimeout(refresh, 1500);
  }
}

refresh();

