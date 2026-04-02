import { initializeApp } from "https://www.gstatic.com/firebasejs/10.14.0/firebase-app.js";
import {
  getFirestore, collection, query, orderBy, limit, onSnapshot
} from "https://www.gstatic.com/firebasejs/10.14.0/firebase-firestore.js";

const firebaseConfig = {
  apiKey: "AIzaSyCIRmSemzGMT_h6POosmYtrCSB-YfmDE6Q",
  authDomain: "surgicalai01.firebaseapp.com",
  projectId: "surgicalai01",
  storageBucket: "surgicalai01.firebasestorage.app",
  messagingSenderId: "297879646298",
  appId: "1:297879646298:web:d94058ceca8bf9754ea863",
  measurementId: "G-QG065MJN2K",
};

const fbApp = initializeApp(firebaseConfig);
const db = getFirestore(fbApp);

// --- DOM refs ---
const elConnStatus = document.getElementById("connection-status");
const elDeviceId = document.getElementById("device-id");
const elAppId = document.getElementById("app-id");
const elState = document.getElementById("device-state");
const elDetail = document.getElementById("device-detail");
const elEventsBody = document.getElementById("events-body");
const elEventCount = document.getElementById("event-count");

// --- Gateway status polling ---
let statusOk = false;

async function pollStatus() {
  try {
    const resp = await fetch("/status");
    if (!resp.ok) throw new Error(resp.statusText);
    const data = await resp.json();

    elDeviceId.textContent = data.device_id || "--";
    elAppId.textContent = data.app_id || "--";

    const state = (data.state || "UNKNOWN").toUpperCase();
    elState.textContent = state;
    elState.className = "state-pill state-" + state.toLowerCase();

    const d = data.display || {};
    const details = [];
    if (d.live_count !== undefined) details.push("Count: " + d.live_count + "/" + (d.target_count || "?"));
    if (d.sku) details.push("SKU: " + d.sku);
    if (d.checklist) details.push("Classes: " + JSON.stringify(d.checklist));
    if (d.total_count !== undefined) details.push("Total: " + d.total_count);
    elDetail.textContent = details.length > 0 ? details.join(" | ") : "--";

    if (!statusOk) {
      elConnStatus.textContent = "Connected";
      elConnStatus.className = "badge badge-green";
      statusOk = true;
    }
  } catch {
    elConnStatus.textContent = "Offline";
    elConnStatus.className = "badge badge-red";
    statusOk = false;
  }
}

setInterval(pollStatus, 2000);
pollStatus();

// --- Firestore event listeners ---
const MAX_EVENTS = 50;
const events = { batch: [], bundle: [], area: [] };

function formatTime(ts) {
  if (!ts) return "--";
  try {
    const d = new Date(ts);
    return d.toLocaleString("en-US", {
      month: "short", day: "numeric",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
      hour12: false,
    });
  } catch {
    return ts;
  }
}

function renderEvents() {
  const all = [
    ...events.batch.map(e => ({ ...e, _type: "batch" })),
    ...events.bundle.map(e => ({ ...e, _type: "bundle" })),
    ...events.area.map(e => ({ ...e, _type: "area" })),
  ];
  all.sort((a, b) => (b.timestamp || "").localeCompare(a.timestamp || ""));
  const display = all.slice(0, MAX_EVENTS);

  elEventCount.textContent = all.length + " events";

  if (display.length === 0) {
    elEventsBody.innerHTML = '<tr><td colspan="5" class="empty-msg">Waiting for events...</td></tr>';
    return;
  }

  elEventsBody.innerHTML = display.map(e => {
    const type = e._type;
    let typeLabel, sku, result, resultClass, details;

    if (type === "batch") {
      typeLabel = "Batch";
      sku = e.sku || "--";
      result = e.result || "--";
      resultClass = result === "PASS" ? "result-pass" : "result-fail";
      details = (e.detected_count || 0) + "/" + (e.target_count || 0);
    } else if (type === "bundle") {
      typeLabel = "Bundle";
      sku = e.sku || "--";
      result = e.result || "--";
      resultClass = result === "PASS" ? "result-pass" : "result-fail";
      const missing = e.missing_classes || [];
      details = missing.length > 0 ? "Missing: " + missing.join(", ") : "All present";
    } else {
      typeLabel = "Area";
      sku = e.location_name || "--";
      result = e.state || "--";
      resultClass = result === "ALERT" ? "result-alert" : "result-monitoring";
      details = "Count: " + (e.count || 0) + " (\u0394" + (e.delta || 0) + ")";
    }

    return `<tr>
      <td>${formatTime(e.timestamp)}</td>
      <td class="type-${type}">${typeLabel}</td>
      <td>${sku}</td>
      <td class="${resultClass}">${result}</td>
      <td>${details}</td>
    </tr>`;
  }).join("");
}

function listenCollection(name, key) {
  const q = query(collection(db, name), orderBy("timestamp", "desc"), limit(25));
  onSnapshot(q, (snap) => {
    events[key] = snap.docs.map(d => d.data());
    renderEvents();
  }, (err) => {
    console.warn("Firestore listener error (" + name + "):", err);
  });
}

listenCollection("inventory_batch_events", "batch");
listenCollection("inventory_bundle_events", "bundle");
listenCollection("inventory_area_snapshots", "area");
