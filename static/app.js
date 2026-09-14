// PrepAI client: record answers, speak questions, draw the orb. All interview logic lives on the server.
const $ = (sel) => document.querySelector(sel);
const screen = $("#screen");

const SILENCE_MS = 1600;   // pause length that ends an answer
const NO_SPEECH_MS = 10000; // give up listening if nothing is said
const MAX_ANSWER_MS = 180000;

const session = { id: null, budget: 0, answered: 0, startedAt: 0, clockTimer: 0 };
let phase = "idle";

// ------------------------------------------------------------------ helpers

async function api(path, body) {
  const res = await fetch(path, body ? { method: "POST", body } : undefined);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `Something went wrong (${res.status})`);
  return data;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text; // never innerHTML: transcript and report are untrusted text
  return node;
}

function show(view) {
  for (const id of ["setup", "interview", "results"]) $("#" + id).hidden = id !== view;
}

function setPhase(next, hint = "") {
  phase = next;
  screen.dataset.phase = next;
  const hints = { idle: "Tap to answer", speaking: "Tap to skip and answer", listening: "Listening. Pause when you're done", thinking: "Thinking" };
  $("#hint").textContent = hint || hints[next] || "";
  orbs.forEach((o) => (o.mode = next));
}

// ------------------------------------------------------------------ orb

class Orb {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.level = 0;
    this.target = 0;
    this.mode = "idle";
    this.seed = Math.random() * 100;
  }

  pulse(amount = 0.6) { this.target = Math.max(this.target, amount); }
  setLevel(v) { this.target = v; }

  draw(t) {
    const { canvas, ctx } = this;
    const size = canvas.clientWidth;
    if (!size) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(size * dpr)) canvas.width = canvas.height = Math.round(size * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    this.level += (this.target - this.level) * 0.18;
    if (this.mode !== "listening") this.target *= 0.9;
    const calm = matchMedia("(prefers-reduced-motion: reduce)").matches ? 0.3 : 1;
    const speed = (this.mode === "thinking" ? 1.8 : 0.7) * calm;
    const energy = (0.55 + Math.min(1, this.level) * 1.3) * calm;
    const cx = size / 2, cy = size / 2, R = size * 0.28; // max wobble stays inside the canvas
    const time = t / 1000 * speed + this.seed;

    // Translucent frilly membranes, largest first, each drifting at its own speed and direction.
    const LAYERS = 6;
    for (let layer = LAYERS - 1; layer >= 0; layer--) {
      const k = layer * 2.3;
      const spin = time * 0.12 * (layer % 2 ? 1 : -1);
      ctx.beginPath();
      for (let i = 0; i <= 140; i++) {
        const a = (i / 140) * Math.PI * 2;
        const wobble =
          0.1 * Math.sin(3 * a + time * 0.8 + k) +
          0.08 * Math.sin(5 * a - time * 1.1 + k * 1.6) +
          0.05 * Math.sin(7 * a + time * 1.5 - k) +
          0.03 * Math.sin(11 * a - time * 1.9 + k * 2.4);
        const r = R * (0.78 + layer * 0.055) * (1 + wobble * energy);
        const x = cx + Math.cos(a + spin) * r;
        const y = cy + Math.sin(a + spin) * r;
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.closePath();
      const g = ctx.createRadialGradient(cx - R * 0.3, cy - R * 0.35, R * 0.05, cx, cy, R * 1.3);
      g.addColorStop(0, "rgba(255,255,255,0.62)");
      g.addColorStop(0.5, "rgba(228,228,228,0.34)");
      g.addColorStop(1, "rgba(150,150,150,0.14)");
      ctx.fillStyle = g;
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.22)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }
}

const orbs = [...document.querySelectorAll("[data-orb]")].map((c) => new Orb(c));
const mainOrb = orbs.find((o) => o.canvas.id === "main-orb");
(function frame(t) {
  orbs.forEach((o) => o.canvas.offsetParent && o.draw(t));
  requestAnimationFrame(frame);
})(0);

// ------------------------------------------------------------------ speaking (karaoke)

let voice = null;
function pickVoice() {
  const voices = speechSynthesis.getVoices().filter((v) => v.lang.startsWith("en"));
  const prefer = [/natural/i, /google us english/i, /aria|jenny|guy/i, /samantha|daniel/i];
  for (const re of prefer) {
    const hit = voices.find((v) => re.test(v.name));
    if (hit) return hit;
  }
  return voices[0] || null;
}
if ("speechSynthesis" in window) speechSynthesis.onvoiceschanged = () => (voice = pickVoice());

function renderQuestion(text) {
  const lines = $("#lines");
  lines.replaceChildren();
  $("#question").classList.remove("settled");
  const words = [];
  let offset = 0;
  for (const word of text.split(/\s+/).filter(Boolean)) {
    const start = text.indexOf(word, offset);
    offset = start + word.length;
    const span = el("span", "", word);
    lines.append(span, " ");
    words.push({ span, start });
  }
  lines.style.transform = "translateY(0)";
  return words;
}

function highlight(words, index) {
  words.forEach(({ span }, i) => {
    span.className = i > index ? "" : i >= index - 1 ? "now" : "said";
  });
  const current = words[Math.max(0, index)]?.span;
  if (current) {
    const lineHeight = parseFloat(getComputedStyle(current).lineHeight);
    $("#lines").style.transform = `translateY(${-Math.max(0, current.offsetTop - lineHeight * 1)}px)`;
  }
}

function speak(text) {
  const words = renderQuestion(text);
  const settle = () => {
    $("#question").classList.add("settled");
    highlight(words, words.length);
  };
  if (!("speechSynthesis" in window)) {
    settle();
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.voice = voice || (voice = pickVoice());
    utterance.rate = 1.02;
    let startedAt = 0, fromBoundary = false, last = -1, done = false;
    const step = (index) => {
      if (index !== last) { last = index; highlight(words, index); mainOrb.pulse(0.45 + Math.random() * 0.35); }
    };
    // Many voices (e.g. Chrome's Google voices) never fire word boundaries, so estimate from elapsed time.
    const timer = setInterval(() => {
      if (!startedAt || fromBoundary) return;
      const char = ((performance.now() - startedAt) / 1000) * 14.5 * utterance.rate;
      step(words.filter((w) => w.start <= char).length - 1);
    }, 90);
    const finish = () => {
      if (done) return;
      done = true;
      clearInterval(timer);
      settle();
      resolve();
    };
    utterance.onstart = () => (startedAt = performance.now());
    utterance.onboundary = (e) => {
      if (e.name && e.name !== "word") return;
      fromBoundary = true;
      step(words.filter((w) => w.start <= e.charIndex).length - 1);
    };
    utterance.onend = finish;
    utterance.onerror = finish;
    speechSynthesis.cancel();
    speechSynthesis.speak(utterance);
    // Safety net: if the engine never starts (no voices installed), don't hang the interview.
    setTimeout(() => !startedAt && finish(), 2500);
    stopSpeaking = () => { speechSynthesis.cancel(); finish(); };
  });
}
let stopSpeaking = () => {};

// ------------------------------------------------------------------ listening (VAD)

let mic = null;
let stopListening = null;
let cancelListening = null;

async function openMic() {
  if (mic) return mic;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
  const ctx = new AudioContext();
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  ctx.createMediaStreamSource(stream).connect(analyser);
  const mimeType = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"].find((t) => MediaRecorder.isTypeSupported(t)) || "";
  return (mic = { stream, ctx, analyser, mimeType });
}

// Resolves with the recorded Blob, or null if nothing was said.
async function record() {
  const { ctx, analyser, stream, mimeType } = await openMic();
  await ctx.resume();
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  const chunks = [];
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.start(250);

  const samples = new Float32Array(analyser.fftSize);
  const began = performance.now();
  let floor = 0.008, spoke = false, loudFrames = 0, lastVoice = began, raf = 0;

  return new Promise((resolve) => {
    const finish = (keep) => {
      cancelAnimationFrame(raf);
      stopListening = cancelListening = null;
      mainOrb.setLevel(0);
      recorder.onstop = () => resolve(keep && spoke ? new Blob(chunks, { type: recorder.mimeType }) : null);
      recorder.stop();
    };
    stopListening = () => finish(true);
    cancelListening = () => finish(false);

    const tick = () => {
      analyser.getFloatTimeDomainData(samples);
      let sum = 0;
      for (const s of samples) sum += s * s;
      const rms = Math.sqrt(sum / samples.length);
      const now = performance.now();
      mainOrb.setLevel(Math.min(1, rms * 9));

      if (now - began < 350) floor = Math.min(0.05, Math.max(floor, rms)); // calibrate to room noise
      if (rms > Math.max(0.02, floor * 2.5)) {
        if (++loudFrames > 4) spoke = true; // ~70ms of sound, not a click
        lastVoice = now;
      } else loudFrames = 0;

      if (spoke && now - lastVoice > SILENCE_MS) return finish(true);
      if (!spoke && now - began > NO_SPEECH_MS) return finish(false);
      if (now - began > MAX_ANSWER_MS) return finish(true);
      raf = requestAnimationFrame(tick);
    };
    tick();
  });
}

// ------------------------------------------------------------------ interview flow

function addBubble(who, text) {
  const log = $("#log");
  log.append(el("div", `bubble ${who}`, text));
  log.scrollTop = log.scrollHeight;
}

function updateProgress() {
  $("#progress").style.transform = `scaleX(${session.budget ? session.answered / session.budget : 0})`;
}

function tickClock() {
  const s = Math.floor((Date.now() - session.startedAt) / 1000);
  const pad = (n) => String(n).padStart(2, "0");
  $("#clock").textContent = `${pad(Math.floor(s / 3600))} : ${pad(Math.floor(s / 60) % 60)} : ${pad(s % 60)}`;
}

async function askQuestion(question) {
  addBubble("ai", question);
  setPhase("speaking");
  await speak(question);
  if (phase !== "speaking") return;
  await new Promise((r) => setTimeout(r, 250)); // let the speaker tail die before the mic opens
  await listen();
}

async function listen() {
  let blob;
  try {
    setPhase("listening");
    blob = await record();
  } catch (err) {
    setPhase("idle", "Mic blocked. Allow it, or type in the transcript panel");
    return;
  }
  if (!blob) {
    // Cancelled because the answer was typed or the interview ended: leave the phase alone.
    if (phase === "listening") setPhase("idle", "Didn't hear anything. Tap when you're ready");
    return;
  }
  const form = new FormData();
  form.append("audio", blob, blob.type.includes("mp4") ? "answer.mp4" : blob.type.includes("ogg") ? "answer.ogg" : "answer.webm");
  await submit(form);
}

async function submit(form) {
  setPhase("thinking");
  $("#heard").textContent = "";
  try {
    const res = await api(`/api/interviews/${session.id}/answer`, form);
    if (res.transcript) {
      addBubble("me", res.transcript);
      $("#heard").textContent = `“${res.transcript}”`;
    }
    session.answered += 1;
    updateProgress();
    if (res.done) return showResults(session.id);
    await askQuestion(res.question);
  } catch (err) {
    setPhase("idle", err.message);
  }
}

$("#mic").addEventListener("click", () => {
  if (phase === "speaking") stopSpeaking();
  else if (phase === "listening") stopListening?.();
  else if (phase === "idle") listen();
});

$("#open-log").addEventListener("click", () => ($("#sheet").hidden = false));
$("#close-log").addEventListener("click", () => ($("#sheet").hidden = true));
$("#sheet").addEventListener("click", (e) => e.target.id === "sheet" && ($("#sheet").hidden = true));

$("#type-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = e.target.elements.text;
  if (!input.value.trim() || phase === "thinking") return;
  stopSpeaking();
  cancelListening?.();
  const form = new FormData();
  form.append("text", input.value.trim());
  input.value = "";
  $("#sheet").hidden = true;
  submit(form);
});

let endArmed = false;
$("#end-btn").addEventListener("click", async (e) => {
  if (!endArmed) {
    endArmed = true;
    e.target.textContent = "Tap again to end and get your report";
    return;
  }
  stopSpeaking();
  cancelListening?.();
  $("#sheet").hidden = true;
  setPhase("thinking", "Wrapping up");
  try {
    await api(`/api/interviews/${session.id}/end`, new FormData());
    showResults(session.id);
  } catch (err) {
    setPhase("idle", err.message);
  }
});

// ------------------------------------------------------------------ setup

const fileInput = $("#start-form").elements.resume;
fileInput.addEventListener("change", () => ($("#file-name").textContent = fileInput.files[0]?.name || "Upload resume"));
const drop = $("#drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    fileInput.dispatchEvent(new Event("change"));
  }
});

$("#start-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = new FormData(e.target);
  $("#setup-error").textContent = "";
  $("#setup-form").hidden = true;
  $("#setup-loading").hidden = false;
  const steps = ["Reading your resume", "Following the links inside it", "Picking what to dig into", "Preparing your first question"];
  let step = 0;
  const cycle = setInterval(() => ($("#loading-text").textContent = steps[Math.min(++step, steps.length - 1)]), 2200);
  // Unlock audio inside the click so the first question can play after the network wait.
  if ("speechSynthesis" in window) speechSynthesis.speak(new SpeechSynthesisUtterance(""));

  try {
    const res = await api("/api/interviews", form);
    Object.assign(session, { id: res.id, budget: res.budget, answered: 0, startedAt: Date.now() });
    history.replaceState(null, "", `#${res.id}`);
    $("#role-title").textContent = res.role;
    updateProgress();
    tickClock();
    clearInterval(session.clockTimer);
    session.clockTimer = setInterval(tickClock, 1000);
    show("interview");
    askQuestion(res.question);
  } catch (err) {
    $("#setup-error").textContent = err.message;
    $("#setup-form").hidden = false;
  } finally {
    clearInterval(cycle);
    $("#setup-loading").hidden = true;
    $("#loading-text").textContent = steps[0];
  }
});

// ------------------------------------------------------------------ report

async function showResults(id) {
  clearInterval(session.clockTimer);
  stopSpeaking();
  setPhase("idle", "");
  show("results");
  $("#report").hidden = true;
  $("#report-loading").hidden = false;
  try {
    const data = await api(`/api/interviews/${id}`);
    if (!data.done) { // reopened mid-interview: nothing to score yet
      $("#report-loading").hidden = true;
      $("#report").hidden = false;
      $("#report").replaceChildren(el("p", "", "This interview isn't finished yet."));
      return;
    }
    renderReport(data);
  } catch (err) {
    $("#report").replaceChildren(el("p", "error", err.message));
    $("#report").hidden = false;
  }
  $("#report-loading").hidden = true;
}

const MOVE_LABEL = { open: "Opener", probe_deeper: "Dug deeper", escalate: "Harder follow-up", verify: "Checked resume", redirect: "Redirected", switch_topic: "New topic" };

function tone(v, max) {
  const r = v / max;
  return r >= 0.7 ? "good" : r >= 0.45 ? "mid" : "low";
}

function renderReport(data) {
  const r = data.report || {};
  const root = $("#report");
  root.replaceChildren();
  $("#report-title").textContent = data.role;
  $("#report-clock").textContent = "INTERVIEW REPORT";

  if (!data.turns.length) {
    root.append(el("p", "", "You ended before answering anything, so there's nothing to score."));
    root.hidden = false;
    return;
  }

  const score = el("div", "score");
  score.append(el("b", "", data.score.toFixed(1)), el("span", "", "/ 10"));
  const verdict = el("span", `pill ${tone(data.score, 10)}`, r.verdict || "");
  root.append(score, verdict);
  if (r.summary) { root.append(el("h3", "", "Summary"), el("p", "", r.summary)); }

  if (r.competencies?.length) {
    root.append(el("h3", "", "Competencies"));
    for (const c of r.competencies) {
      const row = el("div", "comp");
      const bar = el("div", "track");
      const fill = el("i");
      fill.style.transform = `scaleX(${c.score / 10})`;
      bar.append(fill);
      row.append(el("span", "", c.name), el("span", "", `${c.score}/10`), bar, el("small", "", c.evidence));
      root.append(row);
    }
  }

  for (const [title, items] of [["Strengths", r.strengths], ["Gaps", r.gaps]]) {
    if (!items?.length) continue;
    const ul = el("ul", "list");
    items.forEach((item) => ul.append(el("li", "", item)));
    root.append(el("h3", "", title), ul);
  }

  if (r.rewrites?.length) {
    root.append(el("h3", "", "Answer it better"));
    for (const w of r.rewrites) {
      const card = el("div", "card");
      card.append(el("div", "q", w.question), el("div", "label", "Missing"), el("p", "", w.what_was_missing), el("div", "label", "Stronger answer"), el("p", "", w.stronger_answer));
      root.append(card);
    }
  }

  root.append(el("h3", "", "Transcript"));
  data.turns.forEach((t) => {
    const card = el("div", "card");
    const tags = el("div", "tags");
    const quality = ["specificity", "ownership", "depth", "impact", "clarity"].reduce((s, k) => s + t.grade[k], 0) / 5;
    tags.append(el("span", "pill", MOVE_LABEL[t.move] || t.move), el("span", `pill ${tone(quality, 5)}`, `${quality.toFixed(1)} / 5`));
    card.append(tags, el("div", "q", t.question), el("p", "", t.answer));
    if (t.grade.note) card.append(el("small", "", t.grade.note));
    root.append(card);
  });
  root.hidden = false;
}

$("#again").addEventListener("click", () => {
  history.replaceState(null, "", location.pathname);
  location.reload();
});

// Reopen a report by URL hash.
if (location.hash.length > 10) showResults(location.hash.slice(1));
else setPhase("idle", "");
