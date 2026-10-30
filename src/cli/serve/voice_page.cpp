// A self-contained browser client of the realtime conversation WebSocket:
// microphone capture (with the browser's echo cancellation and noise
// suppression), hands-free turns detected by the server or push to talk,
// streamed reply speech and transcript, interruption (cancel and truncate to
// what was heard), typed messages and function tools.
#include <string_view>

#include "src/cli/serve/audio_websocket.hpp"

namespace gufo::server {

std::string_view VoicePage() {
  // Raw literals in pieces under MSVC's 16 KiB limit per literal.
  return R"html(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gufo voice</title>
<style>
:root { --bg: #f6f6f4; --fg: #1d1d1b; --muted: #6b6b66; --card: #fff;
  --line: #ddddd8; --accent: #2f6fdb; --user: #e8eefb; --bot: #fff; --warn: #b3261e; }
@media (prefers-color-scheme: dark) { :root { --bg: #171716; --fg: #ececea;
  --muted: #9a9a94; --card: #222221; --line: #3a3a38; --accent: #7aa7ff;
  --user: #233150; --bot: #222221; --warn: #ff8a80; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.45 system-ui, sans-serif; }
main { max-width: 760px; margin: 0 auto; padding: 16px; }
h1 { font-size: 20px; margin: 4px 0 12px; }
details, .bar, #log, form { background: var(--card); border: 1px solid var(--line);
  border-radius: 10px; padding: 12px; margin-bottom: 12px; }
summary { cursor: pointer; color: var(--muted); }
label { display: block; margin: 10px 0 4px; color: var(--muted); font-size: 13px; }
input, textarea, select { width: 100%; font: inherit; color: inherit;
  background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; }
textarea { min-height: 70px; font-family: ui-monospace, monospace; font-size: 13px; }
.bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
button { font: inherit; border: 0; border-radius: 8px; padding: 8px 14px;
  background: var(--accent); color: #fff; cursor: pointer; }
button.secondary { background: var(--line); color: var(--fg); }
button:disabled { opacity: .45; cursor: default; }
#talk { flex: 1 1 100%; padding: 18px; font-size: 17px; user-select: none; touch-action: none; }
#talk.live { background: var(--warn); }
#status { color: var(--muted); font-size: 13px; flex: 1; min-width: 160px; }
#meter { height: 6px; width: 120px; background: var(--line); border-radius: 3px; overflow: hidden; }
#meter div { height: 100%; width: 0; background: var(--accent); }
#log { min-height: 160px; display: flex; flex-direction: column; gap: 8px; }
.msg { padding: 8px 10px; border-radius: 8px; border: 1px solid var(--line); white-space: pre-wrap; }
.user { background: var(--user); align-self: flex-end; max-width: 85%; }
.bot { background: var(--bot); max-width: 92%; }
.note { color: var(--muted); font-size: 13px; }
.call { font-family: ui-monospace, monospace; font-size: 13px; }
.cut { text-decoration: line-through; opacity: .6; }
form { display: flex; gap: 8px; }
form input { flex: 1; }
</style>
</head>
<body>
<main>
<h1>Gufo voice</h1>
<details>
<summary>Settings</summary>
<label for="key">API key (only if the server uses --api-key)</label>
<input id="key" type="password" autocomplete="off">
<label for="mode">Turns</label>
<select id="mode">
<option value="vad">Hands-free: the server hears when you start and stop</option>
<option value="ptt">Push to talk: hold the button (or Space)</option>
</select>
<label><input id="duplex" type="checkbox" style="width:auto"> I use headphones: keep the
microphone open while the reply plays, so I can talk over it</label>
<label for="output">Replies</label>
<select id="output"><option value="audio">Speech with transcript</option>
<option value="text">Text</option></select>
<label for="silence">Silence that ends a turn (ms)</label>
<input id="silence" type="number" min="200" max="3000" step="50" value="600">
<label for="instructions">Instructions (empty: the model card prompt)</label>
<textarea id="instructions"></textarea>
<label for="tools">Function tools (JSON array of {"type":"function","name",...}); calls appear below</label>
<textarea id="tools" placeholder='[{"type":"function","name":"set_volume","description":"Set the speaker volume.","parameters":{"type":"object","properties":{"volume":{"type":"integer"}},"required":["volume"]}}]'></textarea>
</details>
<div class="bar">
<button id="connect">Connect</button>
<span id="status">Disconnected. The microphone needs localhost or HTTPS.</span>
<div id="meter"><div></div></div>
<button id="talk" disabled>Connect first</button>
</div>
<div id="log"></div>
<form id="typed"><input id="text" placeholder="Type a message" disabled>
<button id="send" disabled>Send</button></form>
</main>
)html"
         R"html(<script>
"use strict";
const $ = (id) => document.getElementById(id);
const RATE = 24000;
let ws = null, mic = null, captureCtx = null, playCtx = null, node = null;
let streaming = false, holding = false, pending = [];
let reply = null;            // {item, el, text, cut}
let play = { item: null, start: 0, end: 0, sources: [] };
const saved = (k) => { try { return sessionStorage.getItem(k) || ""; } catch (e) { return ""; } };
const keep = (k, v) => { try { sessionStorage.setItem(k, v); } catch (e) {} };
for (const id of ["key", "mode", "output", "silence", "instructions", "tools"])
  if (saved(id)) $(id).value = saved(id);
$("duplex").checked = saved("duplex") === "1";

function status(text) { $("status").textContent = text; }
function add(cls, text) {
  const el = document.createElement("div");
  el.className = "msg " + cls; el.textContent = text;
  $("log").appendChild(el); el.scrollIntoView({ block: "end" });
  return el;
}
function send(event) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(event)); }

function b64(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000)
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
function unb64(text) {
  const s = atob(text), out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; ++i) out[i] = s.charCodeAt(i);
  return out;
}

// Capture: the worklet posts input blocks; they are resampled to 24 kHz by
// linear interpolation and sent as PCM16 every 100 ms.
const WORKLET = `class Tap extends AudioWorkletProcessor {
  process(inputs) { const c = inputs[0][0]; if (c) this.port.postMessage(c.slice(0)); return true; }
}
registerProcessor("tap", Tap);`;
let resampleT = 0, last = 0, out = [];
function capture(block, rate) {
  let peak = 0;
  for (const v of block) peak = Math.max(peak, Math.abs(v));
  $("meter").firstChild.style.width = Math.min(100, peak * 140) + "%";
  if (!streaming) return;
  // From loudspeakers the reply would reach the microphone: unless the
  // user wears headphones, the microphone waits until the reply has played.
  if (!$("duplex").checked && !holding && playCtx && playCtx.currentTime < play.end + 0.25)
    return;
  const step = rate / RATE;
  while (resampleT < block.length) {
    const i = Math.floor(resampleT), f = resampleT - i;
    const a = i === 0 ? last : block[i - 1], b = block[i];
    out.push(a + (b - a) * f);
    resampleT += step;
  }
  resampleT -= block.length;
  last = block[block.length - 1];
  if (out.length >= RATE / 10) flush();
}
function flush() {
  if (!out.length) return;
  const pcm = new Int16Array(out.length);
  for (let i = 0; i < out.length; ++i)
    pcm[i] = Math.max(-1, Math.min(1, out[i])) * 32767;
  out = [];
  send({ type: "input_audio_buffer.append", audio: b64(new Uint8Array(pcm.buffer)) });
}

// Playback: reply chunks are scheduled back to back; what was heard is the
// time since the reply started, up to what was scheduled.
function heardMs() {
  if (!play.item) return 0;
  const now = Math.min(playCtx.currentTime, play.end);
  return Math.max(0, Math.round((now - play.start) * 1000));
}
function playChunk(item, bytes) {
  const pcm = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.length >> 1);
  const buffer = playCtx.createBuffer(1, pcm.length, RATE);
  const ch = buffer.getChannelData(0);
  for (let i = 0; i < pcm.length; ++i) ch[i] = pcm[i] / 32768;
  const src = playCtx.createBufferSource();
  src.buffer = buffer; src.connect(playCtx.destination);
  const now = playCtx.currentTime;
  if (play.item !== item) {
    play = { item, start: now + 0.05, end: now + 0.05, sources: [] };
  }
  const at = Math.max(play.end, now);
  if (play.end < now) play.start += now - play.end;  // underrun: the gap is not heard audio
  src.start(at);
  play.end = at + buffer.duration;
  play.sources.push(src);
  src.onended = () => { play.sources = play.sources.filter((s) => s !== src); };
}
// Stops the reply where the user is and keeps only that much in the model.
function interrupt(cancel = true) {
  const playing = play.item && playCtx && playCtx.currentTime < play.end;
  if (cancel && reply && !reply.done) send({ type: "response.cancel" });
  if (!playing) return;
  const ms = heardMs();
  for (const s of play.sources) { try { s.stop(); } catch (e) {} }
  send({ type: "conversation.item.truncate", item_id: play.item, content_index: 0, audio_end_ms: ms });
  if (reply && reply.item === play.item) reply.el.classList.add("cut");
  play = { item: null, start: 0, end: 0, sources: [] };
}

)html"
         R"html(function session() {
  let tools = [];
  const text = $("tools").value.trim();
  if (text) {
    try { tools = JSON.parse(text); } catch (e) { add("note", "Tools are not valid JSON: " + e.message); }
  }
  const vad = $("mode").value === "vad";
  return { type: "session.update", session: {
    type: "realtime", instructions: $("instructions").value,
    output_modalities: [$("output").value], tools,
    audio: { input: { turn_detection: vad ? { type: "server_vad",
      silence_duration_ms: Number($("silence").value) || 600, prefix_padding_ms: 300 } : null } } } };
}

function handle(e) {
  if (location.hash === "#connect" && e.type === "response.done")
    console.log("reply " + e.response.status + ": " + (reply ? reply.text : ""));
  switch (e.type) {
  case "session.created": send(session()); break;
  case "session.updated":
    streaming = $("mode").value === "vad";
    status(streaming ? "Listening. Speak whenever you like." : "Ready. Hold the button to talk.");
    break;
  case "input_audio_buffer.speech_started":
    interrupt(false);  // the server already stopped the reply
    status("Hearing you…");
    break;
  case "input_audio_buffer.speech_stopped": status("Thinking…"); break;
  case "conversation.item.created":
    if (e.item.role === "user" && e.item.content && e.item.content[0].type === "input_audio")
      add("user", "🎤 (spoken)");
    break;
  case "response.created": reply = { item: null, el: null, text: "", done: false }; break;
  case "response.output_item.added":
    if (e.item.type === "message") { reply.item = e.item.id; reply.el = add("bot", ""); }
    break;
  case "response.output_audio_transcript.delta":
  case "response.output_text.delta":
    reply.text += e.delta; reply.el.textContent = reply.text; status("Speaking…");
    break;
  case "response.reasoning_text.delta":
    if (!reply.think) reply.think = add("note", "");
    reply.think.textContent += e.delta;
    break;
  case "response.output_audio.delta": playChunk(e.item_id, unb64(e.delta)); break;
  case "response.output_item.done":
    if (e.item.type === "function_call") {
      const el = add("bot call", "call " + e.item.name + " " + e.item.arguments + "\n");
      const input = document.createElement("input");
      input.placeholder = "Result (JSON or text), Enter to send";
      input.onkeydown = (k) => {
        if (k.key !== "Enter") return;
        send({ type: "conversation.item.create", item: { type: "function_call_output",
          call_id: e.item.call_id, output: input.value || "{}" } });
        send({ type: "response.create" });
        input.disabled = true;
      };
      el.appendChild(input);
    }
    break;
  case "response.done":
    if (reply) reply.done = true;
    if (e.response.status === "failed")
      add("note", "Reply failed: " + JSON.stringify(e.response.status_details));
    status(streaming ? "Listening." : "Ready. Hold the button to talk.");
    break;
  case "error": add("note", "Server: " + e.error.message); break;
  }
}

async function connect() {
  if (ws) { ws.close(); return; }
  for (const id of ["key", "mode", "output", "silence", "instructions", "tools"]) keep(id, $(id).value);
  keep("duplex", $("duplex").checked ? "1" : "");
  try {
    mic = await navigator.mediaDevices.getUserMedia({ audio: {
      channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  } catch (e) {
    status("No microphone: " + e.message + " (open this page on localhost or HTTPS)");
    return;
  }
  captureCtx = new AudioContext();
  playCtx = new AudioContext({ sampleRate: RATE });
  await captureCtx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" })));
  node = new AudioWorkletNode(captureCtx, "tap");
  node.port.onmessage = (m) => capture(m.data, captureCtx.sampleRate);
  captureCtx.createMediaStreamSource(mic).connect(node);
  const key = $("key").value.trim();
  const url = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/v1/realtime";
  ws = new WebSocket(url, key ? ["realtime", "openai-insecure-api-key." + key] : ["realtime"]);
  status("Connecting…");
  ws.onmessage = (m) => handle(JSON.parse(m.data));
  ws.onopen = () => {
    $("connect").textContent = "Disconnect";
    for (const id of ["talk", "text", "send"]) $(id).disabled = false;
    $("talk").textContent = $("mode").value === "vad" ? "Hands-free (click to interrupt)" : "Hold to talk";
  };
  ws.onclose = (c) => {
    status("Disconnected" + (c.reason ? ": " + c.reason : "") + ".");
    ws = null; streaming = false;
    $("connect").textContent = "Connect";
    for (const id of ["talk", "text", "send"]) $(id).disabled = true;
    if (mic) mic.getTracks().forEach((t) => t.stop());
    if (captureCtx) captureCtx.close();
    if (playCtx) playCtx.close();
  };
}

function press() {
  if (!ws || holding) return;
  if ($("mode").value === "vad") { interrupt(); return; }
  holding = true; interrupt();
  resampleT = 0; out = []; streaming = true;
  $("talk").classList.add("live"); $("talk").textContent = "Listening… release to send";
}
function release() {
  if (!holding) return;
  holding = false; streaming = false; flush();
  $("talk").classList.remove("live"); $("talk").textContent = "Hold to talk";
  send({ type: "input_audio_buffer.commit" });
  send({ type: "response.create" });
  status("Thinking…");
}
$("connect").onclick = connect;
$("talk").addEventListener("pointerdown", press);
$("talk").addEventListener("pointerup", release);
$("talk").addEventListener("pointerleave", release);
addEventListener("keydown", (k) => { if (k.code === "Space" && k.target.tagName !== "INPUT" &&
  k.target.tagName !== "TEXTAREA" && !k.repeat) { k.preventDefault(); press(); } });
addEventListener("keyup", (k) => { if (k.code === "Space" && holding) { k.preventDefault(); release(); } });
// /voice#connect connects on load (a bookmark, a kiosk or a headless test)
// and logs each finished reply to the console.
if (location.hash === "#connect") connect();
$("typed").onsubmit = (f) => {
  f.preventDefault();
  const text = $("text").value.trim();
  if (!text || !ws) return;
  interrupt();
  add("user", text);
  send({ type: "conversation.item.create", item: { type: "message", role: "user",
    content: [{ type: "input_text", text }] } });
  send({ type: "response.create" });
  $("text").value = "";
};
</script>
</body>
</html>
)html";
}

}  // namespace gufo::server
