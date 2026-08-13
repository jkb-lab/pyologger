/* mini-dash playback clock (rAF) + keyboard stepping. */
(function () {
  "use strict";

  window.MiniPlayback = {
    playing: false, rate: 1, t: null, lo: 0, hi: 0, af: null, last: 0,
    setBounds: function (lo, hi) { if (lo != null) this.lo = +lo; if (hi != null) this.hi = +hi; if (this.t == null) this.t = this.lo; this.t = Math.max(this.lo, Math.min(this.hi, this.t)); },
    setRate: function (r) { this.rate = +r || 1; },
    sync: function (t) { if (t != null) this.t = +t; },
    start: function () { if (this.playing) return; if (this.t == null) this.t = this.lo; this.playing = true; this.last = performance.now(); this.tick(); },
    stop: function () { this.playing = false; if (this.af) { cancelAnimationFrame(this.af); this.af = null; } },
    tick: function () {
      if (!this.playing) return;
      var now = performance.now();
      this.t += ((now - this.last) / 1000) * this.rate;
      this.last = now;
      if (this.t >= this.hi) {
        this.t = this.hi; this.playing = false;
        var b = document.getElementById("b-play");
        if (b && b.getAttribute("data-playing") === "1") b.click();
        return;
      }
      this.af = requestAnimationFrame(this.tick.bind(this));
    },
  };

  // Keyboard: arrows step ±0.1s, Shift+arrows step ±10s, space toggles play.
  function onKey(e) {
    var el = document.activeElement;
    if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable)) return;
    var hidden = document.getElementById("key");
    var send = function (code) { if (!hidden) return; var s = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set; s.call(hidden, code + ":" + Date.now()); hidden.dispatchEvent(new Event("input", { bubbles: true })); };
    if (e.key === "ArrowLeft") { e.preventDefault(); send(e.shiftKey ? "DL" : "L"); }
    else if (e.key === "ArrowRight") { e.preventDefault(); send(e.shiftKey ? "DR" : "R"); }
    else if (e.code === "Space" || e.key === " ") { e.preventDefault(); var b = document.getElementById("b-play"); if (b) b.click(); }
    else if (!e.ctrlKey && !e.metaKey && !e.altKey) {
      // Event-edit keys (window.MINI_EDIT_KEYS is served from EDIT_BINDINGS in
      // mini_dash.py, so adding a key there is all it takes to bind a new one).
      var editKeys = window.MINI_EDIT_KEYS || {};
      var k = String(e.key || "").toUpperCase();
      if (Object.prototype.hasOwnProperty.call(editKeys, k)) { e.preventDefault(); send("E" + k); }
    }
  }

  function init() { if (window._miniKeys) return; document.addEventListener("keydown", onKey, true); window._miniKeys = true; }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", function () { setTimeout(init, 300); });
  else setTimeout(init, 300);
  setTimeout(init, 1200);
})();

// --- slider tooltip formatting -------------------------------------------- //
// dcc.Slider tooltips render the raw value (a unix epoch here), which shows up
// as an opaque number like 1687545760. `tooltip={"template": "{value}"}` with a
// transform name routes the value through window.dccFunctions first.
window.dccFunctions = window.dccFunctions || {};
window.dccFunctions.epochToTime = function (value) {
    if (value == null || isNaN(value)) return "";
    try {
        return new Date(Number(value) * 1000).toLocaleTimeString("en-US", {
            hour12: false, timeZone: window.MINI_TZ || undefined,
        });
    } catch (e) {
        return String(value);
    }
};
window.dccFunctions.epochToDateTime = function (value) {
    if (value == null || isNaN(value)) return "";
    try {
        const d = new Date(Number(value) * 1000);
        const opts = { hour12: false, timeZone: window.MINI_TZ || undefined };
        return d.toLocaleDateString("en-CA", opts) + " " + d.toLocaleTimeString("en-US", opts);
    } catch (e) {
        return String(value);
    }
};

// --- heartbeat audio -------------------------------------------------------- //
// Plays a beat sound as the playhead crosses each heartbeat event. The Python
// side pushes the beat times (window.MiniHeart.setBeats) and this watches the
// playhead, so it works with either clock (the video's timeupdate or the rAF
// fallback) without either needing to know about audio.
//
// Uses WebAudio with a single decoded buffer: an <audio> element per beat can't
// keep up at real heart rates, and re-decoding would stutter.
window.MiniHeart = (function () {
  var ctx = null, buf = null, loading = false;
  var beats = [];        // sorted epoch seconds
  var lastT = null;      // previous playhead position
  var enabled = false;

  function ensure() {
    if (ctx || loading) return;
    loading = true;
    try {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      fetch("/assets/04_fast_heart_badum_360ms.wav")
        .then(function (r) { return r.arrayBuffer(); })
        .then(function (a) { return ctx.decodeAudioData(a); })
        .then(function (b) { buf = b; })
        .catch(function (e) { console.warn("[mini-dash] heartbeat audio unavailable", e); });
    } catch (e) {
      console.warn("[mini-dash] WebAudio unavailable", e);
    }
  }

  function fire() {
    if (!ctx || !buf) return;
    if (ctx.state === "suspended") ctx.resume();
    var src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    try { src.start(0); } catch (e) {}
  }

  return {
    setBeats: function (arr) { beats = (arr || []).slice().sort(function (a, b) { return a - b; }); lastT = null; },
    setEnabled: function (on) { enabled = !!on; if (on) ensure(); if (!on) lastT = null; },
    reset: function () { lastT = null; },
    // Called on every playhead change. Only fires when moving forward by a
    // small step (i.e. real playback) — scrubbing or jumping is ignored so a
    // seek doesn't machine-gun every beat it skipped over.
    tick: function (t, playing, rate) {
      if (!enabled || t == null) return;
      if (!playing || Math.abs((rate == null ? 1 : rate) - 1) > 0.01) { lastT = t; return; }
      if (lastT == null) { lastT = t; return; }
      var dt = t - lastT;
      if (dt <= 0 || dt > 1.0) { lastT = t; return; }   // paused, rewound, or jumped
      for (var i = 0; i < beats.length; i++) {
        if (beats[i] > lastT && beats[i] <= t) { fire(); break; }  // one per tick
      }
      lastT = t;
    },
  };
})();
