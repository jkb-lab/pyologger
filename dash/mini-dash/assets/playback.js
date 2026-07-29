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
  }

  function init() { if (window._miniKeys) return; document.addEventListener("keydown", onKey, true); window._miniKeys = true; }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", function () { setTimeout(init, 300); });
  else setTimeout(init, 300);
  setTimeout(init, 1200);
})();
