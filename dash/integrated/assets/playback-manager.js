/*
 * Playback clock for the integrated dash app.
 *
 * A single requestAnimationFrame loop advances a continuous "current time"
 * (epoch seconds) between bounds at a chosen playback rate. A dcc.Interval
 * polls `currentTime` into the `playhead-time` store, which cascades to the
 * plot playhead, sliders, and video. Modeled on EcoPhysVideoViz's
 * window.DiveDBPlayback manager.
 */
(function () {
  "use strict";

  window.IntegratedPlayback = {
    isPlaying: false,
    playbackRate: 1,
    currentTime: null,
    minTime: 0,
    maxTime: 0,
    animationId: null,
    lastTickTime: null,
    lastUpdateTime: 0,
    TICK_INTERVAL_MS: 33, // ~30fps store writes

    setBounds: function (minTime, maxTime) {
      if (minTime != null) this.minTime = Number(minTime);
      if (maxTime != null) this.maxTime = Number(maxTime);
      if (this.currentTime == null) this.currentTime = this.minTime;
      // Keep the clock inside the (possibly new) bounds.
      this.currentTime = Math.max(this.minTime, Math.min(this.maxTime, this.currentTime));
    },

    setPlaybackRate: function (rate) {
      this.playbackRate = Number(rate) || 1;
    },

    // Called when the user seeks externally (slider drag, plot click, arrow keys).
    syncTime: function (time) {
      if (time == null) return;
      this.currentTime = Number(time);
    },

    start: function () {
      if (this.isPlaying) return;
      if (this.currentTime == null) this.currentTime = this.minTime;
      this.isPlaying = true;
      this.lastTickTime = performance.now();
      this.lastUpdateTime = 0;
      this.tick();
    },

    stop: function () {
      this.isPlaying = false;
      if (this.animationId) {
        cancelAnimationFrame(this.animationId);
        this.animationId = null;
      }
    },

    tick: function () {
      if (!this.isPlaying) return;
      const now = performance.now();
      const deltaMs = now - this.lastTickTime;
      this.lastTickTime = now;
      const deltaSeconds = (deltaMs / 1000) * this.playbackRate;
      let newTime = (this.currentTime || this.minTime) + deltaSeconds;
      if (newTime >= this.maxTime) {
        // Stop at the end of the window rather than looping.
        newTime = this.maxTime;
        this.currentTime = newTime;
        this.isPlaying = false;
        // Reflect the stop in the play button so Dash state stays in sync.
        const btn = document.getElementById("play-pause-btn");
        if (btn && btn.getAttribute("data-playing") === "1") {
          btn.click();
        }
        return;
      }
      this.currentTime = newTime;
      this.animationId = requestAnimationFrame(() => this.tick());
    },
  };
})();
