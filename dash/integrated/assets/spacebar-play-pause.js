/*
 * Spacebar toggles the play/pause button, unless typing in a field.
 */
(function () {
  "use strict";

  function onKey(e) {
    if (e.code !== "Space" && e.key !== " ") return;
    const el = document.activeElement;
    if (
      el &&
      (el.tagName === "INPUT" ||
        el.tagName === "TEXTAREA" ||
        el.tagName === "SELECT" ||
        el.isContentEditable)
    ) {
      return;
    }
    const btn = document.getElementById("play-pause-btn");
    if (!btn) return;
    e.preventDefault();
    e.stopPropagation();
    btn.click();
  }

  function init() {
    if (window._integratedSpacebarSetup) return;
    document.addEventListener("keydown", onKey, true);
    window._integratedSpacebarSetup = true;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => setTimeout(init, 300));
  } else {
    setTimeout(init, 300);
  }
  setTimeout(init, 1200);
})();
