/*
 * Arrow key nudging for playhead control.
 * Left/Right arrows nudge playhead by -/+5.0s via hidden Dash input.
 */
(function () {
  "use strict";

  function init() {
    if (window._minimalArrowNavSetup) return;
    document.addEventListener("keydown", handleArrowKey, true);
    window._minimalArrowNavSetup = true;
  }

  function handleArrowKey(e) {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;

    const activeEl = document.activeElement;
    if (
      activeEl &&
      activeEl.id !== "arrow-key-input" &&
      (activeEl.tagName === "INPUT" ||
        activeEl.tagName === "TEXTAREA" ||
        activeEl.tagName === "SELECT" ||
        activeEl.isContentEditable)
    ) {
      return;
    }

    e.preventDefault();
    e.stopPropagation();
    e.stopImmediatePropagation();

    const direction = e.key === "ArrowRight" ? 1 : -1;
    updateArrowKeyInput(direction);
  }

  function updateArrowKeyInput(direction) {
    const hiddenInput = document.getElementById("arrow-key-input");
    if (!hiddenInput) return;

    const value = direction + ":" + Date.now();
    const nativeSetter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype,
      "value"
    ).set;
    nativeSetter.call(hiddenInput, value);
    hiddenInput.dispatchEvent(new Event("input", { bubbles: true }));
    hiddenInput.dispatchEvent(new Event("change", { bubbles: true }));
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      setTimeout(init, 300);
    });
  } else {
    setTimeout(init, 300);
  }

  setTimeout(init, 1200);
})();
