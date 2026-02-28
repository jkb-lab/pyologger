(function () {
  function toFormattedParts(raw) {
    const num = Number(raw);
    if (!Number.isFinite(num)) return null;
    const dateObject = new Date(num * 1000);
    if (Number.isNaN(dateObject.getTime())) return null;

    const year = dateObject.getUTCFullYear();
    const month = String(dateObject.getUTCMonth() + 1).padStart(2, "0");
    const day = String(dateObject.getUTCDate()).padStart(2, "0");
    const hours = String(dateObject.getUTCHours()).padStart(2, "0");
    const minutes = String(dateObject.getUTCMinutes()).padStart(2, "0");
    const seconds = String(dateObject.getUTCSeconds()).padStart(2, "0");
    return {
      time: `${hours}:${minutes}:${seconds}.000`,
      date: `${year}-${month}-${day}`,
    };
  }

  function styleSliderTooltip(root) {
    if (!root) return;
    const raw = (root.textContent || "").trim();
    if (!raw) return;
    const parts = toFormattedParts(raw);
    if (!parts) return;

    root.innerHTML =
      `<div class="slider-tooltip-time">${parts.time}</div>` +
      `<div class="slider-tooltip-date">${parts.date}</div>`;
    root.dataset.styledTooltip = "1";
  }

  function scanTooltips() {
    document
      .querySelectorAll(".rc-slider-tooltip-inner")
      .forEach((el) => styleSliderTooltip(el));
  }

  const observer = new MutationObserver(scanTooltips);
  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
    characterData: true,
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scanTooltips);
  } else {
    scanTooltips();
  }
})();
