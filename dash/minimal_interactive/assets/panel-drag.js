(function () {
  const STORAGE_KEY = "dashDockablePanels_v1";
  let dragState = null;

  function loadPositions() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (_) {
      return {};
    }
  }

  function savePositions(data) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data || {}));
    } catch (_) {
      // no-op
    }
  }

  function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
  }

  function isDragTarget(el) {
    if (!el) return false;
    if (el.closest(".dockable-panel-edge")) return true;
    if (
      el.closest(
        "input, textarea, button, select, a, [contenteditable='true'], .Select, .chip-sortable, .chip-item, .rc-slider, .main-plot, .mini-context-plot, .collapsible-summary"
      )
    ) {
      return false;
    }
    return !!el.closest(".dockable-panel");
  }

  function applyFloating(panel, left, top, width) {
    panel.classList.add("floating");
    if (typeof width === "number" && Number.isFinite(width) && width > 180) {
      panel.style.width = `${Math.round(width)}px`;
    }
    panel.style.left = `${Math.round(left)}px`;
    panel.style.top = `${Math.round(top)}px`;
  }

  function dockPanel(panel) {
    panel.classList.remove("floating");
    panel.classList.remove("dragging");
    panel.style.left = "";
    panel.style.top = "";
    panel.style.width = "";
  }

  function persistPanel(panel) {
    if (!panel.id) return;
    const all = loadPositions();
    if (!panel.classList.contains("floating")) {
      delete all[panel.id];
    } else {
      const rect = panel.getBoundingClientRect();
      all[panel.id] = {
        left: Math.round(rect.left),
        top: Math.round(rect.top),
        width: Math.round(rect.width),
      };
    }
    savePositions(all);
  }

  function restorePanel(panel) {
    if (!panel.id) return;
    const all = loadPositions();
    const pos = all[panel.id];
    if (!pos) return;
    const width = pos.width || panel.getBoundingClientRect().width;
    const maxLeft = Math.max(8, window.innerWidth - width - 8);
    const left = clamp(pos.left || 8, 8, maxLeft);
    const height = panel.getBoundingClientRect().height;
    const maxTop = Math.max(8, window.innerHeight - height - 8);
    const top = clamp(pos.top || 8, 8, maxTop);
    applyFloating(panel, left, top, width);
  }

  function onPointerMove(e) {
    if (!dragState) return;
    const { panel, offsetX, offsetY } = dragState;
    const rect = panel.getBoundingClientRect();
    const maxLeft = Math.max(8, window.innerWidth - rect.width - 8);
    const maxTop = Math.max(8, window.innerHeight - rect.height - 8);
    const left = clamp(e.clientX - offsetX, 8, maxLeft);
    const top = clamp(e.clientY - offsetY, 8, maxTop);
    panel.style.left = `${Math.round(left)}px`;
    panel.style.top = `${Math.round(top)}px`;
  }

  function onPointerUp() {
    if (!dragState) return;
    const panel = dragState.panel;
    panel.classList.remove("dragging");
    persistPanel(panel);
    dragState = null;
  }

  function bindPanel(panel) {
    if (!panel || panel.dataset.dockableBound === "1") return;
    panel.dataset.dockableBound = "1";
    panel.classList.add("dockable-panel-has-edges");

    if (!panel.querySelector(".dockable-panel-edge")) {
      ["top", "right", "bottom", "left"].forEach((side) => {
        const edge = document.createElement("div");
        edge.className = `dockable-panel-edge dockable-panel-edge-${side}`;
        panel.appendChild(edge);
      });
    }

    restorePanel(panel);

    panel.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      if (!isDragTarget(e.target)) return;

      const rect = panel.getBoundingClientRect();
      if (!panel.classList.contains("floating")) {
        applyFloating(panel, rect.left, rect.top, rect.width);
      }

      dragState = {
        panel,
        offsetX: e.clientX - panel.getBoundingClientRect().left,
        offsetY: e.clientY - panel.getBoundingClientRect().top,
      };
      panel.classList.add("dragging");
      try {
        panel.setPointerCapture(e.pointerId);
      } catch (_) {
        // no-op
      }
      e.preventDefault();
    });

    // Double-click header/title to re-dock to left shelf.
    panel.addEventListener("dblclick", (e) => {
      if (!isDragTarget(e.target)) return;
      const rect = panel.getBoundingClientRect();
      if (panel.classList.contains("floating")) {
        dockPanel(panel);
      } else {
        applyFloating(panel, rect.left, rect.top, rect.width);
      }
      persistPanel(panel);
      e.preventDefault();
    });
  }

  function initPanels() {
    document.querySelectorAll(".dockable-panel").forEach(bindPanel);
  }

  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
  window.addEventListener("resize", () => {
    document.querySelectorAll(".dockable-panel.floating").forEach((panel) => {
      const rect = panel.getBoundingClientRect();
      const maxLeft = Math.max(8, window.innerWidth - rect.width - 8);
      const maxTop = Math.max(8, window.innerHeight - rect.height - 8);
      panel.style.left = `${Math.round(clamp(rect.left, 8, maxLeft))}px`;
      panel.style.top = `${Math.round(clamp(rect.top, 8, maxTop))}px`;
      persistPanel(panel);
    });
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initPanels);
  } else {
    initPanels();
  }

  const observer = new MutationObserver(() => initPanels());
  observer.observe(document.body, { childList: true, subtree: true });
})();
