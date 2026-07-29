(function () {
  const PANELS = [
    {
      cardId: "location-map-card",
      handleId: "location-map-drag-handle",
      storageKey: "popout_location_map_v2",
      minW: 520,
      minH: 360,
      defaultW: 760,
      defaultH: 500,
      defaultLeft: 24,
      defaultTop: 70,
    },
    {
      cardId: "model-3d-card",
      handleId: "model-3d-drag-handle",
      storageKey: "popout_model_3d_v2",
      minW: 560,
      minH: 460,
      defaultW: 760,
      defaultH: 560,
      defaultLeft: 24,
      defaultTop: 70,
    },
    {
      cardId: "model-3d-depth-card",
      handleId: "model-3d-depth-drag-handle",
      storageKey: "popout_model_3d_depth_v1",
      minW: 420,
      minH: 340,
      defaultW: 560,
      defaultH: 460,
      defaultLeft: 64,
      defaultTop: 96,
    },
    {
      cardId: "model-3d-depth-track-card",
      handleId: "model-3d-depth-track-drag-handle",
      storageKey: "popout_model_3d_depth_track_v1",
      minW: 420,
      minH: 340,
      defaultW: 560,
      defaultH: 460,
      defaultLeft: 84,
      defaultTop: 116,
    },
    {
      cardId: "seg-workflow-diagram-card",
      handleId: "seg-workflow-diagram-drag-handle",
      toggleBtnId: "seg-workflow-diagram-popout-btn",
      storageKey: "popout_seg_workflow_diagram_v1",
      minW: 360,
      minH: 360,
      defaultW: 500,
      defaultH: 680,
      defaultLeft: 140,
      defaultTop: 96,
    },
  ];

  let drag = null;
  const resizeObsByCard = new Map();
  const wasOpenByCard = new Map();

  function resizeLocationMapPlot() {
    const mapEl = document.getElementById("location-map");
    if (!mapEl || !window.Plotly || !window.Plotly.Plots) return;
    // Run a couple of frames later so CSS/layout changes have settled.
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        try {
          window.Plotly.Plots.resize(mapEl);
        } catch (_) {
          // no-op
        }
      });
    });
  }

  function recenterThreeJsView(card) {
    if (!card) return;
    const clickReset = () => {
      const resetBtn = card.querySelector(".btn-resetview");
      if (resetBtn && typeof resetBtn.click === "function") {
        resetBtn.click();
      }
    };
    requestAnimationFrame(() => {
      clickReset();
      requestAnimationFrame(() => clickReset());
    });
  }

  function clamp(v, lo, hi) {
    if (!Number.isFinite(v)) v = lo;
    if (hi < lo) return hi;
    return Math.max(lo, Math.min(hi, v));
  }

  function loadState(storageKey) {
    try {
      const raw = localStorage.getItem(storageKey);
      return raw ? JSON.parse(raw) : null;
    } catch (_) {
      return null;
    }
  }

  function saveState(cfg, card) {
    if (cfg.cardId === "location-map-card") return;
    try {
      const rect = card.getBoundingClientRect();
      localStorage.setItem(
        cfg.storageKey,
        JSON.stringify({
          left: Math.round(rect.left),
          top: Math.round(rect.top),
          width: Math.round(rect.width),
          height: Math.round(rect.height),
        })
      );
    } catch (_) {
      // no-op
    }
  }

  function isOpen(card) {
    return !!card && card.classList.contains("pip-open");
  }

  function clearInlineGeometry(card) {
    if (!card) return;
    card.style.width = "";
    card.style.height = "";
    card.style.left = "";
    card.style.top = "";
    card.style.right = "";
  }

  function applyState(cfg, card) {
    if (!card || !isOpen(card)) return;
    if (cfg.cardId === "location-map-card") {
      const width = clamp(cfg.defaultW, cfg.minW, Math.floor(window.innerWidth * 0.92));
      const height = clamp(cfg.defaultH, cfg.minH, Math.floor(window.innerHeight * 0.88));
      const maxLeft = Math.max(8, window.innerWidth - width - 8);
      const maxTop = Math.max(8, window.innerHeight - height - 8);
      const left = clamp(cfg.defaultLeft, 8, maxLeft);
      const top = clamp(cfg.defaultTop, 8, maxTop);
      card.style.width = `${Math.round(width)}px`;
      card.style.height = `${Math.round(height)}px`;
      card.style.left = `${Math.round(left)}px`;
      card.style.top = `${Math.round(top)}px`;
      card.style.right = "auto";
      return;
    }
    const state = loadState(cfg.storageKey) || {};
    const maxW = Math.floor(window.innerWidth * 0.92);
    const maxH = Math.floor(window.innerHeight * 0.88);
    const width = clamp(Number(state.width) || cfg.defaultW, cfg.minW, maxW);
    const height = clamp(Number(state.height) || cfg.defaultH, cfg.minH, maxH);
    const maxLeft = Math.max(8, window.innerWidth - width - 8);
    const maxTop = Math.max(8, window.innerHeight - height - 8);
    const left = clamp(Number(state.left) || cfg.defaultLeft, 8, maxLeft);
    const top = clamp(Number(state.top) || cfg.defaultTop, 8, maxTop);

    card.style.width = `${Math.round(width)}px`;
    card.style.height = `${Math.round(height)}px`;
    card.style.left = `${Math.round(left)}px`;
    card.style.top = `${Math.round(top)}px`;
    card.style.right = "auto";
  }

  function findCfgByHandle(target) {
    for (const cfg of PANELS) {
      if (target && target.closest(`#${cfg.handleId}`)) return cfg;
    }
    return null;
  }

  function onPointerDown(e) {
    if (e.button !== 0) return;
    const cfg = findCfgByHandle(e.target);
    if (!cfg) return;
    const card = document.getElementById(cfg.cardId);
    if (!isOpen(card)) return;

    const rect = card.getBoundingClientRect();
    card.style.left = `${Math.round(rect.left)}px`;
    card.style.top = `${Math.round(rect.top)}px`;
    card.style.right = "auto";
    drag = {
      cfg,
      card,
      dx: e.clientX - rect.left,
      dy: e.clientY - rect.top,
    };
    card.classList.add("dragging");
    document.body.style.userSelect = "none";
    e.stopPropagation();
    e.preventDefault();
  }

  function onPointerMove(e) {
    if (!drag) return;
    const { card, dx, dy } = drag;
    const rect = card.getBoundingClientRect();
    const maxLeft = Math.max(8, window.innerWidth - rect.width - 8);
    const maxTop = Math.max(8, window.innerHeight - rect.height - 8);
    const left = clamp(e.clientX - dx, 8, maxLeft);
    const top = clamp(e.clientY - dy, 8, maxTop);
    card.style.left = `${Math.round(left)}px`;
    card.style.top = `${Math.round(top)}px`;
  }

  function onPointerUp() {
    if (!drag) return;
    drag.card.classList.remove("dragging");
    saveState(drag.cfg, drag.card);
    drag = null;
    document.body.style.userSelect = "";
  }

  function onSummaryClickCapture(e) {
    // When popped out, don't let one-clicks on the summary/header collapse the panel.
    for (const cfg of PANELS) {
      const card = document.getElementById(cfg.cardId);
      if (!isOpen(card)) continue;
      const summary = e.target && e.target.closest(`#${cfg.cardId} > details > summary`);
      if (!summary) continue;
      // Keep explicit controls working.
      if (e.target.closest("button, input, select, textarea, a")) return;
      e.preventDefault();
      e.stopPropagation();
      return;
    }
  }

  function onDoubleClickCapture(e) {
    for (const cfg of PANELS) {
      const card = document.getElementById(cfg.cardId);
      if (!isOpen(card) || !cfg.toggleBtnId) continue;
      const titleRow = e.target && e.target.closest(`#${cfg.cardId} .card-title-row`);
      if (!titleRow) continue;
      if (e.target.closest("button, input, select, textarea, a")) return;
      const toggleBtn = document.getElementById(cfg.toggleBtnId);
      if (toggleBtn && typeof toggleBtn.click === "function") {
        e.preventDefault();
        e.stopPropagation();
        toggleBtn.click();
      }
      return;
    }
  }

  function bindPanel(cfg) {
    const card = document.getElementById(cfg.cardId);
    if (!card) return;
    // Keep resize persistence for 3D panel, but disable map auto-resize observer
    // to prevent feedback growth loop on map redraws.
    if (
      (cfg.cardId === "model-3d-card" ||
        cfg.cardId === "model-3d-depth-card" ||
        cfg.cardId === "model-3d-depth-track-card") &&
      !resizeObsByCard.has(cfg.cardId)
    ) {
      const ro = new ResizeObserver(() => {
        if (isOpen(card)) {
          saveState(cfg, card);
        }
      });
      ro.observe(card);
      resizeObsByCard.set(cfg.cardId, ro);
    }
    const openNow = isOpen(card);
    const wasOpen = !!wasOpenByCard.get(cfg.cardId);
    if (openNow && !wasOpen) {
      applyState(cfg, card);
      if (cfg.cardId === "location-map-card") resizeLocationMapPlot();
      if (
        cfg.cardId === "model-3d-card" ||
        cfg.cardId === "model-3d-depth-card" ||
        cfg.cardId === "model-3d-depth-track-card"
      ) {
        recenterThreeJsView(card);
      }
    }
    if (!openNow) {
      // On docking, drop stale popout geometry so CSS-controlled docked layout can take over.
      clearInlineGeometry(card);
    }
    if (!openNow && wasOpen && cfg.cardId === "location-map-card") resizeLocationMapPlot();
    wasOpenByCard.set(cfg.cardId, openNow);
  }

  function bindAll() {
    PANELS.forEach(bindPanel);
  }

  window.addEventListener("pointerdown", onPointerDown);
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
  document.addEventListener("click", onSummaryClickCapture, true);
  document.addEventListener("dblclick", onDoubleClickCapture, true);
  window.addEventListener("resize", () => {
    PANELS.forEach((cfg) => {
      const card = document.getElementById(cfg.cardId);
      if (!isOpen(card)) return;
      applyState(cfg, card);
      saveState(cfg, card);
    });
    resizeLocationMapPlot();
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindAll);
  } else {
    bindAll();
  }

  const observer = new MutationObserver(() => bindAll());
  observer.observe(document.body, { childList: true, subtree: true, attributes: true });
})();
