let activeChip = null;
let dragArmedChip = null;
let activeList = null;

function isInteractiveTarget(node) {
  if (!node || !node.closest) return false;
  return Boolean(
    node.closest(
      "button, a, input, select, textarea, summary, .rc-slider, .chip-remove-btn, .chip-color-edit, .chip-color-dot, .legend-signal-toggle, .signal-collapse-toggle, .collapsible-summary"
    )
  );
}

function getDirectSortableItem(root, node) {
  let el = node && node.closest ? node.closest(".chip-item, .dnd-item") : null;
  while (el && el.parentElement !== root) {
    el = el.parentElement && el.parentElement.closest
      ? el.parentElement.closest(".chip-item, .dnd-item")
      : null;
  }
  return el;
}

function setChipDraggable(root) {
  const chips = Array.from(root.children).filter((el) =>
    el.classList && (el.classList.contains("chip-item") || el.classList.contains("dnd-item"))
  );
  chips.forEach((chip) => {
    chip.setAttribute("draggable", "true");
  });
}

function updateOrderPayloadFromList(listEl) {
  const hidden = document.getElementById("chip-order-updates");
  if (!hidden) return;

  const group = listEl.getAttribute("data-order-group");
  const key = listEl.getAttribute("data-order-key");
  if (!group || !key) return;

  const values = Array.from(listEl.children)
    .filter((el) => el.classList && (el.classList.contains("chip-item") || el.classList.contains("dnd-item")))
    .map((el) =>
    el.getAttribute("data-value")
    );

  let payload = {};
  try {
    payload = JSON.parse(hidden.value || "{}");
  } catch (_) {
    payload = {};
  }
  if (!payload[group]) payload[group] = {};
  payload[group][key] = values;

  const nextValue = JSON.stringify(payload);
  // React-controlled inputs require using the native setter.
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype,
    "value"
  )?.set;
  if (setter) {
    setter.call(hidden, nextValue);
  } else {
    hidden.value = nextValue;
  }
  hidden.dispatchEvent(new Event("input", { bubbles: true }));
}

function attachDnDHandlers(root) {
  if (root.dataset.dndBound === "1") return;
  root.dataset.dndBound = "1";

  root.addEventListener("mousedown", (e) => {
    const editBtn = e.target.closest(".chip-color-edit");
    if (!editBtn) return;
    const anchor = document.getElementById("color-editor-anchor-input");
    if (!anchor) return;
    const rect = editBtn.getBoundingClientRect();
    const payload = JSON.stringify({
      x: Math.round(rect.right + window.scrollX),
      y: Math.round(rect.top + window.scrollY),
    });
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype,
      "value"
    )?.set;
    if (setter) {
      setter.call(anchor, payload);
    } else {
      anchor.value = payload;
    }
    anchor.dispatchEvent(new Event("input", { bubbles: true }));
  });

  root.addEventListener("pointerdown", (e) => {
    const handle = e.target.closest(".chip-handle");
    const signalSurface = e.target.closest(
      ".signal-drag-surface, .signal-block-edge, .legend-signal-edge"
    );
    const directChip = getDirectSortableItem(root, e.target);
    const isPeakParamRoot = (root.getAttribute("data-order-group") || "") === "peak_params";
    if (handle || signalSurface) {
      // For peak parameter cards, only handle-initiated drag is allowed.
      // Prevent summary toggle when using the drag handle.
      if (isPeakParamRoot && handle) {
        e.preventDefault();
        e.stopPropagation();
      }
      dragArmedChip = (handle || signalSurface).closest(".chip-item, .dnd-item");
      return;
    }
    if (isPeakParamRoot) {
      dragArmedChip = null;
      return;
    }
    if (directChip && !isInteractiveTarget(e.target)) {
      dragArmedChip = directChip;
      return;
    }
    dragArmedChip = null;
  });

  root.addEventListener("contextmenu", (e) => {
    const chip = e.target.closest(".chip-item");
    if (!chip) return;
    if (chip.getAttribute("data-color-editable") !== "1") return;
    const editBtn = chip.querySelector(".chip-color-edit");
    if (!editBtn) return;
    e.preventDefault();
    editBtn.click();
  });

  root.addEventListener("pointerup", () => {
    dragArmedChip = null;
  });

  root.addEventListener("pointercancel", () => {
    dragArmedChip = null;
  });

  root.addEventListener("dragstart", (e) => {
    const chip = getDirectSortableItem(root, e.target);
    if (!chip) return;
    // Browsers may emit dragstart with target=chip even when drag began on handle.
    if (
      dragArmedChip !== chip &&
      !e.target.closest(
        ".chip-handle, .signal-drag-surface, .signal-block-edge, .legend-signal-edge"
      )
    ) {
      e.preventDefault();
      return;
    }
    activeChip = chip;
    activeList = chip.closest("[data-order-group][data-order-key]");
    chip.classList.add("dragging");
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", chip.getAttribute("data-value") || "");
  });

  root.addEventListener("dragover", (e) => {
    if (!activeChip || !activeList || activeList !== root) return;
    e.preventDefault();
    const target = getDirectSortableItem(root, e.target);
    if (!activeChip || !target || target === activeChip) return;

    const rect = target.getBoundingClientRect();
    const axis = (root.getAttribute("data-dnd-axis") || "x").toLowerCase();
    const after = axis === "y"
      ? e.clientY > rect.top + rect.height / 2
      : e.clientX > rect.left + rect.width / 2;
    if (after) {
      target.after(activeChip);
    } else {
      target.before(activeChip);
    }
  });

  root.addEventListener("drop", (e) => {
    if (!activeChip || !activeList || activeList !== root) return;
    e.preventDefault();
    updateOrderPayloadFromList(root);
  });

  root.addEventListener("dragend", () => {
    if (activeChip) {
      activeChip.classList.remove("dragging");
    }
    activeChip = null;
    dragArmedChip = null;
    activeList = null;
  });
}

function initChipLists() {
  const lists = document.querySelectorAll(".chip-sortable, .signal-sortable");
  lists.forEach((list) => {
    setChipDraggable(list);
    attachDnDHandlers(list);
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initChipLists);
} else {
  initChipLists();
}

const chipObserver = new MutationObserver(() => {
  initChipLists();
});

chipObserver.observe(document.body, { childList: true, subtree: true });
