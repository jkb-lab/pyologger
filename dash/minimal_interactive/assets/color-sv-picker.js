function setDashInputValue(inputEl, value) {
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype,
    "value"
  )?.set;
  if (setter) {
    setter.call(inputEl, value);
  } else {
    inputEl.value = value;
  }
  inputEl.dispatchEvent(new Event("input", { bubbles: true }));
}

function bindSvPicker() {
  const picker = document.getElementById("color-sv-picker");
  const target = document.getElementById("color-sv-input");
  if (!picker || !target) return;
  if (picker.dataset.svBound === "1") return;
  picker.dataset.svBound = "1";

  let dragging = false;

  const updateFromEvent = (ev) => {
    const rect = picker.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const x = Math.max(0, Math.min(rect.width, ev.clientX - rect.left));
    const y = Math.max(0, Math.min(rect.height, ev.clientY - rect.top));
    const sat = Math.round((x / rect.width) * 100);
    const val = Math.round((1 - y / rect.height) * 100);
    setDashInputValue(target, `${sat},${val}`);
  };

  picker.addEventListener("pointerdown", (ev) => {
    dragging = true;
    picker.setPointerCapture?.(ev.pointerId);
    updateFromEvent(ev);
  });

  picker.addEventListener("pointermove", (ev) => {
    if (!dragging) return;
    updateFromEvent(ev);
  });

  const stopDrag = () => {
    dragging = false;
  };
  picker.addEventListener("pointerup", stopDrag);
  picker.addEventListener("pointercancel", stopDrag);
  picker.addEventListener("lostpointercapture", stopDrag);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bindSvPicker);
} else {
  bindSvPicker();
}

const svObserver = new MutationObserver(() => bindSvPicker());
svObserver.observe(document.body, { childList: true, subtree: true });
