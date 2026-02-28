(function () {
  let dragState = null;

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function onMouseMove(e) {
    if (!dragState) return;
    const dx = e.clientX - dragState.startX;
    const dy = e.clientY - dragState.startY;

    const cardRect = dragState.cardRect;
    const maxLeft = Math.max(0, window.innerWidth - cardRect.width - 8);
    const maxTop = Math.max(0, window.innerHeight - cardRect.height - 8);

    const left = clamp(dragState.startLeft + dx, 0, maxLeft);
    const top = clamp(dragState.startTop + dy, 0, maxTop);

    dragState.modal.style.left = `${Math.round(left)}px`;
    dragState.modal.style.top = `${Math.round(top)}px`;
  }

  function endDrag() {
    if (!dragState) return;
    dragState.modal.classList.remove("dragging");
    dragState = null;
    document.removeEventListener("mousemove", onMouseMove);
    document.removeEventListener("mouseup", endDrag);
  }

  document.addEventListener("mousedown", (e) => {
    const edge = e.target.closest(".color-editor-edge");
    const moveHandle = e.target.closest(".color-editor-move-handle");
    if (!edge && !moveHandle) return;

    const origin = edge || moveHandle;
    const modal = origin.closest("#color-editor-modal");
    if (!modal || modal.classList.contains("hidden")) return;

    const card = modal.querySelector(".color-editor-card");
    if (!card) return;

    e.preventDefault();

    const modalRect = modal.getBoundingClientRect();
    const cardRect = card.getBoundingClientRect();

    dragState = {
      modal,
      cardRect,
      startX: e.clientX,
      startY: e.clientY,
      startLeft: modalRect.left,
      startTop: modalRect.top,
    };

    modal.classList.add("dragging");
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", endDrag);
  });
})();
