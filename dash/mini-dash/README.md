# mini-dash

A stripped-down, single-deployment interactive viewer. Built from scratch to be
small and legible (~1 file), reusing only the proven data + Immich-video pieces
from the larger `integrated` app.

## Launch

From `pyologger/`:

```bash
python dash/mini-dash/mini_dash.py                              # default deployment
python dash/mini-dash/mini_dash.py --dataset <id> --deployment <id> --port 8070
```

Open http://127.0.0.1:8070.

## Features (only these, on purpose)

- **One unified timeline.** A single `playhead` store drives the plot's yellow
  line, the synchronized video, and the coverage strip. Move it by dragging the
  **Playhead** slider, clicking the plot, the transport buttons, or the keyboard.
- **Signal plot** with one row per signal.
  - **Edit colors** — the color swatch on each signal chip.
  - **Reorder** — the ↑ / ↓ buttons on each chip (top chip = top row).
  - **Add signals** — the "add a signal…" dropdown.
- **Synchronized video** from the Immich album `DepID_<deployment>`, streamed
  through a server-side proxy (`/mini-video/<asset_id>`) so the API key stays on
  the server and the browser can still seek (HTTP Range). Auto-switches clips as
  the playhead crosses clip boundaries.
- **Transport**: play / pause, and step by **±0.1 s** or **±10 s**.
  - Keyboard: `←`/`→` = ±0.1 s, `Shift+←`/`Shift+→` = ±10 s, `Space` = play/pause.
  - Playback rate: 0.5× / 1× / 2× / 5×.

## Key config parameters

Edit the constants at the top of `mini_dash.py`:

| Param | Meaning |
|-------|---------|
| `DEFAULT_DATASET` / `DEFAULT_DEPLOYMENT` | opened when no CLI args are given |
| `WINDOW_MINUTES` | initial view-window width |
| `TARGET_HZ` | plot downsample target (snappiness vs. detail) |
| `STEP_SMALL` / `STEP_LARGE` | the ±0.1 s / ±10 s step sizes |
| `PLAYBACK_RATES` | rate-selector options |
| `_PALETTE` | default per-signal colors |

## How playback stays smooth

- The **video is the master clock** while playing: a `timeupdate` listener drives
  the `playhead` store, so the plot follows the real frames (no seeking the
  element mid-play, which causes black frames).
- When the playhead is in a gap (no clip), a small rAF clock
  (`assets/playback.js`, `window.MiniPlayback`) advances it instead.
- The **yellow line moves clientside** (`Plotly.relayout`) on every playhead
  change, so drag/play stay smooth without a server figure round-trip.
- The window **follows the playhead** when it scrubs past the edge.

## Requirements

- `IMMICH_API_KEY` / `IMMICH_BASE_URL` in the environment (auto-loaded from
  `EcoPhysVideoViz/.env` if unset). Without Immich, the plot still works; the
  video panel just shows "No video clips for this deployment."
- The app runs `threaded=True` — required so the streaming video proxy doesn't
  block the page's own requests.
