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

- **Data source banner** under the header, stating explicitly where what you're
  looking at came from: the backing store for signals + events (**NetCDF** or
  **data.pkl**, whichever `DeploymentDataSource` resolved) with the filename,
  the video route (**Immich** proxied vs. **local files**) with the clip count,
  and the deployment folder. When a `data.pkl` exists alongside a NetCDF it says
  so, since **Write** updates both. Updates on deployment switch.

- **Dataset / deployment dropdowns** in the header. Switching reloads the
  deployment in-process — no restart — and resets all deployment-scoped state
  (signals, colors, events, window, playhead, video clips, depth context, GUI
  notes path). Pending unwritten edits are **discarded** on switch, since they
  reference the previous deployment's timeline. The deployment list only shows
  folders with an `outputs/` containing a `.nc` or `.pkl`, so unprocessed
  datasets appear empty rather than erroring. `--dataset` / `--deployment` still
  set the initial selection.

- **One unified timeline.** A single `playhead` store drives the plot's yellow
  line, the synchronized video, and the coverage strip. Move it by dragging the
  **Playhead** slider, clicking the plot, the transport buttons, or the keyboard.
- **Signal plot** with one row per signal.
  - **Edit colors** — the color swatch on each signal chip.
  - **Reorder** — the ↑ / ↓ buttons on each chip (top chip = top row).
  - **Add signals** — the "add a signal…" dropdown.
- **Event → Channel Mapper** (Cytoscape, below the plot). Events on the left,
  signals on the right, one edge per mapping. Click an event node to select it,
  then click signal nodes to add/remove links; click an edge to remove it. Every
  change saves to `__event_targets__` in `color_mappings.json` immediately and
  the plot redraws. Needs `dash-cytoscape` (falls back to a hint if absent).
- **Event overlays**, matching `plot_tag_data_interactive`. Each event key is
  drawn on its target signal row. The sidebar lists every event as
  `[checkbox] [color swatch] [name · count · targets]` — the checkbox toggles
  visibility, the swatch edits the event's color (saved to `color_mappings.json`
  as a top-level entry, which takes precedence over `__event_styles__`), and
  long keys wrap rather than clip.
  - **Point events** — exhalation breaths, accepted auto-detected heartbeats
    and strokebeats, manual heartbeats — as markers near the top of their row.
  - **State events** — dives, ECG QC spans — as shaded spans across the row.
    Dives additionally **fill the depth trace** down to the surface.
  - Colors, symbols and target signals come from `color_mappings.json`
    (`__event_styles__` / `__event_targets__`), so mini-dash and the Streamlit
    plots stay consistent. Edit them there, not here.
  - Point vs. span is decided by `type == "state"` **and** a non-zero
    `duration`. Both fields are unreliable alone: accepted heartbeats are tagged
    `state` with zero duration, and accepted strokebeats are tagged `point` but
    carry the stroke interval as a duration.
- **Synchronized video** from the Immich album `DepID_<deployment>`, streamed
  through a server-side proxy (`/mini-video/<asset_id>`) so the API key stays on
  the server and the browser can still seek (HTTP Range). Auto-switches clips as
  the playhead crosses clip boundaries.
- **Timeline stack**, grouped directly under the video: transport, video
  coverage strip, **Playhead** slider, then **Window** slider beside a
  whole-deployment **depth context plot**. The context plot shades the active
  window and marks the playhead; click it to recenter the window there (width
  preserved). The signal plot sits below the stack.
  - Slider tooltips show real timestamps, not raw epochs — the playhead shows
    time-of-day, the window shows date + time, both in the deployment's
    timezone (`window.MINI_TZ`, via `dccFunctions` in `assets/playback.js`).
- **Event editing** (keyboard). Click an event marker to bring the playhead
  there, then press the event's key to **toggle** it:
  - **`B`** — exhalation breath · **`H`** — manual heartbeat
  - On an existing marker (within `snap_s`) the key **removes** it; anywhere
    else it **adds** one. Pressing the key again on a pending edit cancels it.
  - Edits are staged in memory. A banner shows `X added, Y removed` per event
    type with **Write to pkl and nc** / **Discard**. Nothing touches disk until
    Write.
  - **Write** does two things: (1) writes
    `01_raw-data/{deployment_id}_01_GUI_Notes.xlsx` — the same folder
    `DataReader` reads notes from (`data_subfolder="01_raw-data"` in
    `workflows/00_load_data.py`) — and (2) updates `outputs/data.pkl` and/or the
    output NetCDF so the edit is visible immediately. Each file is backed up
    (`.bak`) first. NetCDF-only deployments (no `data.pkl`) are handled: the
    `event_data_*` variables are rewritten in place, leaving signals untouched.
    A partial or failed write is reported in the banner and keeps the edits
    staged so you can retry.
  - The GUI notes file is the durable record: `DataReader.import_notes` reads it
    alongside `{deployment_id}_00_Notes.xlsx` in step 00, so edits survive a
    reprocess from scratch. Rows carry a `deleted` flag — `True` removes a
    matching note rather than adding one. Repeat sessions append to the file.
  - **To bind another event type**, add an entry to `EDIT_BINDINGS` in
    `mini_dash.py`; the key becomes live in the browser automatically.
- **Play heartbeats** (checkbox in the transport row). Plays
  `assets/04_fast_heart_badum_360ms.wav` as the playhead crosses each heartbeat
  event — driven by the event timestamps themselves, not a fixed interval, so it
  tracks the real rhythm including any GUI-added/removed beats.
  - Only fires during **forward 1× playback**: paused, scrubbing, rewinding, or
    any other rate re-seats the cursor silently, so a seek never machine-guns
    every beat it skipped.
  - Uses one decoded WebAudio buffer (an `<audio>` element per beat can't keep
    up at real heart rates). Beat times are pushed per window, so the browser
    holds a few hundred timestamps rather than the whole deployment.
  - Sources `heartbeat_manual_ok`, else `heartbeat_auto_detect_accepted`; the
    checkbox is disabled when a deployment has neither.
- **Transport**: play / pause, and step by **±0.1 s** or **±10 s**.
  - Keyboard: `←`/`→` = ±0.1 s, `Shift+←`/`Shift+→` = ±10 s, `Space` = play/pause.
  - Playback rate: 0.5× / 1× / 2× / 5×.

## Key config parameters

Edit the constants at the top of `mini_dash.py`:

| Param | Meaning |
|-------|---------|
| `DEFAULT_DATASET` / `DEFAULT_DEPLOYMENT` | initial dropdown selection when no CLI args are given |
| `WINDOW_MINUTES` | initial view-window width |
| `TARGET_HZ` | plot downsample target (snappiness vs. detail) |
| `STEP_SMALL` / `STEP_LARGE` | the ±0.1 s / ±10 s step sizes |
| `PLAYBACK_RATES` | rate-selector options |
| `_PALETTE` | default per-signal colors |
| `DEFAULT_EVENT_KEYS` | event keys checked on at startup (others stay listed, unchecked) |
| `STATE_FILL_OPACITY` | span alpha when `__event_styles__` gives none |
| `MAX_EVENT_MARKERS` | per-key marker cap for dense beat detections |
| `CONTEXT_MAX_POINTS` | downsample cap for the whole-deployment depth context plot |

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
