# pyologger NDP Workshop Guide

Step-by-step guide for running the pyologger demo/workshop inside an NDP
workspace: pulling demo data, setting up config, running the processing
pipeline, viewing results in mini-dash, and (optionally) running feature
generation, clustering, and a sleep-stage random forest classifier.

## Quick Launch Reference

Run each from inside `pyologger/`, with the **root** `EcoViz_DiveDB` venv
active (`source ../venv/bin/activate` — not `pyologger/venv`, which has a
stale, non-editable DiveDB install missing `services/immich_service.py`).

```bash
# mini-dash — default (orca) deployment, no args needed
python dash/mini-dash/mini_dash.py

# mini-dash — SnoozySuzy demo deployment (juvenile elephant seal sleep study)
python dash/mini-dash/mini_dash.py --dataset mian-juv-nese_sleep_lml-ano_JKB --deployment 2020-04-10_mian-002 --port 8070

# mini-dash — SnoozySuzy, trimmed 3-hour demo slice (outputs_demo/, matches the NSF_demo_data.zip content)
python dash/mini-dash/mini_dash.py --dataset mian-juv-nese_sleep_lml-ano_JKB --deployment 2020-04-10_mian-002 --port 8070 --demo

# integrated_dash — blue whale heart rate deployment
python dash/integrated/integrated_dash.py --dataset wild-whale-adult_hr-sr_JG-PP --deployment 2018-08-27_bamu-002 --port 8061
```

Assumes you've already completed general NDP onboarding (sign-in, workspace
launch, VS Code). This guide picks up once you have a VS Code terminal open
inside your NDP workspace with the `pyologger` repo cloned.

## Quick Reference

| Step | What you're doing | Est. time |
|---|---|---|
| 1 | Clone pyologger and set up the venv | 5–20 min, installs are slow |
| 2 | Download and unzip the demo dataset (Google Drive) | 2–5 min |
| 3 | Run `setup_demo.py` | 1–2 min |
| 4 | Fill in `.env` (Immich key, optional) | 1 min |
| 5 | Run the Snakemake processing pipeline | 2–10 min |
| 6 | Launch mini-dash and view results | 1 min to launch |
| 7 | (Optional) Feature generation, clustering, sleep-stage RF | 5–15 min |

## Before You Start

**You'll need:** an NDP workspace with the `pyologger` repo cloned, and a
Python environment set up (`python -m venv venv && source venv/bin/activate
&& pip install .` from inside `pyologger/`).

The demo deployment is `2020-04-10_mian-002` — a juvenile northern elephant
seal ("SnoozySuzy") sleep-study deployment with EEG/EOG/EMG, ECG, and a
motion+depth tag. A pre-trimmed slice, packaged as a single zip, is
distributed via Google Drive specifically for this workshop — no lab
credentials needed for anything in this guide.

## Step 1: Download the Demo Dataset

**[Download demo data](https://drive.google.com/file/d/1zqO8pzu48BWOJp2OlvllIueT_P0jloCi/view?usp=sharing)**
(`NSF_demo_data.zip`), and unzip it into a folder that sits *next to* your
`pyologger/` clone — the folder name doesn't matter, the next step finds it
automatically:

```
_User-Persistent-Storage_CephBlock_/     (or wherever you cloned pyologger)
├── pyologger/                            <- this repo
└── NSF_demo_data/                        <- unzipped download
    ├── 2020-04-10_mian-002_data.pkl
    ├── 2020-04-10_mian-002.nc
    ├── 2020-04-10_mian-002_trimmed.EDF
    ├── 2020-04-10_mian-002_NL-D2_001_trimmed.csv
    ├── metadata_snapshot.pkl
    └── parameter_log.json
```

> Don't worry about exactly what you name the unzipped folder — `setup_demo.py`
> (Step 2) searches every sibling folder next to `pyologger/` for the expected
> files, so there's no need to `cd` into it or rename anything.

## Step 2: Run `setup_demo.py`

From inside your `pyologger/` clone, with your venv active:

```bash
cd pyologger
python setup_demo.py
```

This single command:

1. Finds the unzipped demo files in whichever sibling folder they landed in
2. Builds the dataset/deployment hierarchy pyologger expects, inside a new
   **sibling** folder called `pyologger_demo_data/` (kept separate from the
   repo itself, so `pyologger/` stays clean) — including the pre-trimmed
   `metadata_snapshot.pkl` and `parameter_log.json` that came in the zip, so
   `workflows/00_load_data.py` can run fully offline with no Notion token
3. Installs `demo_config.yaml` as `config.yaml` (only if `config.yaml`
   doesn't already exist — safe to re-run)
4. Installs `.env` from `.env.example` (only if `.env` doesn't already
   exist), with Notion tokens and unused variables commented out, and
   `CONFIG_PATH` filled in automatically

> **If you already have a stale `config.yaml`** from a previous NDP session
> (e.g. from an older version of this workshop, before the `paths:` block was
> added), `setup_demo.py` will NOT overwrite it — delete it yourself first
> (`rm config.yaml`) and re-run the script. A quick way to check: `grep paths
> config.yaml` should show a `paths:` block; if it prints nothing, your
> `config.yaml` is stale.

## Step 3: Fill In `.env` (Optional — for Video Playback)

Open `.env` in the editor. Everything works for signal processing and
viewing signals/events without touching this file. If you want synchronized
video playback in mini-dash, fill in the two Immich lines with the
workshop's restricted demo credentials (ask the workshop organizer if you
don't have them):

```
IMMICH_API_KEY=<restricted demo key>
IMMICH_BASE_URL=https://jkb-immich.nrp-nautilus.io/api
```

> The demo Immich key is deliberately scoped to read-only access to exactly
> one shared album (`DepID_2020-04-10_mian-002`, 7 videos + 1 photo) — no
> admin, system, or other-album access.

## Step 4: Run the Processing Pipeline

```bash
snakemake --cores 1
```

This runs the full pyologger pipeline (data import → calibration → detection
steps) against the demo deployment, using the offline metadata snapshot
installed in Step 2 (no live Notion connection needed).

> **`KeyError: 'paths'`?** Your `config.yaml` doesn't have a `paths:` block —
> see the stale-config callout in Step 2.

## Step 5: View Results in mini-dash

```bash
python dash/mini-dash/mini_dash.py --dataset mian-juv-nese_sleep_lml-ano_JKB --deployment 2020-04-10_mian-002 --port 8070
```

> mini-dash's own defaults point at a different (orca) deployment — always
> pass `--dataset` and `--deployment` explicitly for the workshop demo.

A **"View in Browser"** button/notification should appear once the server
starts; click it, or find port `8070` under VS Code's Ports panel and open
it. The URL will look like:

```
https://ndp-jupyterhub.nrp-nautilus.io/user/<your-email>/vscode/proxy/8070/
```

You should see: signal plot (depth, ECG, heart rate, PRH, etc.), sleep-state
event overlays, and (if you filled in Step 3) synchronized video clips.

---

## Step 7: Feature Generation, Clustering, and Sleep-Stage RF (Optional)

This section walks through pyologger's segmentation/machine-learning stack —
the same pipeline behind the lab's sleep-classification analyses — scaled
down to run against just the one demo deployment in a few minutes.

It has three stages, run in order for a named **segmentation run**:

1. **Feature generation** (`workflows/11_feature_generation.py`) — slices
   each signal into overlapping 30 s windows and computes summary features
   per window (mean/std/slope/etc., depending on each channel's configured
   `feature_set`).
2. **Unsupervised clustering** (`workflows/12_unsupervised_segmentation.py`)
   — clusters those windows (k=2 and k=5 passes) using only the depth
   channel, as an unsupervised look at behavioral/postural states.
3. **Supervised random forest** (`workflows/13_supervised_segmentation.py`)
   — trains a random forest to classify sleep stage (SWS vs. REM) against
   the deployment's manually-scored labels, with **channel ablation**: it
   re-trains the classifier once per feature group, so you can see accuracy
   climb (or not) as more sensor channels are added.

Two demo segmentation runs are already defined in `segmentation_runs.yaml`,
both scoped to just `2020-04-10_mian-002` (a plain random 70/30 train/test
split within that one deployment — not a cross-deployment holdout, since the
demo only ships one deployment's data):

| Run name | Ablation feature groups |
|---|---|
| `demo_sleep_rf_full` | TDR Only → TDR+SR → TDR+SR+PRH → +HR → +HRV → +EEG_DELTA (6 steps, ending with the full EEG+ECG+HR+motion stack) |
| `demo_sleep_rf_hr_only` | HR Only → HR+HRV (heart-rate-derived channels only, no EEG, no motion) |

Running both back-to-back is the point: `demo_sleep_rf_full`'s final row
shows what accuracy looks like with everything (brain, heart, and motion
signals), while `demo_sleep_rf_hr_only` shows how much of that comes from
heart rate alone — a quick, concrete answer to "do you actually need the EEG
to detect sleep stage?"

### Run it

Make sure `run_selection.segmentation: true` is set in `config.yaml` (it's
`false` by default in the base demo config, since Steps 1–6 above don't need
it):

```bash
sed -i.bak 's/segmentation: false/segmentation: true/' config.yaml
```

Then target each run's summary marker directly — Snakemake resolves feature
generation → clustering → RF automatically as prerequisites:

```bash
# Full sensor stack (EEG + ECG/HR + motion)
snakemake --cores 1 "../pyologger_demo_data/00_Meta-Analysis/segmentation/demo_sleep_rf_full/14_summary.done"

# Heart-rate-only comparison
snakemake --cores 1 "../pyologger_demo_data/00_Meta-Analysis/segmentation/demo_sleep_rf_hr_only/14_summary.done"
```

Each command runs all three stages for that run (skipping any already
completed — Snakemake tracks per-stage `.done` markers, so re-running after
a failure resumes rather than restarting). Outputs land under:

```
pyologger_demo_data/00_Meta-Analysis/segmentation/<run_name>/
  features/features_filtered.parquet     # stage 1 output
  clustering/clustered_windows.parquet   # stage 2 output
  supervised/                            # stage 3: per-ablation-group metrics, confusion matrices
  qc/qc_channels.csv
```

Look in `supervised/` for a per-feature-group accuracy/F1 table — that's
where the "does EEG help" comparison lives. `workflows/14_summary.py`
(the last stage) also generates an interactive HTML review plot for the full
deployment, similar to the one produced during Step 4's main pipeline run.

### Running a single stage directly (without Snakemake)

If you'd rather step through one stage at a time — useful for a live
teaching demo — each workflow script can be invoked directly. `--config`
points at your installed `config.yaml`; `SEGMENTATION_RUNS_PATH` is
optional (it defaults to `segmentation_runs.yaml` next to `config.yaml`,
which is where it already lives):

```bash
python3 workflows/11_feature_generation.py --config config.yaml --run-name demo_sleep_rf_full
python3 workflows/12_unsupervised_segmentation.py --config config.yaml --run-name demo_sleep_rf_full
python3 workflows/13_supervised_segmentation.py --config config.yaml --run-name demo_sleep_rf_full
```

> Run stage 1 (`11_feature_generation.py`) for a run name once before its
> stage 2/3 — they read stage 1's parquet output. Stages 2 and 3 don't
> depend on each other and can run in either order.

---

## Known Issues

- **Video playback doesn't work through NDP's VS Code proxy.** Signals,
  events, and plots render correctly, but clicking play on a video shows a
  black frame that never advances (Chrome) or a broken-video icon (Safari).
  Root cause confirmed: mini-dash's own video-proxy route is correct —
  tested directly, it returns proper `206 Partial Content` with correct
  `Content-Range` headers for the real 680MB demo video files, and Immich's
  own endpoint handles Range requests correctly too. But the same request
  through NDP's `.../vscode/proxy/<port>/` layer comes back as a plain `200`
  with no `Content-Range`/`Accept-Ranges` — meaning NDP's proxy is stripping
  the Range header or collapsing `206→200` somewhere in transit. Not fixable
  from the pyologger side. **Status: reported to NDP admin, awaiting fix.**
- **Stale `config.yaml` from older workshop runs lacks the `paths:` block**,
  causing `KeyError: 'paths'` in Snakemake. `setup_demo.py` never overwrites
  an existing `config.yaml`, so this has to be caught manually — see Step 2's
  callout.
- **Pelican/OSDF (`jkb-lab-public` namespace) became unreliable as of
  2026-09-17** — the origin (`unl-origin.nationalresearchplatform.org:8443`)
  started timing out, and the OSDF director reported "no origins found" for
  both `jkb-lab` and `jkb-lab-public`. This is why the workshop now
  distributes demo data as a Google Drive zip instead of fetching it live via
  `pelican`/`osdf://`. Re-link to Pelican after the workshop once the origin
  is confirmed healthy again (check
  https://osdf-director.osg-htc.org/view/director/).

## For Maintainers

This page should be updated whenever these files change:

- `pyologger/setup_demo.py`
- `pyologger/demo_config.yaml`
- `pyologger/segmentation_runs.yaml` (the `demo_sleep_rf_full` /
  `demo_sleep_rf_hr_only` runs)
- `pyologger/dash/mini-dash/mini_dash.py` (especially anything proxy/NDP-related)
- The demo files inside `NSF_demo_data.zip` on Google Drive (staged at
  `.../mian-juv-nese_sleep_lml-ano_JKB/00_Demo-Data/2026_NSF-APS_Workshop_Demo-Data/`;
  re-zip and re-share the link if any of the 6 files inside change)
