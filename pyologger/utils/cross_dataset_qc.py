from __future__ import annotations

import gc
import json
import os
import pickle
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import xarray as xr

from pyologger.utils.folder_manager import load_combined_config
from pyologger.utils.segmentation_qc import normalize_unit_token
from pyologger.utils.workflow_netcdf import latest_processing_netcdf_path


ScopeItem = Dict[str, str]


def _list_deployments(dataset_path: str) -> List[str]:
    if not os.path.isdir(dataset_path):
        return []
    return sorted(
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d)) and not str(d).startswith("00_")
    )


def _resolve_scope(
    data_root: str,
    datasets_cfg: Dict[str, Any],
    dataset_ids: Iterable[str] | None = None,
    deployment_ids: Iterable[str] | None = None,
    dataset_deployments: Dict[str, Iterable[str]] | None = None,
) -> List[ScopeItem]:
    dataset_ids = [str(x) for x in (dataset_ids or []) if str(x).strip()]
    deployment_ids = [str(x) for x in (deployment_ids or []) if str(x).strip()]
    dataset_deployments = {str(k): [str(vv) for vv in v] for k, v in (dataset_deployments or {}).items()}

    all_dataset_ids = sorted(
        ds for ds in (datasets_cfg or {}).keys()
        if os.path.isdir(os.path.join(data_root, str(ds)))
    )
    if not dataset_ids:
        dataset_ids = all_dataset_ids

    scope: List[ScopeItem] = []
    for ds in dataset_ids:
        ds_path = os.path.join(data_root, ds)
        if not os.path.isdir(ds_path):
            continue
        if ds in dataset_deployments:
            deps = [d for d in dataset_deployments[ds] if os.path.isdir(os.path.join(ds_path, d))]
        elif deployment_ids:
            deps = [d for d in deployment_ids if os.path.isdir(os.path.join(ds_path, d))]
        else:
            deps = _list_deployments(ds_path)
        for dep in deps:
            scope.append({"dataset_id": ds, "deployment_id": dep})
    return sorted(scope, key=lambda x: (x["dataset_id"], x["deployment_id"]))


def _scan_netcdf(data_root: str, dataset_id: str, deployment_id: str) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """Open the netCDF header only (no data loaded) and return (attrs, signal_channels_map)."""
    deployment_folder = os.path.join(data_root, str(dataset_id), str(deployment_id))
    netcdf_path = latest_processing_netcdf_path(deployment_folder, str(deployment_id))
    if not netcdf_path or not os.path.exists(netcdf_path):
        return {}, {}
    prefix = "signal_data_"
    try:
        with xr.open_dataset(netcdf_path) as ds:
            attrs = dict(ds.attrs)
            signals: Dict[str, List[str]] = {}
            for var in ds.variables:
                if not str(var).startswith(prefix):
                    continue
                signal_id = str(var)[len(prefix):]
                da = ds[var]
                raw = da.attrs.get("variables") or da.attrs.get("variable")
                if raw is None:
                    continue
                channels = [str(c) for c in (raw if not isinstance(raw, str) else [raw])]
                channels = [c for c in channels if c and c != "datetime"]
                if channels:
                    signals[signal_id] = sorted(channels)
            return attrs, signals
    except Exception:
        return {}, {}


def _read_data_pkl_one(data_root: str, dataset_id: str, deployment_id: str) -> Any:
    pkl_path = os.path.join(data_root, str(dataset_id), str(deployment_id), "outputs", "data.pkl")
    if not os.path.exists(pkl_path):
        return None
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _extract_channel_units(
    netcdf_attrs: Dict[str, Any],
    signal_id: str,
    channel_id: str,
    data_pkl: Any = None,
) -> Dict[str, str]:
    nc_unit = normalize_unit_token(
        netcdf_attrs.get(f"signal_info_{signal_id}_metadata_{channel_id}_unit")
        or netcdf_attrs.get(f"signal_info_{signal_id}_units")
    )
    nc_std_unit = normalize_unit_token(
        netcdf_attrs.get(f"signal_info_{signal_id}_metadata_{channel_id}_standardized_unit")
    )
    pkl_unit = ""
    pkl_std_unit = ""
    if data_pkl is not None and not (nc_std_unit or nc_unit):
        signal_info = getattr(data_pkl, "signal_info", {}) or {}
        sig_info = signal_info.get(signal_id) if isinstance(signal_info, dict) else {}
        sig_info = sig_info if isinstance(sig_info, dict) else {}
        meta = sig_info.get("metadata") if isinstance(sig_info.get("metadata"), dict) else {}
        ch_meta = meta.get(channel_id) if isinstance(meta, dict) else {}
        ch_meta = ch_meta if isinstance(ch_meta, dict) else {}
        pkl_unit = normalize_unit_token(ch_meta.get("unit") or sig_info.get("units"))
        pkl_std_unit = normalize_unit_token(ch_meta.get("standardized_unit"))
    effective = nc_std_unit or nc_unit or pkl_std_unit or pkl_unit
    return {
        "unit_data_pkl": pkl_std_unit or pkl_unit,
        "unit_netcdf": nc_std_unit or nc_unit,
        "unit_effective": effective,
    }


def _normalize_signal_filters(
    signal_filters: Iterable[str] | None = None,
    channel_filters: Iterable[str] | None = None,
) -> Tuple[set[str], Dict[str, set[str]]]:
    signals = {str(x).strip() for x in (signal_filters or []) if str(x).strip()}
    per_signal_channels: Dict[str, set[str]] = {}

    for raw in list(channel_filters or []):
        text = str(raw).strip()
        if not text:
            continue
        if "." not in text:
            signals.add(text)
            continue
        sig, ch = text.split(".", 1)
        sig = sig.strip()
        ch = ch.strip()
        if not sig or not ch:
            continue
        per_signal_channels.setdefault(sig, set()).add(ch)

    return signals, per_signal_channels


def run_cross_dataset_qc(
    config_path: str,
    dataset_ids: Iterable[str] | None = None,
    deployment_ids: Iterable[str] | None = None,
    dataset_deployments: Dict[str, Iterable[str]] | None = None,
    signal_filters: Iterable[str] | None = None,
    channel_filters: Iterable[str] | None = None,
    channel_alias_map: Dict[str, Iterable[str]] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg, _, _ = load_combined_config(config_path=config_path)
    data_root = str((cfg.get("paths") or {}).get("local_private_data") or "").strip()
    if not data_root:
        raise ValueError("Missing paths.local_private_data in config.")

    scope = _resolve_scope(
        data_root=data_root,
        datasets_cfg=cfg.get("datasets") or {},
        dataset_ids=dataset_ids,
        deployment_ids=deployment_ids,
        dataset_deployments=dataset_deployments,
    )
    if not scope:
        raise ValueError("No deployments resolved for requested scope.")

    signal_set, explicit_channels = _normalize_signal_filters(
        signal_filters=signal_filters,
        channel_filters=channel_filters,
    )
    alias_lookup: Dict[str, List[str]] = {}
    for key, values in (channel_alias_map or {}).items():
        key_text = str(key or "").strip()
        if "." not in key_text:
            continue
        normalized_values: List[str] = []
        for val in list(values or []):
            v = str(val or "").strip()
            if "." not in v:
                continue
            if v not in normalized_values:
                normalized_values.append(v)
        if key_text not in normalized_values:
            normalized_values.insert(0, key_text)
        alias_lookup[key_text] = normalized_values

    # Phase 1: scan netCDF headers only — no pkl loaded into memory.
    available_map: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
    attrs_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    channel_universe: Dict[str, set[str]] = {}

    for item in scope:
        ds, dep = item["dataset_id"], item["deployment_id"]
        key = (ds, dep)
        attrs, signals = _scan_netcdf(data_root, ds, dep)
        attrs_map[key] = attrs
        available_map[key] = signals
        for sig, channels in signals.items():
            channel_universe.setdefault(sig, set()).update(channels)

    target_channels: Dict[str, List[str]] = {}
    if explicit_channels:
        for sig, channels in explicit_channels.items():
            target_channels[sig] = sorted(channels)
    elif signal_set:
        for sig in sorted(signal_set):
            chans = sorted(channel_universe.get(sig, set()))
            if chans:
                target_channels[sig] = chans
    else:
        for sig, chans in sorted(channel_universe.items()):
            target_channels[sig] = sorted(chans)

    if not target_channels:
        raise ValueError("No signal/channel targets resolved. Check scope and filters.")

    # Phase 2: build detail rows. Units come from netCDF attrs; only load pkl (one at a
    # time, immediately released) when netCDF attrs lack unit info for a present channel.
    detail_rows: List[Dict[str, Any]] = []
    for item in scope:
        ds, dep = item["dataset_id"], item["deployment_id"]
        key = (ds, dep)
        avail = available_map[key]
        attrs = attrs_map[key]

        # Determine which present channels are missing unit info in the netCDF so we
        # know whether we need to load the pkl at all for this deployment.
        needs_pkl_for: List[Tuple[str, str]] = []
        present_channels: List[Tuple[str, str, str, str]] = []  # (sig, ch, rsig, rch)
        for sig, channels in target_channels.items():
            for ch in channels:
                target_key = f"{sig}.{ch}"
                candidate_keys = alias_lookup.get(target_key, [target_key])
                resolved_signal, resolved_channel = sig, ch
                present = False
                for candidate_key in candidate_keys:
                    if "." not in str(candidate_key):
                        continue
                    cand_sig, cand_ch = str(candidate_key).split(".", 1)
                    if cand_sig in avail and cand_ch in avail[cand_sig]:
                        present = True
                        resolved_signal, resolved_channel = cand_sig, cand_ch
                        break
                if present:
                    present_channels.append((sig, ch, resolved_signal, resolved_channel))
                    nc_unit = attrs.get(f"signal_info_{resolved_signal}_metadata_{resolved_channel}_unit") \
                        or attrs.get(f"signal_info_{resolved_signal}_units")
                    nc_std_unit = attrs.get(f"signal_info_{resolved_signal}_metadata_{resolved_channel}_standardized_unit")
                    if not (nc_std_unit or nc_unit):
                        needs_pkl_for.append((resolved_signal, resolved_channel))
                else:
                    present_channels.append((sig, ch, resolved_signal, resolved_channel))

        # Load pkl only if at least one present channel is missing netCDF unit info.
        data_pkl = None
        if needs_pkl_for:
            data_pkl = _read_data_pkl_one(data_root, ds, dep)

        for sig, ch, resolved_signal, resolved_channel in present_channels:
            present = (resolved_signal in avail and resolved_channel in avail.get(resolved_signal, []))
            units = {"unit_data_pkl": "", "unit_netcdf": "", "unit_effective": ""}
            if present:
                units = _extract_channel_units(attrs, resolved_signal, resolved_channel, data_pkl)
            detail_rows.append(
                {
                    "dataset_id": ds,
                    "deployment_id": dep,
                    "signal_id": sig,
                    "channel_id": ch,
                    "resolved_signal_id": resolved_signal,
                    "resolved_channel_id": resolved_channel,
                    "resolved_channel_key": f"{resolved_signal}.{resolved_channel}",
                    "present": present,
                    "unit_data_pkl": units["unit_data_pkl"],
                    "unit_netcdf": units["unit_netcdf"],
                    "unit_effective": units["unit_effective"],
                }
            )

        # Release pkl immediately after this deployment is processed.
        if data_pkl is not None:
            del data_pkl
            gc.collect()

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        return detail_df, detail_df

    summary_rows: List[Dict[str, Any]] = []
    total_scopes = int(detail_df[["dataset_id", "deployment_id"]].drop_duplicates().shape[0])
    for (sig, ch), sub in detail_df.groupby(["signal_id", "channel_id"], sort=True):
        n_present = int(sub["present"].sum())
        present_sub = sub[sub["present"]].copy()
        units_seen = sorted({u for u in present_sub["unit_effective"].astype(str).tolist() if str(u).strip()})
        unit_match_all = bool(len(units_seen) <= 1) if n_present > 0 else False
        present_in_all = bool(n_present == total_scopes)
        ds_cov = (
            present_sub.groupby("dataset_id")["deployment_id"].nunique().to_dict()
            if not present_sub.empty
            else {}
        )
        summary_rows.append(
            {
                "signal_id": sig,
                "channel_id": ch,
                "n_scopes": total_scopes,
                "n_present": n_present,
                "presence_rate": (float(n_present) / float(total_scopes)) if total_scopes else 0.0,
                "present_in_all_scopes": present_in_all,
                "units_seen": ",".join(units_seen),
                "unit_count": len(units_seen),
                "unit_match_all_present_scopes": unit_match_all,
                "matched_across_datasets_and_deployments": bool(present_in_all and unit_match_all),
                "dataset_presence_counts": json.dumps(ds_cov, sort_keys=True),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["signal_id", "channel_id"]).reset_index(drop=True)
    return summary_df, detail_df
