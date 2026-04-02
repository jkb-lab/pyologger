from __future__ import annotations

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


def _read_netcdf_attrs(data_root: str, dataset_id: str, deployment_id: str) -> Dict[str, Any]:
    deployment_folder = os.path.join(data_root, str(dataset_id), str(deployment_id))
    netcdf_path = latest_processing_netcdf_path(deployment_folder, str(deployment_id))
    if not netcdf_path or not os.path.exists(netcdf_path):
        return {}
    try:
        with xr.open_dataset(netcdf_path) as ds:
            return dict(ds.attrs)
    except Exception:
        return {}


def _read_data_pkl(data_root: str, dataset_id: str, deployment_id: str):
    pkl_path = os.path.join(data_root, str(dataset_id), str(deployment_id), "outputs", "data.pkl")
    if not os.path.exists(pkl_path):
        return None
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _extract_signal_channels(data_pkl: Any) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    signal_data = getattr(data_pkl, "signal_data", {}) or {}
    for signal_id, sdf in signal_data.items():
        if sdf is None or not hasattr(sdf, "columns"):
            continue
        channels = [str(c) for c in list(sdf.columns) if str(c) != "datetime"]
        if channels:
            out[str(signal_id)] = sorted(channels)
    return out


def _extract_channel_units(
    data_pkl: Any,
    netcdf_attrs: Dict[str, Any],
    signal_id: str,
    channel_id: str,
) -> Dict[str, str]:
    signal_info = getattr(data_pkl, "signal_info", {}) or {}
    sig_info = signal_info.get(signal_id) if isinstance(signal_info, dict) else {}
    sig_info = sig_info if isinstance(sig_info, dict) else {}
    meta = sig_info.get("metadata") if isinstance(sig_info.get("metadata"), dict) else {}
    ch_meta = meta.get(channel_id) if isinstance(meta, dict) else {}
    ch_meta = ch_meta if isinstance(ch_meta, dict) else {}

    pkl_unit = normalize_unit_token(ch_meta.get("unit") or sig_info.get("units"))
    pkl_std_unit = normalize_unit_token(ch_meta.get("standardized_unit"))
    nc_unit = normalize_unit_token(
        netcdf_attrs.get(f"signal_info_{signal_id}_metadata_{channel_id}_unit")
        or netcdf_attrs.get(f"signal_info_{signal_id}_units")
    )
    nc_std_unit = normalize_unit_token(
        netcdf_attrs.get(f"signal_info_{signal_id}_metadata_{channel_id}_standardized_unit")
    )
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

    available_map: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
    data_obj_map: Dict[Tuple[str, str], Any] = {}
    attrs_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    channel_universe: Dict[str, set[str]] = {}

    for item in scope:
        ds, dep = item["dataset_id"], item["deployment_id"]
        key = (ds, dep)
        data_pkl = _read_data_pkl(data_root, ds, dep)
        data_obj_map[key] = data_pkl
        attrs_map[key] = _read_netcdf_attrs(data_root, ds, dep)
        avail = _extract_signal_channels(data_pkl) if data_pkl is not None else {}
        available_map[key] = avail
        for sig, channels in avail.items():
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

    detail_rows: List[Dict[str, Any]] = []
    for item in scope:
        ds, dep = item["dataset_id"], item["deployment_id"]
        key = (ds, dep)
        data_pkl = data_obj_map[key]
        avail = available_map[key]
        attrs = attrs_map[key]
        for sig, channels in target_channels.items():
            for ch in channels:
                target_key = f"{sig}.{ch}"
                candidate_keys = alias_lookup.get(target_key, [target_key])
                resolved_signal = sig
                resolved_channel = ch
                present = False
                for candidate_key in candidate_keys:
                    if "." not in str(candidate_key):
                        continue
                    cand_sig, cand_ch = str(candidate_key).split(".", 1)
                    if cand_sig in avail and cand_ch in avail[cand_sig]:
                        present = True
                        resolved_signal = cand_sig
                        resolved_channel = cand_ch
                        break
                units = {"unit_data_pkl": "", "unit_netcdf": "", "unit_effective": ""}
                if present and data_pkl is not None:
                    units = _extract_channel_units(data_pkl, attrs, resolved_signal, resolved_channel)
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
