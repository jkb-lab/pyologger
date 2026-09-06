"""Build a minimal, public metadata_snapshot.pkl for one or more deployments.

The real snapshot (00_Metadata/metadata_snapshot.pkl) is a `Metadata` object
whose `.metadata` dict holds one DataFrame per Notion database, covering the
whole lab (all datasets/deployments/collaborators, published or not). This
script filters that down to just the rows relevant to the requested
deployments/datasets, so a scoped snapshot is safe to hand to a demo,
collaborator, or workshop without exposing unrelated lab data.

Filtering strategy: join by string containment on the resolved (human-readable)
relation columns each table already carries (e.g. recording_DB["Deployment ID"]
contains "2020-04-10_mian-002"), since Metadata.find_relations() already resolves
Notion relation UUIDs to names before the snapshot is pickled.

collaborator_DB carries real names/affiliations/ORCIDs and isn't tied to a
single deployment in a way that's safe to filter automatically, so it's kept
as an empty (same-columns) DataFrame — nothing in the processing workflows
(00_load_data.py) reads it beyond assigning it to an unused variable.

Usage:
    # One deployment
    python scripts/trim_metadata_snapshot.py \
        --source /path/to/real/00_Metadata/metadata_snapshot.pkl \
        --deployment-id 2020-04-10_mian-002 \
        --out /tmp/trimmed_metadata_snapshot.pkl

    # Multiple deployments and/or whole datasets (all deployments under them)
    python scripts/trim_metadata_snapshot.py \
        --source /path/to/real/00_Metadata/metadata_snapshot.pkl \
        --deployment-id 2020-04-10_mian-002 --deployment-id 2020-04-24_mian-003 \
        --dataset-id oror-adult-orca_hr-sr-vid_sw_JKB-PP \
        --out /tmp/trimmed_metadata_snapshot.pkl

Review the printed row counts and the output file's contents before
publishing or handing off the result — especially any table with more rows
than expected.
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

DROP_TABLES = {"collaborator_DB"}


def contains_any(df: pd.DataFrame, column: str, needles: list[str]) -> pd.Series:
    if column not in df.columns or not needles:
        return pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    for needle in needles:
        mask = mask | df[column].astype(str).str.contains(needle, na=False, regex=False)
    return mask


def resolve_deployment_ids(
    deployment_df: pd.DataFrame, deployment_ids: list[str], dataset_ids: list[str]
) -> list[str]:
    resolved = set(deployment_ids)
    if dataset_ids and "Dataset ID" in deployment_df.columns:
        mask = contains_any(deployment_df, "Dataset ID", dataset_ids)
        resolved |= set(deployment_df.loc[mask, "Deployment ID"].dropna().astype(str))
    elif dataset_ids:
        raise SystemExit(
            "deployment_DB has no 'Dataset ID' column to resolve --dataset-id against; "
            "pass --deployment-id values directly instead."
        )
    if not resolved:
        raise SystemExit("No deployment IDs resolved from --deployment-id/--dataset-id.")
    return sorted(resolved)


def filter_table(
    name: str,
    df: pd.DataFrame,
    *,
    deployment_ids: list[str],
    animal_ids: list[str],
    dataset_names: list[str],
    locality_needles: list[str],
    montage_needles: list[str],
) -> pd.DataFrame:
    if name in DROP_TABLES:
        return df.iloc[0:0].copy()

    if name in ("signal_DB", "standardizedchannel_DB"):
        # Shared vocabulary tables, no PII, safe to keep in full.
        return df.copy()

    if name == "deployment_DB":
        mask = contains_any(df, "Deployment ID", deployment_ids) | contains_any(df, "page_id", deployment_ids)
        return df.loc[mask].copy()

    if name in ("recording_DB", "procedure_DB", "observation_DB"):
        mask = contains_any(df, "Deployment ID", deployment_ids)
        return df.loc[mask].copy()

    if name == "animal_DB":
        mask = contains_any(df, "Organism ID", animal_ids) | contains_any(df, "Deployments", deployment_ids)
        return df.loc[mask].copy()

    if name == "dataset_DB":
        mask = contains_any(df, "Dataset ID", dataset_names)
        return df.loc[mask].copy()

    if name == "location_DB":
        mask = contains_any(df, "Locality", locality_needles) | contains_any(df, "page_id", locality_needles)
        return df.loc[mask].copy()

    if name == "montage_DB":
        mask = contains_any(df, "Datasets", dataset_names)
        return df.loc[mask].copy()

    if name == "attachment_DB":
        mask = contains_any(df, "Dataset DB", dataset_names)
        return df.loc[mask].copy()

    if name == "logger_DB":
        mask = contains_any(df, "Recordings", deployment_ids)
        return df.loc[mask].copy()

    if name == "originalchannel_DB":
        mask = contains_any(df, "Montages", montage_needles)
        return df.loc[mask].copy()

    return df.iloc[0:0].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Path to the real metadata_snapshot.pkl")
    parser.add_argument(
        "--deployment-id",
        action="append",
        default=[],
        dest="deployment_ids",
        help="Deployment ID to keep, e.g. 2020-04-10_mian-002 (repeatable)",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=[],
        dest="dataset_ids",
        help="Dataset ID (folder name) to keep all deployments of, e.g. mian-juv-nese_sleep_lml-ano_JKB (repeatable)",
    )
    parser.add_argument("--out", required=True, help="Where to write the trimmed snapshot")
    args = parser.parse_args()

    if not args.deployment_ids and not args.dataset_ids:
        raise SystemExit("Pass at least one --deployment-id or --dataset-id.")

    with open(args.source, "rb") as f:
        payload = pickle.load(f)
    metadata = payload["metadata_obj"]

    deployment_df = metadata.metadata["deployment_DB"]
    deployment_ids = resolve_deployment_ids(deployment_df, args.deployment_ids, args.dataset_ids)
    animal_ids = sorted({dep_id.split("_")[1] if "_" in dep_id else dep_id for dep_id in deployment_ids})

    dep_mask = contains_any(deployment_df, "Deployment ID", deployment_ids) | contains_any(
        deployment_df, "page_id", deployment_ids
    )
    dep_rows = deployment_df.loc[dep_mask]
    if dep_rows.empty:
        raise SystemExit(f"No deployment_DB rows found for {deployment_ids}")

    animal_df = metadata.metadata["animal_DB"]
    animal_mask = contains_any(animal_df, "Organism ID", animal_ids) | contains_any(
        animal_df, "Deployments", deployment_ids
    )
    animal_rows = animal_df.loc[animal_mask]
    dataset_names = sorted(set(animal_rows.get("Dataset ID", pd.Series(dtype=str)).dropna().astype(str)))

    locality_needles = sorted(
        {
            str(v)
            for col in ("Deployment Locality", "Recovery Locality")
            for v in dep_rows.get(col, pd.Series(dtype=str)).dropna().astype(str)
            if str(v) and str(v) != "nan"
        }
    )

    montage_df = metadata.metadata["montage_DB"]
    montage_mask = contains_any(montage_df, "Datasets", dataset_names)
    montage_needles = list(montage_df.loc[montage_mask, "page_id"].astype(str))

    print(
        f"Trimming snapshot to deployment_ids={deployment_ids} animal_ids={animal_ids} "
        f"dataset_names={dataset_names}\n"
    )
    trimmed = {}
    for name, df in metadata.metadata.items():
        new_df = filter_table(
            name,
            df,
            deployment_ids=deployment_ids,
            animal_ids=animal_ids,
            dataset_names=dataset_names,
            locality_needles=locality_needles,
            montage_needles=montage_needles,
        )
        trimmed[name] = new_df
        flag = " (DROPPED for privacy)" if name in DROP_TABLES else ""
        print(f"  {name}: {len(df)} -> {len(new_df)} rows{flag}")

    metadata.metadata = trimmed
    metadata.notion = None
    metadata.data_source_cache = {}
    metadata.page_id_lookup = {}

    snapshot_meta = {
        "notion_version": getattr(metadata, "notion_version", None),
        "created_at": datetime.now().isoformat(),
        "class": "Metadata",
        "trimmed_for_deployments": deployment_ids,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"snapshot_meta": snapshot_meta, "metadata_obj": metadata}, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nWrote trimmed snapshot to {out_path}")
    print("Review its contents before publishing or handing it off — especially any table with more rows than expected.")


if __name__ == "__main__":
    main()
