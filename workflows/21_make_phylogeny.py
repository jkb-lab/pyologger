"""
Snakemake workflow script: make_phylogeny

Two-phase design mirrors 20_make_map.py:

  resolve  — validate config, fetch PhyloPic UUIDs, write a JSON manifest
  render   — build NCBI topology, annotate with silhouettes, export PNG/SVG/Newick

Usage (called by Snakemake rules):
  python3 workflows/21_make_phylogeny.py --config config.yaml --run-name <name> resolve --output /tmp/<name>_phylo_manifest.json
  python3 workflows/21_make_phylogeny.py --config config.yaml --run-name <name> render  --output /tmp/<name>_phylo_summary.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ── config helpers ────────────────────────────────────────────────────────────

def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _run_cfg(config: dict, run_name: str) -> dict:
    runs = config.get("make_phylogeny_runs") or {}
    if run_name not in runs:
        raise ValueError(
            f"Run '{run_name}' not found in make_phylogeny_runs. "
            f"Available: {sorted(runs)}"
        )
    return dict(runs[run_name])


# ── dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class PhylogenyRunContext:
    run_name: str
    cfg: dict
    output_dir: Path
    species: list[str]
    # populated during resolve
    phylopic_cache: dict[str, str] = field(default_factory=dict)  # species -> png path


# ── PhyloPic API ──────────────────────────────────────────────────────────────
# Uses /resolve/ncbi.nlm.nih.gov/taxid/{taxid} (current v2 API).
# The old /nodes?filter_name= endpoint returns HTTP 410 Gone.

PHYLOPIC_API = "https://api.phylopic.org"
_LINEAGE_SKIP = {1, 131567}  # root, cellular organisms


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _fetch_png_url_for_taxid(taxid: int) -> str | None:
    """Return a remote PNG URL for taxid via the PhyloPic v2 resolve endpoint, or None."""
    import urllib.error
    import time
    try:
        resolve_url = (
            f"{PHYLOPIC_API}/resolve/ncbi.nlm.nih.gov/taxid/{taxid}"
            "?embed%5BprimaryImage%5D=true"
        )
        node_data = _get_json(resolve_url)
        img_href = node_data.get("_links", {}).get("primaryImage", {}).get("href", "")
        if not img_href:
            return None
        img_uuid = img_href.rstrip("/").split("/")[-1].split("?")[0]
        raster_url = f"{PHYLOPIC_API}/images/{img_uuid}?embed%5BrasterFiles%5D=true"
        raster_files = _get_json(raster_url).get("_links", {}).get("rasterFiles", [])
        png_url = raster_files[-1].get("href", "") if raster_files else ""
        time.sleep(0.12)
        return png_url or None
    except urllib.error.HTTPError:
        return None
    except Exception as exc:
        print(f"  [phylopic] warning: resolve failed for taxid {taxid}: {exc}")
        return None


def _fetch_phylopic_silhouettes(
    species: list[str],
    cache_dir: Path,
    ncbi,  # ete3 NCBITaxa instance
    name_to_taxid: dict[str, list[int]],
) -> dict[str, str]:
    """
    Return mapping species_name -> remote PNG URL.
    Uses NCBI taxid resolution, then walks up lineage as fallback.
    Results are cached to cache_dir/phylopic_url_cache.json.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "phylopic_url_cache.json"
    url_cache: dict[str, str | None] = {}
    if cache_file.exists():
        url_cache = json.loads(cache_file.read_text())

    result: dict[str, str] = {}

    for sp in species:
        taxids = name_to_taxid.get(sp)
        if not taxids:
            print(f"  [phylopic] no NCBI taxid for '{sp}' — skipping")
            continue
        taxid = taxids[0]
        cache_key = str(taxid)

        if cache_key in url_cache:
            url = url_cache[cache_key]
            if url:
                result[sp] = url
                print(f"  [phylopic] cached: {sp}")
            else:
                print(f"  [phylopic] cached miss: {sp}")
            continue

        print(f"  [phylopic] fetching: {sp} (taxid={taxid})")
        url = _fetch_png_url_for_taxid(taxid)

        if url is None:
            # Walk up NCBI lineage toward root
            try:
                lineage = ncbi.get_lineage(taxid)
                names_map = ncbi.get_taxid_translator(lineage)
                ranks_map = ncbi.get_rank(lineage)
                for ancestor in reversed(lineage[:-1]):
                    if ancestor in _LINEAGE_SKIP:
                        continue
                    url = _fetch_png_url_for_taxid(ancestor)
                    if url:
                        aname = names_map.get(ancestor, str(ancestor))
                        arank = ranks_map.get(ancestor, "?")
                        print(f"    → matched ancestor [{arank}] {aname}")
                        break
            except Exception as exc:
                print(f"  [phylopic] lineage fallback error for '{sp}': {exc}")

        url_cache[cache_key] = url
        if url:
            result[sp] = url
        else:
            print(f"  [phylopic] no match for '{sp}' — will skip silhouette")

    cache_file.write_text(json.dumps(url_cache, indent=2))
    return result


def _common_name_map(name_to_taxid: dict[str, list[int]], ncbi_db: str | None) -> dict[str, str]:
    """
    Return {scientific_name: common_name} by querying the NCBI taxa SQLite DB directly.
    Falls back to an empty string if not found.
    """
    import sqlite3
    import re

    if not ncbi_db or not Path(ncbi_db).exists():
        return {}

    result: dict[str, str] = {}
    try:
        con = sqlite3.connect(ncbi_db)
        cur = con.cursor()
        for sp, taxids in name_to_taxid.items():
            if not taxids:
                continue
            taxid = taxids[0]
            row = cur.execute(
                "SELECT common FROM species WHERE taxid=?", (taxid,)
            ).fetchone()
            if row and row[0]:
                result[sp] = row[0]
        con.close()
    except Exception as exc:
        print(f"  [phylopic] common-name lookup failed: {exc}")

    return result


def _safe_filename(s: str) -> str:
    """Replace spaces and punctuation with underscores for use in filenames."""
    import re
    return re.sub(r"[^\w]+", "_", s).strip("_")


def _download_silhouettes(
    url_map: dict[str, str],
    cache_dir: Path,
    common_names: dict[str, str] | None = None,
) -> dict[str, str]:
    """Download remote PNG URLs to cache_dir; return species -> local path.

    Files are named ``Common_name_Scientific_name.png`` when a common name is
    available, otherwise ``Scientific_name.png``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_map: dict[str, str] = {}
    for sp, url in url_map.items():
        common = (common_names or {}).get(sp, "")
        if common:
            fname = f"{_safe_filename(common)}_{_safe_filename(sp)}.png"
        else:
            fname = f"{_safe_filename(sp)}.png"
        dest = cache_dir / fname
        if not dest.exists():
            try:
                with urllib.request.urlopen(url, timeout=15) as resp:
                    dest.write_bytes(resp.read())
                print(f"  [phylopic] downloaded → {fname}")
            except Exception as exc:
                print(f"  [phylopic] download failed for '{sp}': {exc}")
                continue
        local_map[sp] = str(dest)
    return local_map


# ── resolve ───────────────────────────────────────────────────────────────────

def _init_ncbi(ncbi_db: str | None):
    from ete3 import NCBITaxa
    kwargs = {"dbfile": ncbi_db} if ncbi_db else {}
    return NCBITaxa(**kwargs)


def cmd_resolve(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    cfg = _run_cfg(config, args.run_name)

    species = cfg.get("species") or []
    if not species:
        raise ValueError(f"Run '{args.run_name}' has no species list.")

    output_dir = Path(cfg.get("output_dir", ".")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    ncbi_db = cfg.get("ncbi_db")
    if ncbi_db and not Path(ncbi_db).exists():
        print(
            f"  [resolve] warning: ncbi_db not found at '{ncbi_db}'. "
            "ete3 will download taxdump on first use."
        )

    annotations = cfg.get("annotations") or {}
    phylopic_url_map: dict[str, str] = {}
    if annotations.get("phylopic", False):
        ncbi = _init_ncbi(ncbi_db)
        name_to_taxid = ncbi.get_name_translator(species)
        cache_dir = output_dir / "phylopic_cache"
        phylopic_url_map = _fetch_phylopic_silhouettes(
            species, cache_dir, ncbi, name_to_taxid
        )

    manifest = {
        "run_name": args.run_name,
        "species": species,
        "ncbi_db": ncbi_db,
        "output_dir": str(output_dir),
        "cfg": cfg,
        "phylopic_url_map": phylopic_url_map,
        "resolved_species_count": len(species),
        "phylopic_hits": len(phylopic_url_map),
    }

    Path(args.output).write_text(json.dumps(manifest, indent=2))
    print(
        f"[resolve] done — {len(species)} species, "
        f"{len(phylopic_url_map)} PhyloPic silhouettes resolved"
    )
    print(f"[resolve] manifest → {args.output}")


# ── render ────────────────────────────────────────────────────────────────────

def _build_ncbi_tree(species: list[str], ncbi_db: str | None):
    """Build an ete3 topology; return (ete_tree, ncbi, name_to_taxid)."""
    ncbi = _init_ncbi(ncbi_db)

    name_to_taxid = ncbi.get_name_translator(species)
    missing = [s for s in species if s not in name_to_taxid]
    if missing:
        print(f"  [render] warning: could not resolve NCBI taxids for: {missing}")

    taxids = [name_to_taxid[s][0] for s in species if s in name_to_taxid]
    tree = ncbi.get_topology(taxids)

    taxid_to_name = ncbi.get_taxid_translator([leaf.name for leaf in tree.iter_leaves()])
    for leaf in tree.iter_leaves():
        leaf.name = taxid_to_name[int(leaf.name)]

    return tree, ncbi, name_to_taxid


def _assign_tip_angles(ete_tree) -> dict:
    """
    Return {node_name: angle_radians} for every leaf, evenly spaced over
    the top semicircle (pi → 0, i.e. left → right).
    """
    import numpy as np
    leaves = ete_tree.get_leaves()
    n = len(leaves)
    angles = {}
    for i, leaf in enumerate(leaves):
        # map index 0..n-1 to pi..0 (top half, counter-clockwise)
        angles[leaf.name] = np.pi * (1.0 - i / max(n - 1, 1))
    return angles


def _node_depth(node) -> float:
    """Distance from root to node (number of edges; branch lengths all =1)."""
    d = 0
    n = node
    while n.up:
        d += 1
        n = n.up
    return d


def _assign_node_positions(ete_tree, tip_angles: dict) -> dict:
    """
    Return {node_name_or_id: (r, theta)} in polar coords.
    r = depth from root (0 = root).  For internal nodes, theta = mean of children.
    """
    import numpy as np

    # max depth for normalisation
    max_depth = max(_node_depth(l) for l in ete_tree.get_leaves())

    positions: dict = {}

    def _visit(node):
        depth = _node_depth(node)
        r = depth / max(max_depth, 1)
        if node.is_leaf():
            theta = tip_angles[node.name]
        else:
            child_thetas = []
            for child in node.children:
                _visit(child)
                child_thetas.append(positions[id(child)][1])
            theta = float(np.mean(child_thetas))
        positions[id(node)] = (r, theta)
        node._polar = (r, theta)   # cache on node for branch drawing

    _visit(ete_tree)
    return positions


def _polar_to_xy(r, theta):
    import numpy as np
    return r * np.cos(theta), r * np.sin(theta)


def _render_tree(
    ete_tree,
    species: list[str],
    cfg: dict,
    output_dir: Path,
    local_png_map: dict[str, str],
) -> dict[str, str]:
    """Render a semicircular cladogram with PhyloPic silhouettes at tips."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import numpy as np
    from PIL import Image

    layout_cfg = cfg.get("layout") or {}
    figsize = tuple(layout_cfg.get("figsize", [14, 8]))
    annotations_cfg = cfg.get("annotations") or {}
    sil_size = annotations_cfg.get("silhouette_size", 0.06)
    sil_alpha = annotations_cfg.get("silhouette_alpha", 0.85)
    outputs_cfg = cfg.get("outputs") or {}
    analysis_id = cfg.get("analysis_id", "phylogeny")
    written: dict[str, str] = {}

    # ── Newick ──────────────────────────────────────────────────────────────
    newick_str = ete_tree.write(format=1)
    if outputs_cfg.get("newick", True):
        newick_path = output_dir / f"{analysis_id}.newick"
        newick_path.write_text(newick_str)
        written["newick"] = str(newick_path)
        print(f"  [render] wrote Newick → {newick_path}")

    # ── Compute layout ───────────────────────────────────────────────────────
    tip_angles = _assign_tip_angles(ete_tree)
    _assign_node_positions(ete_tree, tip_angles)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    line_kw = dict(color="#333333", linewidth=0.8, solid_capstyle="round")

    # ── Draw branches (parent→child as radial + arc segments) ───────────────
    def _draw_branch(parent, child):
        pr, pt = parent._polar
        cr, ct = child._polar
        # radial segment: parent radius → child radius at parent angle
        px0, py0 = _polar_to_xy(pr, pt)
        px1, py1 = _polar_to_xy(cr, pt)
        ax.plot([px0, px1], [py0, py1], **line_kw)
        # arc segment: sweep from parent theta to child theta at child radius
        t0, t1 = sorted([pt, ct])
        thetas = np.linspace(t0, t1, 60)
        ax.plot(cr * np.cos(thetas), cr * np.sin(thetas), **line_kw)

    def _traverse(node):
        for child in node.children:
            _draw_branch(node, child)
            _traverse(child)

    _traverse(ete_tree)

    # ── Silhouette sizing: compute uniform target pixel width ────────────────
    tip_r = 1.0
    n_leaves = len(ete_tree.get_leaves())
    # arc gap between adjacent tips in radians; silhouettes get 70% of that arc
    arc_gap_rad = np.pi / max(n_leaves - 1, 1)

    # We'll set axis limits after placing everything; use a placeholder for now
    # to get a rough figure-width-to-data-unit ratio.
    # Silhouettes are placed on a ring just outside the label text.
    label_r = tip_r + 0.07
    sil_r   = tip_r + 0.38   # centre of silhouette ring — push out to reduce apex crowding

    # ── Load all silhouette images up front to know their aspect ratios ───────
    loaded: dict[str, tuple] = {}   # species → (img_arr, h, w)
    for leaf in ete_tree.get_leaves():
        png_path = local_png_map.get(leaf.name)
        if not png_path:
            continue
        try:
            img = Image.open(png_path).convert("RGBA")
            arr = np.array(img)
            loaded[leaf.name] = (arr, arr.shape[0], arr.shape[1])
        except Exception as exc:
            print(f"  [render] warning: could not load '{leaf.name}': {exc}")

    # ── Tip labels ───────────────────────────────────────────────────────────
    for leaf in ete_tree.get_leaves():
        theta = tip_angles[leaf.name]
        lx, ly = _polar_to_xy(label_r, theta)
        # right half: text runs left-to-right away from centre
        # left half: flip so text still reads outward
        if np.cos(theta) >= 0:
            ha = "left"
            rotation = np.degrees(theta)
        else:
            ha = "right"
            rotation = np.degrees(theta) + 180
        ax.text(
            lx, ly,
            leaf.name,
            ha=ha, va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=5.5,
            fontstyle="italic",
            color="#222222",
        )

    # ── PhyloPic silhouettes on outer ring ───────────────────────────────────
    if loaded:
        # After drawing labels we know the approximate data extent.
        # Use figure dimensions to convert sil_size (fraction of fig width) → pixels.
        fig_w_px = fig.get_figwidth() * fig.dpi

        for leaf in ete_tree.get_leaves():
            if leaf.name not in loaded:
                continue
            arr, h, w = loaded[leaf.name]
            theta = tip_angles[leaf.name]
            sx, sy = _polar_to_xy(sil_r, theta)
            zoom = (sil_size * fig_w_px) / w
            oi = OffsetImage(arr, zoom=zoom, alpha=sil_alpha)
            ab = AnnotationBbox(
                oi, (sx, sy),
                frameon=False,
                box_alignment=(0.5, 0.5),
                xycoords="data",
            )
            ax.add_artist(ab)

    # ── Axis limits ───────────────────────────────────────────────────────────
    # Give enough room for silhouettes + labels on all sides.
    # The semicircle spans x: [-sil_r-pad, sil_r+pad], y: [-label_r, sil_r+pad]
    pad = sil_r + 0.55
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-0.12, pad)   # tight bottom — semicircle base is near y=0

    plt.tight_layout(pad=0.3)

    if outputs_cfg.get("png", True):
        png_path = output_dir / f"{analysis_id}.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        written["png"] = str(png_path)
        print(f"  [render] wrote PNG → {png_path}")

    if outputs_cfg.get("svg", True):
        svg_path = output_dir / f"{analysis_id}.svg"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        written["svg"] = str(svg_path)
        print(f"  [render] wrote SVG → {svg_path}")

    plt.close(fig)
    return written


def cmd_render(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    cfg = _run_cfg(config, args.run_name)

    species = cfg.get("species") or []
    output_dir = Path(cfg.get("output_dir", ".")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations_cfg = cfg.get("annotations") or {}
    cache_dir = output_dir / "phylopic_cache"

    print(f"[render] building NCBI topology for {len(species)} species …")
    ete_tree, ncbi, name_to_taxid = _build_ncbi_tree(species, cfg.get("ncbi_db"))

    # Prefer URL map from the resolve manifest; fall back to fetching now
    manifest_path = Path(f"/tmp/{args.run_name}_phylo_manifest.json")
    phylopic_url_map: dict[str, str] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        phylopic_url_map = manifest.get("phylopic_url_map") or {}
    elif annotations_cfg.get("phylopic", False):
        phylopic_url_map = _fetch_phylopic_silhouettes(
            species, cache_dir, ncbi, name_to_taxid
        )

    # Download remote PNGs to local cache with human-readable filenames
    local_png_map: dict[str, str] = {}
    if phylopic_url_map:
        print(f"[render] downloading {len(phylopic_url_map)} PhyloPic silhouettes …")
        common_names = _common_name_map(name_to_taxid, cfg.get("ncbi_db"))
        local_png_map = _download_silhouettes(phylopic_url_map, cache_dir, common_names)

    print("[render] rendering …")
    written = _render_tree(ete_tree, species, cfg, output_dir, local_png_map)

    summary = {
        "run_name": args.run_name,
        "species_count": len(species),
        "phylopic_silhouettes": len(local_png_map),
        "outputs": written,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(f"[render] done — summary → {args.output}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="make_phylogeny workflow")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--run-name", required=True, help="Key in make_phylogeny_runs")

    sub = parser.add_subparsers(dest="command", required=True)

    p_resolve = sub.add_parser("resolve", help="Validate config and fetch PhyloPic silhouettes")
    p_resolve.add_argument("--output", required=True, help="Path for JSON manifest")

    p_render = sub.add_parser("render", help="Build tree and render outputs")
    p_render.add_argument("--output", required=True, help="Path for JSON summary")

    args = parser.parse_args()
    if args.command == "resolve":
        cmd_resolve(args)
    elif args.command == "render":
        cmd_render(args)


if __name__ == "__main__":
    main()
