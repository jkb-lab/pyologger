import re
from typing import Dict, List, Optional, Sequence, Tuple


# Warm -> cool, aligned to project preference.
HR_RAINBOW_MUTED = [
    "#c93a3a",  # red
    "#df7f2d",  # orange
    "#d8b63c",  # yellow
    "#5ea85e",  # green
    "#2f9c95",  # teal
    "#4f7ecf",  # blue
    "#8b63c7",  # purple
]

# Warm -> cool anchors (ColorBrewer Spectral 11).
SPECTRAL_WARM_TO_COOL_11 = [
    "#9e0142",
    "#d53e4f",
    "#f46d43",
    "#fdae61",
    "#fee08b",
    "#ffffbf",
    "#e6f598",
    "#abdda4",
    "#66c2a5",
    "#3288bd",
    "#5e4fa2",
]

_HR_SUBSET_INDEXES_WARM_TO_COOL = {
    2: [1, 5],              # orange, blue
    3: [1, 3, 6],           # orange, green, purple
    4: [0, 2, 4, 6],        # red, yellow, teal, purple
    5: [0, 1, 3, 5, 6],     # red, orange, green, blue, purple
    6: [0, 1, 2, 4, 5, 6],  # red, orange, yellow, teal, blue, purple
    7: [0, 1, 2, 3, 4, 5, 6],
}


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    color = str(color).strip().lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {color}")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def _sample_palette_linear(anchors: Sequence[str], n: int) -> List[str]:
    if n <= 0:
        return []
    if n == 1:
        return [anchors[len(anchors) // 2]]

    anchor_rgb = [_hex_to_rgb(c) for c in anchors]
    max_idx = len(anchor_rgb) - 1
    out = []
    for i in range(n):
        pos = (i * max_idx) / float(n - 1)
        lo = int(pos)
        hi = min(lo + 1, max_idx)
        frac = pos - lo
        if hi == lo:
            rgb = anchor_rgb[lo]
        else:
            rgb = (
                int(round(anchor_rgb[lo][0] + frac * (anchor_rgb[hi][0] - anchor_rgb[lo][0]))),
                int(round(anchor_rgb[lo][1] + frac * (anchor_rgb[hi][1] - anchor_rgb[lo][1]))),
                int(round(anchor_rgb[lo][2] + frac * (anchor_rgb[hi][2] - anchor_rgb[lo][2]))),
            )
        out.append(_rgb_to_hex(rgb))
    return out


def parse_cluster_rank_from_key(key: str) -> Optional[int]:
    text = str(key).strip()

    if text.isdigit():
        return int(text)

    m = re.match(r"^cluster[_-]?(\d+)$", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    m = re.match(r"^(\d+)(?:[_-]|$)", text)
    if m:
        return int(m.group(1))

    m = re.search(r"(?:^|[_-])cluster[_-]?(\d+)(?:[_-]|$)", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    return None


def looks_like_ordered_cluster_key(key: str) -> bool:
    return parse_cluster_rank_from_key(key) is not None


def ordered_cluster_palette_low_to_high(n: int) -> List[str]:
    if n <= 0:
        return []
    if n <= 7:
        idx = _HR_SUBSET_INDEXES_WARM_TO_COOL[n]
        warm_to_cool = [HR_RAINBOW_MUTED[i] for i in idx]
    else:
        warm_to_cool = _sample_palette_linear(SPECTRAL_WARM_TO_COOL_11, n)
    # low->high should be cool->warm
    return list(reversed(warm_to_cool))


def build_ordered_cluster_color_map(keys: Sequence[str]) -> Dict[str, str]:
    unique_keys = sorted({str(k) for k in keys if k is not None})
    if not unique_keys:
        return {}

    ranked: List[Tuple[int, str]] = []
    unranked: List[str] = []
    for k in unique_keys:
        r = parse_cluster_rank_from_key(k)
        if r is None:
            unranked.append(k)
        else:
            ranked.append((r, k))

    ranked.sort(key=lambda x: (x[0], x[1]))
    if not ranked:
        # Keep behavior deterministic even if the caller passed non-ranked labels only.
        colors = ordered_cluster_palette_low_to_high(len(unranked))
        return {k: colors[i] for i, k in enumerate(unranked)}

    colors_ranked = ordered_cluster_palette_low_to_high(len(ranked))
    out: Dict[str, str] = {k: colors_ranked[i] for i, (_, k) in enumerate(ranked)}

    # Non-ranked labels are mapped after ranked labels deterministically.
    if unranked:
        colors_unranked = ordered_cluster_palette_low_to_high(len(unranked))
        for i, k in enumerate(unranked):
            out[k] = colors_unranked[i]

    return out
