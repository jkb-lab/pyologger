import sys
from pathlib import Path


# Allow imports when running tests from repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyologger.utils.cluster_colors import (  # noqa: E402
    build_ordered_cluster_color_map,
    ordered_cluster_palette_low_to_high,
    parse_cluster_rank_from_key,
)


def test_parse_cluster_rank_from_key_patterns():
    assert parse_cluster_rank_from_key("3_dynX+dynY+odba+pitch+roll_full_30S_PCA") == 3
    assert parse_cluster_rank_from_key("cluster_3") == 3
    assert parse_cluster_rank_from_key("3") == 3
    assert parse_cluster_rank_from_key("logger_status") is None


def test_palette_for_two_clusters_orange_blue_with_low_to_high_order():
    palette = ordered_cluster_palette_low_to_high(2)
    assert palette == ["#4f7ecf", "#df7f2d"]  # low=cool blue, high=warm orange


def test_palette_for_three_clusters_orange_green_purple_with_low_to_high_order():
    palette = ordered_cluster_palette_low_to_high(3)
    assert palette == ["#8b63c7", "#5ea85e", "#df7f2d"]  # low->high


def test_build_ordered_cluster_color_map_uses_rank_not_lexicographic_order():
    keys = ["10_feature_PCA", "2_feature_PCA", "1_feature_PCA"]
    cmap = build_ordered_cluster_color_map(keys)
    ordered = [cmap["1_feature_PCA"], cmap["2_feature_PCA"], cmap["10_feature_PCA"]]
    assert ordered == ["#8b63c7", "#5ea85e", "#df7f2d"]


def test_palette_for_more_than_seven_clusters_is_spectral_and_warm_cool_ordered():
    keys = [f"cluster_{i}" for i in range(10)]
    cmap = build_ordered_cluster_color_map(keys)
    # low rank cluster_0 should be coolest side of Spectral-like ramp
    assert cmap["cluster_0"] != cmap["cluster_9"]
    assert cmap["cluster_0"].lower() == "#5e4fa2"
    assert cmap["cluster_9"].lower() == "#9e0142"
