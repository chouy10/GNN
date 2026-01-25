import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "programl_pb2"))

from programl.proto import util_pb2 as util_pb2


def pick_first_label_file():
    base = ROOT / "data" / "subset_reachability"
    files = sorted(base.rglob("*.ProgramGraphFeaturesList.pb"))
    if not files:
        raise SystemExit("No *.ProgramGraphFeaturesList.pb found under data/subset_reachability")
    return files[0]


def get_featurelist_lengths(feature_lists_msg, key: str):
    """
    feature_lists_msg: tensorflow.FeatureLists
    key: feature_list name, e.g. 'data_flow_root_node' or 'data_flow_value'
    Returns: number of entries in the FeatureList (should match #nodes in ProgramGraph)
    """
    fl_map = feature_lists_msg.feature_list
    if key not in fl_map:
        return 0
    return len(fl_map[key].feature)


def get_int64_list(feature):
    # tensorflow.Feature has oneof: bytes_list / float_list / int64_list
    if feature.HasField("int64_list"):
        return list(feature.int64_list.value)
    return []


def get_float_list(feature):
    if feature.HasField("float_list"):
        return list(feature.float_list.value)
    return []


def summarize_one_step(step_msg, step_idx: int, max_show_roots=10):
    nf = step_msg.node_features

    # --- root node (usually int64 list, one value per node) ---
    roots = []
    if "data_flow_root_node" in nf.feature_list:
        feats = nf.feature_list["data_flow_root_node"].feature
        for i, feat in enumerate(feats):
            vals = get_int64_list(feat)
            # common encodings: [0/1] or empty
            if len(vals) > 0 and vals[0] != 0:
                roots.append(i)

    # --- data_flow_value (could be int64 or float depending on dataset) ---
    has_value = 0
    value_kinds = Counter()
    if "data_flow_value" in nf.feature_list:
        feats = nf.feature_list["data_flow_value"].feature
        for feat in feats:
            if feat.HasField("int64_list"):
                value_kinds["int64"] += 1
                vals = get_int64_list(feat)
                if len(vals) > 0:
                    has_value += 1
            elif feat.HasField("float_list"):
                value_kinds["float"] += 1
                vals = get_float_list(feat)
                if len(vals) > 0:
                    has_value += 1
            elif feat.HasField("bytes_list"):
                value_kinds["bytes"] += 1
                # bytes usually means "exists", but keep conservative:
                if len(feat.bytes_list.value) > 0:
                    has_value += 1
            else:
                value_kinds["empty"] += 1

    # node count inferred from featurelist length
    n_nodes_root = get_featurelist_lengths(nf, "data_flow_root_node")
    n_nodes_val = get_featurelist_lengths(nf, "data_flow_value")

    print(f"\n=== Step {step_idx} ===")
    print(f"node_features lengths: root_node={n_nodes_root}, value={n_nodes_val}")
    print(f"root nodes (#nonzero): {len(roots)}  sample: {roots[:max_show_roots]}")
    print(f"data_flow_value: nodes with non-empty value = {has_value}")
    print(f"data_flow_value feature kinds: {dict(value_kinds)}")

    # graph-level features keys
    if step_msg.features and hasattr(step_msg.features, "feature"):
        keys = sorted(step_msg.features.feature.keys())
        print(f"graph-level feature keys: {keys}")
        if "data_flow_step_count" in step_msg.features.feature:
            vals = get_int64_list(step_msg.features.feature["data_flow_step_count"])
            if vals:
                print(f"data_flow_step_count = {vals[0]}")
        if "data_flow_active_node_count" in step_msg.features.feature:
            vals = get_int64_list(step_msg.features.feature["data_flow_active_node_count"])
            if vals:
                print(f"data_flow_active_node_count = {vals[0]}")


def main():
    f = pick_first_label_file()
    y = util_pb2.ProgramGraphFeaturesList()
    y.ParseFromString(f.read_bytes())

    print("Label file:", f)
    print("Num steps (y.graph):", len(y.graph))

    for i, step in enumerate(y.graph):
        summarize_one_step(step, i)


if __name__ == "__main__":
    main()
