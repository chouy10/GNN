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


def feat_to_int(feat):
    # Most of your data_flow_value are int64_list with exactly one int
    if feat.HasField("int64_list") and len(feat.int64_list.value) > 0:
        return int(feat.int64_list.value[0])
    # Fallbacks (rare)
    if feat.HasField("float_list") and len(feat.float_list.value) > 0:
        return int(feat.float_list.value[0])
    if feat.HasField("bytes_list") and len(feat.bytes_list.value) > 0:
        return 1
    return 0


def main():
    f = pick_first_label_file()
    y = util_pb2.ProgramGraphFeaturesList()
    y.ParseFromString(f.read_bytes())

    print("Label file:", f)
    print("Num steps:", len(y.graph))

    for si, step in enumerate(y.graph):
        nf = step.node_features

        # roots
        roots = []
        roots_feats = nf.feature_list["data_flow_root_node"].feature
        for i, feat in enumerate(roots_feats):
            v = feat_to_int(feat)
            if v != 0:
                roots.append(i)

        # value histogram
        vals_feats = nf.feature_list["data_flow_value"].feature
        vals = [feat_to_int(feat) for feat in vals_feats]
        hist = Counter(vals)

        # quick active ratio if it's 0/1
        active = hist.get(1, 0)
        total = len(vals)

        print(f"\n=== Step {si} ===")
        print("root nodes:", roots)
        print("value hist:", dict(hist))
        print(f"active(==1): {active}/{total} = {active/total:.3f}")

        # graph-level
        if "data_flow_step_count" in step.features.feature:
            sc = feat_to_int(step.features.feature["data_flow_step_count"])
            print("data_flow_step_count:", sc)
        if "data_flow_active_node_count" in step.features.feature:
            ac = feat_to_int(step.features.feature["data_flow_active_node_count"])
            print("data_flow_active_node_count:", ac)

        # sanity: compare label active_node_count vs hist
        if "data_flow_active_node_count" in step.features.feature:
            ac = feat_to_int(step.features.feature["data_flow_active_node_count"])
            print("sanity (hist==1 vs active_node_count):", active, "vs", ac)


if __name__ == "__main__":
    main()
