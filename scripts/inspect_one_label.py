import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 讓 `import programl...` 能找到你產生的 pb2（src/programl_pb2/programl/...）
sys.path.insert(0, str(ROOT / "src" / "programl_pb2"))

from programl.proto import util_pb2 as util_pb2


def field_names(msg):
    return [f.name for f in msg.DESCRIPTOR.fields]


def summarize_feature_map(features_msg, max_items=10):
    """
    For tensorflow.Features:
      has map<string, Feature> feature
    """
    if not hasattr(features_msg, "feature"):
        return "(no .feature map)"
    keys_sorted = sorted(list(features_msg.feature.keys()))
    shown = keys_sorted[:max_items]
    return f"keys[{len(keys_sorted)}]: {shown}"


def summarize_featurelists_map(fl_msg, max_items=30):
    """
    For tensorflow.FeatureLists:
      has map<string, FeatureList> feature_list
    """
    if not hasattr(fl_msg, "feature_list"):
        return "(no .feature_list map)"
    keys_sorted = sorted(list(fl_msg.feature_list.keys()))
    shown = keys_sorted[:max_items]
    return f"keys[{len(keys_sorted)}]: {shown}"


def describe_message(msg, indent=""):
    print(indent + f"{msg.__class__.__name__} fields: {field_names(msg)}")


def main():
    base = ROOT / "data" / "subset_reachability"
    label_files = sorted(base.rglob("*.ProgramGraphFeaturesList.pb"))
    if not label_files:
        raise SystemExit("No *.ProgramGraphFeaturesList.pb found under data/subset_reachability")

    f = label_files[0]
    y = util_pb2.ProgramGraphFeaturesList()
    y.ParseFromString(f.read_bytes())

    print("Label file:", f)
    describe_message(y, indent="")

    # scan top-level fields
    for name in field_names(y):
        val = getattr(y, name)

        # repeated field? (protobuf repeated behaves like sequence)
        if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
            try:
                ln = len(val)
            except TypeError:
                ln = None

            if ln is None:
                continue

            print(f"\nTop-level repeated field: y.{name}  len={ln}")
            if ln <= 0:
                continue

            item0 = val[0]
            describe_message(item0, indent="  ")

            # inspect one level deeper
            for sub in field_names(item0):
                subv = getattr(item0, sub)

                # repeated subfield?
                if hasattr(subv, "__len__") and not isinstance(subv, (str, bytes)):
                    try:
                        subln = len(subv)
                    except TypeError:
                        subln = None
                    if subln is not None:
                        print(f"  - {sub}: len={subln}")
                        if subln > 0:
                            item00 = subv[0]
                            if hasattr(item00, "DESCRIPTOR"):
                                describe_message(item00, indent="    ")
                            else:
                                print("    first item type:", type(item00))
                    continue

                # scalar / message field
                if hasattr(subv, "DESCRIPTOR"):
                    cls = subv.__class__.__name__
                    if cls == "FeatureLists":
                        print(f"  - {sub}: FeatureLists  {summarize_featurelists_map(subv)}")
                    elif cls == "Features":
                        print(f"  - {sub}: Features  {summarize_feature_map(subv)}")
                    else:
                        print(f"  - {sub}: {cls}")
                else:
                    print(f"  - {sub}: {subv}")

    # quick: if it has .graph and its elements have .features
    if hasattr(y, "graph") and len(y.graph) > 0:
        g0 = y.graph[0]
        if hasattr(g0, "features"):
            print("\n[quick] y.graph[0].features:", summarize_feature_map(g0.features))

        # also show FeatureLists keys quickly (common ones)
        for fld in ("node_features", "edge_features", "function_features", "module_features"):
            if hasattr(g0, fld):
                fl = getattr(g0, fld)
                if fl.__class__.__name__ == "FeatureLists":
                    print(f"[quick] y.graph[0].{fld}:", summarize_featurelists_map(fl))


if __name__ == "__main__":
    main()
