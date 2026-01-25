from pathlib import Path
import random
import re

RAW = Path(__file__).resolve().parents[1] / "data" / "deepdataflow_raw"
GRAPH_ROOT = RAW / "graphs_20.06.01" / "dataflow" / "graphs"
LABEL_ROOT = RAW / "labels_reachability_20.06.01" / "dataflow" / "labels" / "reachability"

random.seed(123)

def graph_key(name: str) -> str:
    # xxx.ProgramGraph.pb -> xxx
    if name.endswith(".ProgramGraph.pb"):
        return name[:-len(".ProgramGraph.pb")]
    return Path(name).stem

# Several candidate label key normalizers (we'll test which matches graphs)
def label_key_v1(name: str) -> str:
    # xxx.ProgramFeaturesList.pb -> xxx
    for suf in [".ProgramFeaturesList.pb", ".ProgramFeatures.pb", ".pb"]:
        if name.endswith(suf):
            return name[:-len(suf)]
    return Path(name).stem

def label_key_v2_strip_reachability_tail(name: str) -> str:
    # xxx.reachability.ProgramFeaturesList.pb -> xxx
    base = label_key_v1(name)
    if base.endswith(".reachability"):
        base = base[:-len(".reachability")]
    return base

def label_key_v3_strip_any_reachability_segment(name: str) -> str:
    # remove ".reachability" segment anywhere at the end-ish
    base = label_key_v1(name)
    base = re.sub(r"\.reachability$", "", base)
    base = re.sub(r"\.reachability\.", ".", base)
    return base

def label_key_v4_take_prefix_before_reachability(name: str) -> str:
    # xxx.reachability.something.pb -> xxx
    base = label_key_v1(name)
    if ".reachability" in base:
        return base.split(".reachability")[0]
    return base

CANDIDATES = [
    ("v1_basic", label_key_v1),
    ("v2_strip_tail_.reachability", label_key_v2_strip_reachability_tail),
    ("v3_strip_segment", label_key_v3_strip_any_reachability_segment),
    ("v4_prefix_before_.reachability", label_key_v4_take_prefix_before_reachability),
]

def main():
    assert GRAPH_ROOT.exists(), f"GRAPH_ROOT not found: {GRAPH_ROOT}"
    assert LABEL_ROOT.exists(), f"LABEL_ROOT not found: {LABEL_ROOT}"

    graphs = list(GRAPH_ROOT.rglob("*.ProgramGraph.pb"))
    labels = list(LABEL_ROOT.rglob("*.pb"))

    print("Graphs:", len(graphs))
    print("Labels:", len(labels))

    # Build graph key set
    gkeys = set()
    for p in graphs:
        gkeys.add(graph_key(p.name))
    print("Unique graph keys:", len(gkeys))

    # Test each candidate label key function: count matches without storing all label keys
    best = None
    for tag, fn in CANDIDATES:
        hit = 0
        example = None
        for p in labels:
            lk = fn(p.name)
            if lk in gkeys:
                hit += 1
                if example is None:
                    example = (lk, p.name)
        print(f"\n[{tag}] matched labels:", hit)
        if example:
            print("  example match key:", example[0])
            print("  label filename:", example[1])
            # find the corresponding graph filename example
            # not exact unique mapping, but we can show one possible
            # (search among a few graphs for speed)
            for gp in graphs[:2000]:
                if graph_key(gp.name) == example[0]:
                    print("  graph filename:", gp.name)
                    break

        if best is None or hit > best[1]:
            best = (tag, hit)

    print("\n=== BEST ===")
    print("Best rule:", best[0], "with matched labels:", best[1])

if __name__ == "__main__":
    main()
