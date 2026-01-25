from pathlib import Path
import random
import shutil

RAW = Path(__file__).resolve().parents[1] / "data" / "deepdataflow_raw"

GRAPH_ROOT = RAW / "graphs_20.06.01" / "dataflow" / "graphs"
LABEL_ROOT = RAW / "labels_reachability_20.06.01" / "dataflow" / "labels" / "reachability"

OUT = Path(__file__).resolve().parents[1] / "data" / "subset_reachability"

SEED = 123
N_TRAIN = 800
N_VAL = 100
N_TEST = 100

GRAPH_SUFFIX = ".ProgramGraph.pb"
LABEL_SUFFIX = ".ProgramGraphFeaturesList.pb"

def graph_key(p: Path) -> str:
    name = p.name
    return name[:-len(GRAPH_SUFFIX)] if name.endswith(GRAPH_SUFFIX) else p.stem

def label_key(p: Path) -> str:
    name = p.name
    return name[:-len(LABEL_SUFFIX)] if name.endswith(LABEL_SUFFIX) else p.stem

def main():
    random.seed(SEED)

    assert GRAPH_ROOT.exists(), f"GRAPH_ROOT not found: {GRAPH_ROOT}"
    assert LABEL_ROOT.exists(), f"LABEL_ROOT not found: {LABEL_ROOT}"

    graphs = list(GRAPH_ROOT.rglob(f"*{GRAPH_SUFFIX}"))
    labels = list(LABEL_ROOT.rglob(f"*{LABEL_SUFFIX}"))

    print("Graphs:", len(graphs))
    print("Labels:", len(labels))

    gmap = {graph_key(p): p for p in graphs}
    lmap = {label_key(p): p for p in labels}

    keys = sorted(set(gmap.keys()) & set(lmap.keys()))
    print("Paired (graph+label) samples:", len(keys))
    assert len(keys) > 0, "No paired samples found — check suffix/path."

    need = N_TRAIN + N_VAL + N_TEST
    if len(keys) < need:
        print(f"⚠️ Not enough pairs for requested split ({need}). Using all pairs.")
        picked = keys
    else:
        picked = random.sample(keys, need)

    splits = {
        "train": picked[:N_TRAIN],
        "val": picked[N_TRAIN:N_TRAIN + N_VAL],
        "test": picked[N_TRAIN + N_VAL:],
    }

    for split, ks in splits.items():
        gdst = OUT / split / "graphs"
        ldst = OUT / split / "labels"
        gdst.mkdir(parents=True, exist_ok=True)
        ldst.mkdir(parents=True, exist_ok=True)

        for k in ks:
            shutil.copy2(gmap[k], gdst / gmap[k].name)
            shutil.copy2(lmap[k], ldst / lmap[k].name)

    print("✅ Subset written to:", OUT)
    ex = picked[0]
    print("Example key:", ex)
    print(" - graph:", gmap[ex].name)
    print(" - label:", lmap[ex].name)

if __name__ == "__main__":
    main()
