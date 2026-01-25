from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "deepdataflow_raw"

def main():
    # Reachability labels are usually stored as ProgramFeaturesList.pb
    feats = list(ROOT.rglob("*.ProgramFeaturesList.pb"))
    print("ROOT:", ROOT)
    print("Found ProgramFeaturesList.pb:", len(feats))

    # Show a few examples
    for p in feats[:20]:
        print(" -", p.relative_to(ROOT))

    # Also show directories that look like labels
    print("\nPossible label dirs (name contains 'label' or 'reach'):")
    for p in sorted([p for p in ROOT.rglob("*") if p.is_dir()]):
        name = p.name.lower()
        if "label" in name or "reach" in name:
            print(" -", p.relative_to(ROOT))

if __name__ == "__main__":
    main()
