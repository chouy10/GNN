from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1] / "data"

def main():
    print("ROOT =", ROOT)
    assert ROOT.exists(), f"{ROOT} not found"

    print("\n=== Level-1 directories ===")
    for p in sorted(ROOT.iterdir()):
        print(" -", p.name)

    print("\n=== Search for .pb files (first 20) ===")
    pb_files = list(ROOT.rglob("*.pb"))
    for p in pb_files[:20]:
        print(" -", p.relative_to(ROOT))
    print(f"Total .pb files: {len(pb_files)}")

    print("\n=== File extension histogram ===")
    ext = Counter(p.suffix for p in ROOT.rglob("*") if p.is_file())
    for k, v in ext.most_common(15):
        print(f"{k or '<no ext>'}: {v}")

if __name__ == "__main__":
    main()
