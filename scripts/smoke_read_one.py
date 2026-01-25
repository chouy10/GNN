import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT / "src" / "programl_pb2"))  # <-- 讓 programl 可被 import

from programl.proto import program_graph_pb2 as pg_pb2

base = ROOT / "data" / "subset_reachability"
pb_files = list(base.rglob("*.ProgramGraph.pb"))
print("Found ProgramGraph.pb:", len(pb_files))
if not pb_files:
    raise SystemExit("No *.ProgramGraph.pb found. Check data/subset_reachability path.")

f = pb_files[0]
g = pg_pb2.ProgramGraph()
g.ParseFromString(f.read_bytes())

print("Parsed OK")
print("File:", f)
print("Nodes:", len(g.node))
print("Edges:", len(g.edge))
