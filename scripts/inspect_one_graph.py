import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "programl_pb2"))

from programl.proto import program_graph_pb2 as pg_pb2

def main():
    base = ROOT / "data" / "subset_reachability"
    f = sorted(base.rglob("*.ProgramGraph.pb"))[0]

    g = pg_pb2.ProgramGraph()
    g.ParseFromString(f.read_bytes())

    print("File:", f)
    print("Nodes:", len(g.node))
    print("Edges:", len(g.edge))

    # node + edge type distribution (字段名可能依 proto 版本不同；先做安全檢查)
    node_fields = set(g.node[0].DESCRIPTOR.fields_by_name.keys()) if g.node else set()
    edge_fields = set(g.edge[0].DESCRIPTOR.fields_by_name.keys()) if g.edge else set()
    print("Node fields:", sorted(node_fields))
    print("Edge fields:", sorted(edge_fields))

    # 常見欄位: type / text / function / opcode... 看你這份 proto 實際有哪些
    if "type" in node_fields:
        cnt = Counter([n.type for n in g.node])
        print("Top node.type:", cnt.most_common(10))
    if "flow" in edge_fields:
        cnt = Counter([e.flow for e in g.edge])
        print("Top edge.flow:", cnt.most_common(10))

if __name__ == "__main__":
    main()
