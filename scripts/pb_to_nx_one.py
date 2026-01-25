import sys
from pathlib import Path
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "programl_pb2"))
from programl.proto import program_graph_pb2 as pg_pb2

def load_graph(pb_path: Path) -> pg_pb2.ProgramGraph:
    g = pg_pb2.ProgramGraph()
    g.ParseFromString(pb_path.read_bytes())
    return g

def to_networkx(g: pg_pb2.ProgramGraph) -> nx.DiGraph:
    G = nx.DiGraph()

    # nodes
    for i, n in enumerate(g.node):
        G.add_node(
            i,
            type=int(n.type),
            text=str(n.text),
            function=int(n.function),
            block=int(n.block),
            # features (可先不展開，先確認存在)
            has_features=bool(n.features.feature),  # feature is a map in proto
        )

    # edges
    for e in g.edge:
        G.add_edge(
            int(e.source),
            int(e.target),
            flow=int(e.flow),
            position=int(e.position),
            has_features=bool(e.features.feature),
        )
    return G

def main():
    base = ROOT / "data" / "subset_reachability"
    pb = sorted(base.rglob("*.ProgramGraph.pb"))[0]
    g = load_graph(pb)
    G = to_networkx(g)

    print("PB:", pb)
    print("NX nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())

    # quick checks
    flows = {}
    for _, _, d in G.edges(data=True):
        flows[d["flow"]] = flows.get(d["flow"], 0) + 1
    print("Edge flow counts:", dict(sorted(flows.items())))

    types = {}
    for _, d in G.nodes(data=True):
        types[d["type"]] = types.get(d["type"], 0) + 1
    print("Node type counts:", dict(sorted(types.items())))

if __name__ == "__main__":
    main()
