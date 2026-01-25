import sys
from pathlib import Path
from typing import Dict, List, Tuple
import random

import torch
from torch.utils.data import Dataset

# ---- project paths ----
ROOT = Path(__file__).resolve().parents[1]
# so we can import programl_pb2 generated code
sys.path.insert(0, str(ROOT / "src" / "programl_pb2"))

from programl.proto import program_graph_pb2 as pg_pb2
from programl.proto import util_pb2 as util_pb2


def _feat_to_int64(feat) -> int:
    # tensorflow.Feature-like
    if feat.HasField("int64_list") and len(feat.int64_list.value) > 0:
        return int(feat.int64_list.value[0])
    if feat.HasField("float_list") and len(feat.float_list.value) > 0:
        return int(feat.float_list.value[0])
    if feat.HasField("bytes_list") and len(feat.bytes_list.value) > 0:
        return 1
    return 0


def read_program_graph(pb_path: Path) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (num_nodes, edge_index[2,E], node_type[N], edge_flow[E])"""
    g = pg_pb2.ProgramGraph()
    g.ParseFromString(pb_path.read_bytes())

    n = len(g.node)
    e = len(g.edge)

    # node type
    node_type = torch.tensor([int(nd.type) for nd in g.node], dtype=torch.long)

    # edges
    src = torch.empty(e, dtype=torch.long)
    dst = torch.empty(e, dtype=torch.long)
    flow = torch.empty(e, dtype=torch.long)
    for i, ed in enumerate(g.edge):
        src[i] = int(ed.source)
        dst[i] = int(ed.target)
        flow[i] = int(ed.flow)

    edge_index = torch.stack([src, dst], dim=0)  # [2, E]
    return n, edge_index, node_type, flow


def read_label_steps(label_pb_path: Path) -> List[Tuple[int, torch.Tensor, int, int]]:
    """
    For each step:
      - root_idx (int)
      - y_node [N] 0/1
      - step_count (int)   data_flow_step_count
      - active_count (int) data_flow_active_node_count
    """
    y = util_pb2.ProgramGraphFeaturesList()
    y.ParseFromString(label_pb_path.read_bytes())

    steps = []
    for step in y.graph:
        nf = step.node_features

        # root_idx: find the one index with nonzero
        roots = nf.feature_list["data_flow_root_node"].feature
        root_idx = None
        for i, feat in enumerate(roots):
            if _feat_to_int64(feat) != 0:
                root_idx = i
                break
        if root_idx is None:
            # should not happen, but safe fallback
            root_idx = 0

        # node labels
        vals = nf.feature_list["data_flow_value"].feature
        y_node = torch.tensor([_feat_to_int64(f) for f in vals], dtype=torch.long)  # [N]

        # graph-level stats
        sc = _feat_to_int64(step.features.feature["data_flow_step_count"]) if "data_flow_step_count" in step.features.feature else 0
        ac = _feat_to_int64(step.features.feature["data_flow_active_node_count"]) if "data_flow_active_node_count" in step.features.feature else int(y_node.sum().item())

        steps.append((root_idx, y_node, sc, ac))

    return steps


def _key_from_graph_filename(name: str) -> str:
    # "github.27966.c.ProgramGraph.pb" -> "github.27966.c"
    if name.endswith(".ProgramGraph.pb"):
        return name[: -len(".ProgramGraph.pb")]
    return name


class ReachabilityPyGDataset(Dataset):
    """
    Returns a dict usable for PyTorch Geometric training:
      {
        "edge_index": LongTensor [2,E],
        "node_type": LongTensor [N],
        "edge_flow": LongTensor [E],
        "root_idx": LongTensor [],
        "y": LongTensor [N],  # 0/1
        "step_count": LongTensor [],
        "active_count": LongTensor [],
        "key": str,
        "split": str,
        "step_id": int,
      }
    """

    def __init__(self, subset_dir: Path, split: str, max_graphs: int | None = None, seed: int = 0):
        self.subset_dir = Path(subset_dir)
        self.split = split

        graphs_dir = self.subset_dir / split / "graphs"
        labels_dir = self.subset_dir / split / "labels"

        graph_files = sorted(graphs_dir.glob("*.ProgramGraph.pb"))
        if max_graphs is not None:
            graph_files = graph_files[:max_graphs]

        # map key -> label file
        label_map: Dict[str, Path] = {}
        for lf in labels_dir.glob("*.ProgramGraphFeaturesList.pb"):
            k = lf.name[: -len(".ProgramGraphFeaturesList.pb")]
            label_map[k] = lf

        # build samples = (graph_pb, label_pb, key, step_id)
        samples = []
        for gf in graph_files:
            key = _key_from_graph_filename(gf.name)
            lf = label_map.get(key)
            if lf is None:
                continue
            # each label contains multiple steps
            steps = read_label_steps(lf)
            for step_id in range(len(steps)):
                samples.append((gf, lf, key, step_id))

        # shuffle deterministically (optional)
        random.Random(seed).shuffle(samples)
        self.samples = samples

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found under {subset_dir} split={split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        gf, lf, key, step_id = self.samples[idx]

        n, edge_index, node_type, edge_flow = read_program_graph(gf)
        steps = read_label_steps(lf)
        root_idx, y_node, step_count, active_count = steps[step_id]

        # sanity: ensure y length matches node count
        if y_node.numel() != n:
            raise RuntimeError(f"Mismatch N: graph has {n} nodes but label has {y_node.numel()} nodes for key={key}")

        return {
            "edge_index": edge_index,
            "node_type": node_type,
            "edge_flow": edge_flow,
            "root_idx": torch.tensor(root_idx, dtype=torch.long),
            "y": y_node,
            "step_count": torch.tensor(step_count, dtype=torch.long),
            "active_count": torch.tensor(active_count, dtype=torch.long),
            "key": key,
            "split": self.split,
            "step_id": step_id,
        }


def main():
    subset = ROOT / "data" / "subset_reachability"
    ds = ReachabilityPyGDataset(subset, split="test", max_graphs=3)
    print("len(ds) =", len(ds))
    ex = ds[0]
    print("key:", ex["key"], "step:", ex["step_id"])
    print("N:", ex["node_type"].numel(), "E:", ex["edge_index"].size(1))
    print("root:", int(ex["root_idx"]))
    print("y sum:", int(ex["y"].sum().item()), "active_count:", int(ex["active_count"]))


if __name__ == "__main__":
    main()
