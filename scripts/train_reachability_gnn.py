# train_reachability_gnn.py (with visible predict + CSV/DOT/PNG export)
# - Fixes import sys usage
# - Adds --out_png + --max_nodes to draw a PNG directly (networkx + matplotlib)
# - Keeps existing --out_dot (GraphViz) and --out_csv
#
# Usage:
#   Train:
#     python .\scripts\train_reachability_gnn.py --mode train
#
#   Predict (print + save PNG):
#     python .\scripts\train_reachability_gnn.py --mode predict --split test --index 0 --out_png outputs\pred0.png
#
#   Predict (save DOT for GraphViz):
#     python .\scripts\train_reachability_gnn.py --mode predict --split test --index 0 --out_dot outputs\pred0.dot
#     dot -Tpng outputs\pred0.dot -o outputs\pred0.png

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# For PNG drawing (optional but requested)
import matplotlib.pyplot as plt
import networkx as nx

# ---- project paths ----
ROOT = Path(__file__).resolve().parents[1]

# ✅ Make project root importable so "scripts.*" works
_root_str = str(ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

# ✅ Keep your generated ProGraML protobufs importable too (if any script needs it)
_prog_pb2 = str(ROOT / "src" / "programl_pb2")
if _prog_pb2 not in sys.path:
    sys.path.insert(0, _prog_pb2)

from scripts.dataset_pyg_reachability import ReachabilityPyGDataset


def pick_device() -> str:
    """
    torch.cuda.is_available() may be True even when the installed PyTorch CUDA build
    doesn't support this GPU's compute capability (e.g., sm_120).
    So we try a tiny CUDA op to confirm kernels can run; otherwise fall back to CPU.
    """
    if not torch.cuda.is_available():
        return "cpu"
    try:
        _ = torch.empty(1, device="cuda") + 1  # force a CUDA kernel
        return "cuda"
    except Exception as e:
        print(f"[WARN] CUDA available but not runnable ({type(e).__name__}: {e}). Falling back to CPU.")
        return "cpu"


# -----------------------------
# Collate -> make a single big graph batch
# -----------------------------
def collate_graph_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    node_offsets = []
    total_nodes = 0
    for b in batch:
        node_offsets.append(total_nodes)
        total_nodes += b["node_type"].numel()

    node_type = torch.cat([b["node_type"] for b in batch], dim=0)  # [sumN]
    y = torch.cat([b["y"] for b in batch], dim=0).float()          # [sumN]

    root_flag = torch.zeros(total_nodes, dtype=torch.float32)
    for i, b in enumerate(batch):
        off = node_offsets[i]
        root_flag[off + int(b["root_idx"])] = 1.0

    edge_index_list = []
    edge_flow_list = []
    for i, b in enumerate(batch):
        off = node_offsets[i]
        ei = b["edge_index"] + off
        edge_index_list.append(ei)
        edge_flow_list.append(b["edge_flow"])
    edge_index = torch.cat(edge_index_list, dim=1)  # [2, sumE]
    edge_flow = torch.cat(edge_flow_list, dim=0)    # [sumE]

    return {
        "edge_index": edge_index,
        "node_type": node_type,
        "root_flag": root_flag,
        "edge_flow": edge_flow,
        "y": y,
        "meta": [(b.get("key", None), b.get("step_id", None), b.get("split", None)) for b in batch],
        "num_graphs": len(batch),
        "num_nodes": total_nodes,
        "num_edges": edge_flow.numel(),
    }


# -----------------------------
# Simple Edge-aware GraphSAGE
# -----------------------------
class EdgeSAGEConv(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.w_self = nn.Linear(hidden, hidden, bias=True)
        self.w_nei = nn.Linear(hidden, hidden, bias=False)
        self.w_edge = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        msg = self.w_nei(x[src]) + self.w_edge(edge_attr)  # [E,H]

        N = x.size(0)
        out = torch.zeros((N, x.size(1)), device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg)

        deg = torch.zeros((N,), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones((dst.numel(),), device=x.device, dtype=x.dtype))
        deg = deg.clamp_min(1.0).unsqueeze(1)
        out = out / deg

        return F.relu(self.w_self(x) + out)


class ReachabilityGNN(nn.Module):
    def __init__(self, num_node_types: int, num_edge_flows: int, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.node_emb = nn.Embedding(num_node_types, hidden)
        self.edge_emb = nn.Embedding(num_edge_flows, hidden)
        self.root_proj = nn.Linear(1, hidden, bias=False)

        self.convs = nn.ModuleList([EdgeSAGEConv(hidden) for _ in range(layers)])
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_type: torch.Tensor, root_flag: torch.Tensor, edge_index: torch.Tensor, edge_flow: torch.Tensor):
        x = self.node_emb(node_type)
        x = x + self.root_proj(root_flag.unsqueeze(1))
        e = self.edge_emb(edge_flow)

        for conv in self.convs:
            x = conv(x, edge_index, e)

        logits = self.mlp(x).squeeze(1)
        return logits


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def compute_metrics(logits: torch.Tensor, y: torch.Tensor, thresh: float = 0.0) -> Dict[str, float]:
    pred = (logits >= thresh).to(torch.long)
    y_i = y.to(torch.long)

    tp = int(((pred == 1) & (y_i == 1)).sum().item())
    tn = int(((pred == 0) & (y_i == 0)).sum().item())
    fp = int(((pred == 1) & (y_i == 0)).sum().item())
    fn = int(((pred == 0) & (y_i == 1)).sum().item())

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)
    iou = tp / max(1, tp + fp + fn)

    return {"acc": acc, "f1": f1, "iou": iou, "tp": tp, "fp": fp, "fn": fn}


def estimate_type_counts(subset_dir: Path) -> Dict[str, int]:
    ds = ReachabilityPyGDataset(subset_dir, split="train", max_graphs=200, seed=0)
    if len(ds) == 0:
        raise RuntimeError(f"Empty train split under {subset_dir}. Cannot estimate type counts.")

    max_node_type = 0
    max_edge_flow = 0
    for i in range(min(len(ds), 500)):
        ex = ds[i]
        max_node_type = max(max_node_type, int(ex["node_type"].max().item()))
        max_edge_flow = max(max_edge_flow, int(ex["edge_flow"].max().item()))
    return {"num_node_types": max_node_type + 1, "num_edge_flows": max_edge_flow + 1}


# -----------------------------
# "Visible application" helpers
# -----------------------------
def _get_meta(ex: Dict[str, Any]) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    return (ex.get("key", None), ex.get("step_id", None), ex.get("split", None))


@torch.no_grad()
def predict_one(
    model: ReachabilityGNN,
    ex: Dict[str, Any],
    device: str,
    prob_thresh: float = 0.5,
) -> Dict[str, Any]:
    model.eval()

    node_type = ex["node_type"].to(device)
    edge_index = ex["edge_index"].to(device)
    edge_flow = ex["edge_flow"].to(device)
    y = ex["y"].float().to(device)
    root_idx = int(ex["root_idx"])

    root_flag = torch.zeros(node_type.numel(), device=device, dtype=torch.float32)
    root_flag[root_idx] = 1.0

    logits = model(node_type, root_flag, edge_index, edge_flow)
    prob = torch.sigmoid(logits)

    pred = (prob >= prob_thresh).to(torch.long)
    y_i = y.to(torch.long)

    # metrics at prob_thresh
    tp = int(((pred == 1) & (y_i == 1)).sum().item())
    fp = int(((pred == 1) & (y_i == 0)).sum().item())
    fn = int(((pred == 0) & (y_i == 1)).sum().item())
    iou = tp / max(1, tp + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)

    # reference metrics at logits>=0 (prob>=0.5)
    m_ref = compute_metrics(logits, y, thresh=0.0)

    pred_idx = pred.nonzero(as_tuple=False).squeeze(1).detach().cpu().tolist()
    true_idx = y_i.nonzero(as_tuple=False).squeeze(1).detach().cpu().tolist()

    topk = min(10, prob.numel())
    top_prob, top_i = torch.topk(prob, k=topk)
    top_list = [(int(i.item()), float(p.item())) for p, i in zip(top_prob.detach().cpu(), top_i.detach().cpu())]

    return {
        "root_idx": root_idx,
        "num_nodes": int(node_type.numel()),
        "num_edges": int(edge_flow.numel()),
        "prob_thresh": prob_thresh,
        "metrics_prob_thresh": {"iou": iou, "f1": f1, "prec": prec, "rec": rec, "tp": tp, "fp": fp, "fn": fn},
        "metrics_logit0_thresh": m_ref,
        "pred_reachable_nodes": pred_idx,
        "true_reachable_nodes": true_idx,
        "top_prob_nodes": top_list,
        "prob": prob.detach().cpu(),
        "pred": pred.detach().cpu(),
        "y": y_i.detach().cpu(),
        "edge_index": ex["edge_index"].detach().cpu(),
        "edge_flow": ex["edge_flow"].detach().cpu(),
        "node_type": ex["node_type"].detach().cpu(),
        "meta": _get_meta(ex),
    }


def export_csv(out_path: Path, result: Dict[str, Any]) -> None:
    prob = result["prob"].numpy()
    pred = result["pred"].numpy()
    y = result["y"].numpy()
    node_type = result["node_type"].numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("node_id,node_type,y_true,y_pred,prob\n")
        for i in range(len(prob)):
            f.write(f"{i},{int(node_type[i])},{int(y[i])},{int(pred[i])},{float(prob[i]):.6f}\n")


def export_dot(out_path: Path, result: Dict[str, Any]) -> None:
    """
    GraphViz DOT export (no extra deps).
    Node label shows: id | type | true/pred
    """
    edge_index = result["edge_index"].numpy()
    edge_flow = result["edge_flow"].numpy()
    y = result["y"].numpy()
    pred = result["pred"].numpy()
    node_type = result["node_type"].numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("digraph G {\n")
        f.write('  rankdir=LR;\n')
        f.write('  node [shape=box, fontname="Consolas"];\n')

        for i in range(len(node_type)):
            if pred[i] == 1 and y[i] == 1:
                color = "palegreen"   # TP
            elif pred[i] == 1 and y[i] == 0:
                color = "gold"        # FP
            elif pred[i] == 0 and y[i] == 1:
                color = "tomato"      # FN
            else:
                color = "lightgray"   # TN

            label = f"{i} | t={int(node_type[i])} | y={int(y[i])}/p={int(pred[i])}"
            f.write(f'  n{i} [label="{label}", style=filled, fillcolor="{color}"];\n')

        for e in range(edge_index.shape[1]):
            s = int(edge_index[0, e])
            d = int(edge_index[1, e])
            ef = int(edge_flow[e])
            f.write(f'  n{s} -> n{d} [label="f={ef}", fontsize=10];\n')

        f.write("}\n")


def export_png(out_path: Path, result: Dict[str, Any], max_nodes: int = 300) -> None:
    """
    Draw a PNG using networkx + matplotlib.
    - Node colors: TP=green, FP=orange, FN=red, TN=gray.
    - Caps nodes to max_nodes to avoid freezing on large graphs.
    """
    edge_index = result["edge_index"].numpy()
    y = result["y"].numpy()
    pred = result["pred"].numpy()

    n = len(y)
    keep_n = min(n, max_nodes)

    G = nx.DiGraph()
    G.add_nodes_from(range(keep_n))
    for e in range(edge_index.shape[1]):
        s = int(edge_index[0, e])
        d = int(edge_index[1, e])
        if s < keep_n and d < keep_n:
            G.add_edge(s, d)

    colors = []
    for i in range(keep_n):
        if pred[i] == 1 and y[i] == 1:
            colors.append("tab:green")    # TP
        elif pred[i] == 1 and y[i] == 0:
            colors.append("tab:orange")   # FP
        elif pred[i] == 0 and y[i] == 1:
            colors.append("tab:red")      # FN
        else:
            colors.append("tab:gray")     # TN

    out_path.parent.mkdir(parents=True, exist_ok=True)

    pos = nx.spring_layout(G, seed=0)

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=220, alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrows=True, width=0.8, alpha=0.35)

    if keep_n <= 80:
        nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(
        f"Reachability Prediction (showing {keep_n}/{n} nodes)\n"
        f"TP=green FP=orange FN=red TN=gray   prob_thresh={result['prob_thresh']}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def load_checkpoint(model: ReachabilityGNN, ckpt_path: Path, device: str) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device)  # keep compatible with older torch
    model.load_state_dict(ckpt["model"])
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="train",
        help="train: train + eval + save checkpoint. predict: load checkpoint and show a visible sample output.",
    )
    parser.add_argument("--subset", type=str, default=str(ROOT / "data" / "subset_reachability"))
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "reachability_gnn.pt"))

    # predict options
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--index", type=int, default=0, help="sample index within the chosen split")
    parser.add_argument("--prob_thresh", type=float, default=0.5, help="probability threshold for reachable")
    parser.add_argument("--out_csv", type=str, default="", help="optional output csv path")
    parser.add_argument("--out_dot", type=str, default="", help="optional output GraphViz dot path")
    parser.add_argument("--out_png", type=str, default="", help="optional output PNG path (networkx+matplotlib)")
    parser.add_argument("--max_nodes", type=int, default=300, help="cap nodes when drawing PNG to avoid freezing")

    args = parser.parse_args()

    device = pick_device()
    subset = Path(args.subset)

    # infer vocab sizes
    counts = estimate_type_counts(subset)
    num_node_types = counts["num_node_types"]
    num_edge_flows = counts["num_edge_flows"]

    model = ReachabilityGNN(num_node_types=num_node_types, num_edge_flows=num_edge_flows, hidden=64, layers=3).to(device)

    if args.mode == "predict":
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        load_checkpoint(model, ckpt_path, device)

        ds = ReachabilityPyGDataset(subset, split=args.split, seed=0)
        if len(ds) == 0:
            raise RuntimeError(f"Empty split: {args.split} under {subset}")

        idx = args.index % len(ds)
        ex = ds[idx]
        result = predict_one(model, ex, device=device, prob_thresh=args.prob_thresh)

        key, step_id, split = result["meta"]
        print("\n================= PREDICT (VISIBLE OUTPUT) =================")
        print(f"device={device}")
        print(f"ckpt={ckpt_path}")
        print(f"sample: split={args.split} index={idx} meta(key={key}, step_id={step_id}, split={split})")
        print(f"graph: nodes={result['num_nodes']} edges={result['num_edges']} root_idx={result['root_idx']}")
        print(f"threshold: prob_thresh={result['prob_thresh']}")

        print("\n--- Metrics @ prob_thresh ---")
        m2 = result["metrics_prob_thresh"]
        print(
            f"iou={m2['iou']:.3f} f1={m2['f1']:.3f} prec={m2['prec']:.3f} rec={m2['rec']:.3f}  "
            f"tp={m2['tp']} fp={m2['fp']} fn={m2['fn']}"
        )

        print("\n--- Predicted reachable nodes (first 50) ---")
        pr = result["pred_reachable_nodes"]
        print(pr[:50], ("...(truncated)" if len(pr) > 50 else ""))
        print(f"pred_count={len(pr)}")

        print("\n--- True reachable nodes (first 50) ---")
        tr = result["true_reachable_nodes"]
        print(tr[:50], ("...(truncated)" if len(tr) > 50 else ""))
        print(f"true_count={len(tr)}")

        print("\n--- Top prob nodes (id, prob) ---")
        print(result["top_prob_nodes"])

        if args.out_csv:
            out_csv = Path(args.out_csv)
            export_csv(out_csv, result)
            print(f"\n[Saved] CSV: {out_csv}")

        if args.out_dot:
            out_dot = Path(args.out_dot)
            export_dot(out_dot, result)
            print(f"[Saved] DOT: {out_dot}")
            print("        (View DOT with GraphViz: dot -Tpng file.dot -o file.png)")

        if args.out_png:
            out_png = Path(args.out_png)
            export_png(out_png, result, max_nodes=args.max_nodes)
            print(f"[Saved] PNG: {out_png}")

        print("============================================================\n")
        return

    # -----------------------------
    # TRAIN mode (original behavior)
    # -----------------------------
    train_ds = ReachabilityPyGDataset(subset, split="train", seed=0)
    val_ds = ReachabilityPyGDataset(subset, split="val", seed=1)
    test_ds = ReachabilityPyGDataset(subset, split="test", seed=2)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_graph_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_graph_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_graph_batch, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    # compute pos_weight from a few train batches
    pos = 0.0
    neg = 0.0
    for k, batch in enumerate(train_loader):
        yy = batch["y"]
        pos += float((yy == 1).sum().item())
        neg += float((yy == 0).sum().item())
        if k >= 20:
            break
    pos_weight = torch.tensor([neg / max(1.0, pos)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"device={device}  num_node_types={num_node_types}  num_edge_flows={num_edge_flows}  pos_weight={pos_weight.item():.2f}")
    print(f"train_samples={len(train_ds)}  val_samples={len(val_ds)}  test_samples={len(test_ds)}")

    def run_epoch(loader, train: bool):
        model.train() if train else model.eval()

        total_loss = 0.0
        total = {"acc": 0.0, "f1": 0.0, "iou": 0.0}
        steps = 0

        for batch in loader:
            node_type = batch["node_type"].to(device)
            root_flag = batch["root_flag"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_flow = batch["edge_flow"].to(device)
            yy = batch["y"].to(device)

            logits = model(node_type, root_flag, edge_index, edge_flow)
            loss = loss_fn(logits, yy)

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            m = compute_metrics(logits.detach(), yy.detach(), thresh=0.0)
            total_loss += float(loss.item())
            total["acc"] += m["acc"]
            total["f1"] += m["f1"]
            total["iou"] += m["iou"]
            steps += 1

        for kk in total:
            total[kk] /= max(1, steps)

        return total_loss / max(1, steps), total

    best_val_iou = -1.0
    best_path = Path(args.ckpt)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 11):
        tr_loss, tr_m = run_epoch(train_loader, train=True)
        va_loss, va_m = run_epoch(val_loader, train=False)

        print(
            f"[{epoch:02d}] train loss={tr_loss:.4f} acc={tr_m['acc']:.3f} f1={tr_m['f1']:.3f} iou={tr_m['iou']:.3f} | "
            f"val loss={va_loss:.4f} acc={va_m['acc']:.3f} f1={va_m['f1']:.3f} iou={va_m['iou']:.3f}"
        )

        if va_m["iou"] > best_val_iou:
            best_val_iou = va_m["iou"]
            torch.save({"model": model.state_dict(), "num_node_types": num_node_types, "num_edge_flows": num_edge_flows}, best_path)

    # test with best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te_loss, te_m = run_epoch(test_loader, train=False)
    print(f"[TEST] loss={te_loss:.4f} acc={te_m['acc']:.3f} f1={te_m['f1']:.3f} iou={te_m['iou']:.3f}")
    print(f"Saved best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
