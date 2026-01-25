# GNN
https://zenodo.org/records/4247595
data:Need to download graphs_20.06.01.tar.bz2  labels_reachability_20.06.01.tar.bz2 vocab_20.06.01.tar.bz2 from above link
Project Overview: Reachability Prediction on Program Graphs
1) Purpose (What problem are we solving?)

This project builds a Graph Neural Network (GNN) system to predict reachability / activation of nodes in a program graph, conditioned on a given root node.
In other words: given a program represented as a graph and a starting point (“root”), we want the model to learn which nodes become reachable / active.

This is useful as a learning exercise for:

program-graph representations (ProGraML / ProgramGraph protobuf)

graph machine learning

converting structured compiler graphs into training-ready datasets

2) Data & Labels (What are the inputs and outputs?)

Input graph (ProgramGraph.pb)

Nodes: include fields like type, text, etc.

Edges: include a flow category (e.g., data flow vs control flow types)

Label file (ProgramGraphFeaturesList.pb)

Each graph has multiple “steps” (e.g., len(y.graph)=7)

Each step contains:

data_flow_root_node: a one-hot indicator showing which node is the root

data_flow_value: a per-node binary label (0/1) indicating whether the node is active/reachable

graph-level statistics such as:

data_flow_active_node_count

data_flow_step_count

Sanity checks confirmed:

the number of nodes in labels matches the graph

sum(data_flow_value == 1) matches data_flow_active_node_count

3) Model Design (How does the model work?)

We use a lightweight Edge-aware GraphSAGE-style GNN:

Node features

node_type → learned embedding

root_flag (1 if node is root else 0) → projected and added into node embeddings

Edge features

edge_flow → learned embedding

edge embeddings are injected into message passing:

<img width="313" height="33" alt="image" src="https://github.com/user-attachments/assets/d44e4d76-9742-4cc4-ad3b-73a5185f025b" />

	​


then aggregated by mean over incoming edges.

Prediction

an MLP outputs one logit per node

sigmoid(logit) gives probability of being reachable/active

Loss

BCEWithLogitsLoss with pos_weight to handle class imbalance (usually reachable nodes are fewer than non-reachable nodes)

4) Pipeline & Functionality (What can the code do?)

The script supports two main modes:

A) Train mode

trains the GNN for 10 epochs

evaluates on validation set

saves the best checkpoint based on IoU (Jaccard score)

Command:

python .\scripts\train_reachability_gnn.py --mode train

B) Predict mode (visible demo)

loads a trained checkpoint

runs inference on one graph sample

prints:

graph size (nodes, edges)

root index

metrics (IoU, F1, precision, recall)

predicted reachable node list vs ground truth

top probability nodes

optionally exports:

CSV (node-wise predictions)

DOT (GraphViz visualization)

PNG (networkx + matplotlib visualization)

Example:

python .\scripts\train_reachability_gnn.py --mode predict --split test --index 0 --out_png outputs\pred0.png

5) Results (What did we observe?)

For a sample test graph:

The graph contains about 134 nodes and ~225 edges

Labels show only a subset of nodes are active, for example:

18 active nodes out of 134 in one step (~13%)

In prediction mode, the system reports:

IoU / F1 / precision / recall, which reflect how well the model identifies the active node set

Visualization outputs (DOT/PNG) clearly show:

True positives (correctly predicted active)

False positives (predicted active but actually inactive)

False negatives (missed active nodes)

True negatives

This makes the project not only trainable, but also easy to demonstrate and debug visually.

6) Key Takeaways

We successfully converted ProGraML protobuf graphs into a training dataset.

The GNN learns node-level reachability conditioned on a root node.

The predict mode provides strong “demo value” via:

printed metrics

CSV export

DOT/PNG visualization

<img width="640" height="482" alt="image" src="https://github.com/user-attachments/assets/60a7285e-99ba-4f52-a785-de4ab1ab6608" />



