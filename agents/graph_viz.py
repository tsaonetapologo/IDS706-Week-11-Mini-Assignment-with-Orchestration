"""Live agent graph: re-rendered to output/agent_graph.png every time the active agent changes."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
EDGES = [
    ("DataAgent", "FeatureAgent", ""),
    ("FeatureAgent", "TrainerAgent", ""),
    ("TrainerAgent", "EvaluatorAgent", ""),
    ("EvaluatorAgent", "TunerAgent", "val_acc < target"),
    ("TunerAgent", "TrainerAgent", "next config"),
    ("EvaluatorAgent", "ReporterAgent", "done"),
]
POS = {
    "DataAgent": (0, 0), "FeatureAgent": (1, 0), "TrainerAgent": (2, 0),
    "EvaluatorAgent": (3, 0), "TunerAgent": (2.5, -1.3), "ReporterAgent": (4.4, 0),
}


def draw(active=None, visited=(), path=None):
    path = path or os.path.join(OUTPUT_DIR, "agent_graph.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    g = nx.DiGraph()
    g.add_weighted_edges_from([(a, b, 1) for a, b, _ in EDGES])
    colors = ["#f4a261" if n == active else "#90be6d" if n in visited else "#cfd8dc" for n in g.nodes]
    fig, ax = plt.subplots(figsize=(12, 5))
    nx.draw_networkx_nodes(g, POS, node_color=colors, node_size=3800, ax=ax)
    nx.draw_networkx_labels(g, POS, font_size=8, ax=ax)
    nx.draw_networkx_edges(g, POS, arrows=True, node_size=3800, arrowsize=18,
                           connectionstyle="arc3,rad=0.15", ax=ax)
    nx.draw_networkx_edge_labels(g, POS, {(a, b): l for a, b, l in EDGES if l}, font_size=7,
                                 label_pos=0.5, ax=ax)
    ax.set_title(f"Agent graph (orange = active, green = done) - active: {active}")
    ax.axis("off")
    ax.margins(0.15)
    fig.savefig(path, dpi=110)
    plt.close(fig)
