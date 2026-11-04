"""Run graph render + eval suite for 3 systems + dashboard. Usage: python -m adaptive.run_all"""
import base64
import html
import io
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from adaptive import baselines, eval_suite, graph

OUT = os.environ.get("OUTPUT_DIR", "output")
SYSTEMS = {
    "airflow_fixed": baselines.run_airflow_fixed,
    "simple_loop": baselines.run_simple_loop,
    "adaptive_langgraph": graph.run_adaptive,
}


def render_graph(path_png, path_mmd):
    g = graph.build().get_graph()
    open(path_mmd, "w").write(g.draw_mermaid())
    nx_g = nx.DiGraph()
    cond = set()
    for e in g.edges:
        nx_g.add_edge(e.source, e.target)
        if e.conditional:
            cond.add((e.source, e.target))
    pos = nx.spring_layout(nx_g, seed=4, k=1.4, iterations=300)
    fig, ax = plt.subplots(figsize=(13, 8))
    nx.draw_networkx_nodes(nx_g, pos, node_color="#90be6d", node_size=3600, ax=ax)
    nx.draw_networkx_labels(nx_g, pos, font_size=8, ax=ax)
    solid = [e for e in nx_g.edges if e not in cond]
    kw = dict(node_size=3600, arrowsize=16, connectionstyle="arc3,rad=0.12", ax=ax)
    nx.draw_networkx_edges(nx_g, pos, edgelist=solid, **kw)
    nx.draw_networkx_edges(nx_g, pos, edgelist=list(cond), style="dashed", edge_color="#e76f51", **kw)
    ax.set_title("Adaptive LangGraph workflow (dashed orange = conditional routing, solid = fixed edge)")
    ax.axis("off")
    fig.savefig(path_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    os.makedirs(OUT, exist_ok=True)
    render_graph(f"{OUT}/adaptive_graph.png", f"{OUT}/adaptive_graph.mmd")
    rows = []
    for case in eval_suite.CASES:
        for name, fn in SYSTEMS.items():
            res = fn(dict(case.cfg))
            sc = eval_suite.score_case(case, res)
            rows.append({"case": case.id, "system": name, "result": res, **sc})
            print(f"{case.id:22s} {name:20s} score={sc['score']:.2f} pass={sc['passed']} "
                  f"metric={res.get('test_metric', float('nan')):.3f} fits={res['fits']} path={'>'.join(res['path'])}")
    json.dump(rows, open(f"{OUT}/eval_results.json", "w"), indent=2, default=str)

    systems = list(SYSTEMS)
    summ = {s: {"pass": sum(r["passed"] for r in rows if r["system"] == s),
                "mean": sum(r["score"] for r in rows if r["system"] == s) / len(eval_suite.CASES),
                "fits": sum(r["result"]["fits"] for r in rows if r["system"] == s),
                "coord": sum(r["coordination_score"] or 0 for r in rows if r["system"] == s) / len(eval_suite.CASES)}
            for s in systems}
    json.dump(summ, open(f"{OUT}/eval_summary.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.27
    for i, s in enumerate(systems):
        ax.bar([j + i * w for j in range(len(eval_suite.CASES))],
               [next(r["score"] for r in rows if r["case"] == c.id and r["system"] == s) for c in eval_suite.CASES],
               w, label=s)
    ax.set_xticks([j + w for j in range(len(eval_suite.CASES))])
    ax.set_xticklabels([c.id.split("_")[0] for c in eval_suite.CASES])
    ax.axhline(0.8, color="k", ls="--", lw=1)
    ax.set_ylabel("case score (pass >= 0.8)")
    ax.legend(fontsize=8)
    chart = b64(fig)

    h = ["<html><head><meta charset='utf-8'><title>Agent eval dashboard</title><style>body{font-family:sans-serif;margin:24px}"
         "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #bbb;padding:4px 8px;font-size:13px}"
         ".p{background:#d8f3dc}.f{background:#ffd6d6}</style></head><body><h1>Multi-agent evaluation dashboard</h1>"]
    h.append("<h2>Workflow graph</h2><img src='data:image/png;base64,%s' width=900>" %
             base64.b64encode(open(f"{OUT}/adaptive_graph.png", "rb").read()).decode())
    h.append("<h2>Summary</h2><table><tr><th>System</th><th>Cases passed</th><th>Mean score</th><th>Mean coordination score</th><th>Total model fits</th></tr>")
    for s, v in summ.items():
        h.append(f"<tr><td>{s}</td><td>{v['pass']}/{len(eval_suite.CASES)}</td><td>{v['mean']:.2f}</td><td>{v['coord']:.2f}</td><td>{v['fits']}</td></tr>")
    h.append(f"</table><img src='data:image/png;base64,{chart}'>")
    h.append("<h2>Test cases</h2><table><tr><th>ID</th><th>Input</th><th>Expected</th><th>Scoring (weight)</th><th>Pass threshold</th></tr>")
    for c in eval_suite.CASES:
        sc = ", ".join(f"{k.name} [{k.kind}, w={k.weight}]" for k in c.checks)
        h.append(f"<tr><td>{c.id}</td><td>{html.escape(c.description)}<br><code>{html.escape(json.dumps(c.cfg))}</code></td>"
                 f"<td>{html.escape(c.expected)}</td><td>{html.escape(sc)}</td><td>{c.pass_threshold}</td></tr>")
    h.append("</table><h2>Results</h2><table><tr><th>Case</th><th>System</th><th>Score</th><th>Pass</th><th>Metric</th><th>Fits</th><th>Retries</th><th>Trace</th><th>Failed checks</th></tr>")
    for r in rows:
        res = r["result"]
        failed = ", ".join(k for k, v in r["checks"].items() if not v) or "-"
        m = f"{res['test_metric']:.3f}" if res.get("completed") else "n/a"
        h.append(f"<tr class={'p' if r['passed'] else 'f'}><td>{r['case']}</td><td>{r['system']}</td><td>{r['score']:.2f}</td>"
                 f"<td>{'PASS' if r['passed'] else 'FAIL'}</td><td>{m}</td><td>{res['fits']}</td><td>{res['retries']}</td>"
                 f"<td>{'&rarr;'.join(res['path'])}</td><td>{html.escape(failed)}</td></tr>")
    h.append("</table></body></html>")
    open(f"{OUT}/dashboard.html", "w").write("".join(h))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
