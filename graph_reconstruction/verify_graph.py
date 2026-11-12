"""Independent post-build verification (plan §5: V5, V8, V9) from the on-disk SAGE outputs only.

Recomputes the doc-doc thresholds and the edge set from SAGE's saved batch CSVs with NumPy
(not with SAGE code) and checks the released edges file against it.

Usage: python -I verify_graph.py <run_dir>
"""
import glob
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

run = Path(sys.argv[1]).resolve()
summary = json.loads((run / "build_summary.json").read_text())
batches = sorted(glob.glob(str(run / "output" / "batch_results" / "batch_*.csv")))
df = pd.concat((pd.read_csv(f, usecols=["source_chunk_id", "target_chunk_id", "edge_type", "topic_similarity",
                                         "content_similarity", "entity_count", "event_count"]) for f in batches),
               ignore_index=True)
out = {"batch_files": len(batches), "pairs_in_batches": len(df)}
dd = df[df.edge_type == "doc-doc"]
tt = float(np.quantile(dd.topic_similarity.to_numpy(np.float64), 0.99))   # numpy linear == pandas default
tc = float(np.quantile(dd.content_similarity.to_numpy(np.float64), 0.99))
out["thresholds_numpy"] = {"topic_similarity": tt, "content_similarity": tc}
out["thresholds_match_build"] = (abs(tt - summary["thresholds_q99"]["topic_similarity"]) < 1e-12 and
                                 abs(tc - summary["thresholds_q99"]["content_similarity"]) < 1e-12)

and_mask = (dd.content_similarity > tc) & (dd.topic_similarity > tt)
ent_mask = dd.entity_count > 0
exp = dd[and_mask | ent_mask]
expected = set(zip(exp.source_chunk_id, exp.target_chunk_id))
out["expected_edges"] = {"AND_rule": int(and_mask.sum()), "entity_rule": int(ent_mask.sum()),
                         "both": int((and_mask & ent_mask).sum()), "union": len(expected)}

edges = json.loads((run / summary["edges_file"]).read_text())
got = {(e["source_chunk_id"], e["target_chunk_id"]) for e in edges}
out["edges_file_count"] = len(edges)
out["V8_edges_file_equals_independent_rule"] = got == expected
out["V8_missing_vs_rule"], out["V8_extra_vs_rule"] = len(expected - got), len(got - expected)
out["V8_entity_count_field_all_zero_quirk_L3045"] = all(e["entity_count"] == 0 for e in edges)
out["no_self_loops"] = all(s != t for s, t in got)
undirected = {frozenset(x) for x in got}
out["no_duplicate_undirected_edges"] = len(undirected) == len(got)
out["edge_ids_unique"] = len({e["edge_id"] for e in edges}) == len(edges)
out["edge_keys"] = sorted(edges[0].keys())

# V5: pairs whose AND-rule status could flip under a ±1e-5 float perturbation
near_t = (dd.topic_similarity - tt).abs() < 1e-5
near_c = (dd.content_similarity - tc).abs() < 1e-5
fragile = (near_t | near_c) & ~ent_mask
out["V5_near_threshold_pairs_not_saved_by_entity_rule"] = int(fragile.sum())

# V9: structure
ids = pd.read_csv(run / "output" / "indexing_filtered.csv").chunk_id.tolist()
G = nx.Graph(); G.add_nodes_from(ids); G.add_edges_from(got)
deg = np.array([d for _, d in G.degree()])
comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
out["graph"] = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                "endpoints_all_in_doc_set": all(s in G and t in G for s, t in got) and G.number_of_nodes() == len(ids),
                "isolated_nodes": int((deg == 0).sum()), "degree_mean": round(float(deg.mean()), 2),
                "degree_median": float(np.median(deg)), "degree_p99": float(np.quantile(deg, .99)), "degree_max": int(deg.max()),
                "components": len(comps), "largest_component": comps[0], "largest_component_frac": round(comps[0] / len(ids), 4)}
out["entity_count_distribution_over_edges"] = dd[ent_mask].entity_count.value_counts().sort_index().head(10).to_dict()
out["ALL_PASS"] = all([out["thresholds_match_build"], out["V8_edges_file_equals_independent_rule"],
                       out["V8_entity_count_field_all_zero_quirk_L3045"], out["no_self_loops"],
                       out["no_duplicate_undirected_edges"], out["edge_ids_unique"], out["graph"]["endpoints_all_in_doc_set"],
                       out["pairs_in_batches"] == summary["candidate_pairs"]])
(run / "verification_report.json").write_text(json.dumps(out, indent=2, default=int))
print(json.dumps(out, indent=2, default=int))
