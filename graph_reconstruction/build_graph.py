"""Full SAGE-style document-only graph build on CPU (decisions G1-G5 of GRAPH_RECONSTRUCTION_PLAN.md).

Runs SAGE's released steps: load_chunks -> build_hnsw_index -> analyze_chunk_similarities(k_neighbors=200)
-> _generate_outlier_analysis (doc-doc 99th-pct AND rule + entity_count>0) -> _extract_edges_for_graph.
Skipped (report-only, G3/G4): section plots, outlier content JSONs, connected components, Leiden, KG export.

Must be launched with OMP_NUM_THREADS=1 (faiss-cpu and torch bundle separate libomp copies; multithreaded
OpenMP in one process crashes natively on this Mac).

Usage: OMP_NUM_THREADS=1 python -I build_graph.py <run_dir>
"""
import glob
import json
import os
import sys
import threading
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sage_cpu_runner as R  # noqa: E402


class PeakRSS(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.proc, self.stop = 0, psutil.Process(), False

    def run(self):
        while not self.stop:
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.5)


def main():
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("set OMP_NUM_THREADS=1 (see module docstring)")
    run_dir = Path(sys.argv[1]).resolve()
    ids = R.prepare_run_dir(run_dir)
    R.setup_logging(run_dir)
    R.apply_patches()
    os.chdir(run_dir)

    mon = PeakRSS(); mon.start()
    stages, summary = {}, {"skipped_steps": ["section plots/summary (generate_analysis_reports sections 1-3, overall)",
                                             "outlier content JSONs (_save_outliers_with_content)",
                                             "analyze_connected_components", "Leiden community detection",
                                             "KnowledgeGraph object creation/export"]}

    def stage(name, fn):
        t, r0 = time.time(), mon.proc.memory_info().rss
        out = fn()
        stages[name] = {"seconds": round(time.time() - t, 1), "rss_after_gb": round(mon.proc.memory_info().rss / 1e9, 2),
                        "peak_so_far_gb": round(mon.peak / 1e9, 2)}
        print(f"[{name}] {stages[name]}", flush=True)
        return out

    t_all = time.time()
    p = stage("init", R.new_pipeline)
    stage("load_chunks", p.load_chunks)
    n_doc = sum(c.chunk_type == "document" for c in p.chunks)
    n_tab = sum(c.chunk_type == "table" for c in p.chunks)
    assert n_doc == 21742 and n_tab == 0 and len(ids) == 21742, (n_doc, n_tab, len(ids))
    stage("build_hnsw_index", p.build_hnsw_index)

    # record the candidate-pair count without altering SAGE's method
    orig_pairs = p._generate_unique_pairs
    def counted(rel):
        out = orig_pairs(rel)
        summary["candidate_pairs"] = len(out)
        summary["chunks_with_neighbors"] = len(rel)
        summary["neighbor_list_len_min_max"] = [min(map(len, rel.values())), max(map(len, rel.values()))]
        return out
    p._generate_unique_pairs = counted
    stage("analyze_chunk_similarities_k200", lambda: p.analyze_chunk_similarities(k_neighbors=200))

    df = stage("similarity_dataframe", p._create_similarity_dataframe)
    dd = df[df["edge_type"] == "doc-doc"]
    # same expressions as graph_analysis_pipeline.py L2101-2102 (recorded for the report)
    summary["thresholds_q99"] = {"topic_similarity": float(dd["topic_similarity"].quantile(0.99)),
                                 "content_similarity": float(dd["content_similarity"].quantile(0.99))}
    summary["edge_type_counts"] = df["edge_type"].value_counts().to_dict()

    skipped_rows = {}
    def no_content_dump(outliers_df, file_path, description, comparison_type=""):
        skipped_rows[description] = len(outliers_df)  # report-only (G3); edge storage already happened
    p._save_outliers_with_content = no_content_dump
    reports = p.cache_dir / "analysis_reports"; reports.mkdir(exist_ok=True)
    stage("outlier_edge_selection", lambda: p._generate_outlier_analysis(df, reports))
    summary["outlier_rows_by_rule"] = skipped_rows
    summary["stored_outlier_edges_before_dedup"] = len(p.outlier_edges)

    edges = stage("extract_edges_for_graph", lambda: p._extract_edges_for_graph(df))
    summary["final_edges"] = len(edges)
    summary["edges_by_reason"] = {}
    for e in edges:
        summary["edges_by_reason"][e["reason"]] = summary["edges_by_reason"].get(e["reason"], 0) + 1
    summary["edges_file"] = sorted(glob.glob(str(p.cache_dir / "knowledge_graph" / "unique_edges_*.json")))[-1]

    mon.stop = True
    summary.update(nodes=n_doc, stages=stages, total_seconds=round(time.time() - t_all, 1),
                   peak_rss_gb=round(mon.peak / 1e9, 2), patch_calls=dict(R.PATCH_LOG),
                   omp_num_threads=os.environ.get("OMP_NUM_THREADS"))
    (run_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "stages"}, indent=2))


if __name__ == "__main__":
    main()
