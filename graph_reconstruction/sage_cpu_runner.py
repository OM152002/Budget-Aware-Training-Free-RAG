"""Run SAGE's released graph builder (scripts/analysis/graph_analysis_pipeline.py) on CPU, documents only.

The SAGE repository is NOT modified. This module imports the unmodified SAGE classes and applies two
CUDA-removal patches (P1, P2), each with the same body as the original apart from the CUDA call:

  P1  OptimizedEmbeddingService._generate_embeddings_massive_batch
      torch.cuda.Event / torch.cuda.synchronize (timing only) -> time.perf_counter. encode() call unchanged.
  P2  GraphAnalysisPipeline._calculate_similarities_for_pairs
      drops the `if not torch.cuda.is_available(): raise` guard (L1590-1592). Body otherwise verbatim.

Report-only side effects are skipped by the caller (user decision G3): section plots, outlier JSONs with
content (_save_outliers_with_content), connected components, Leiden (G4), KnowledgeGraph export.
Edge selection (_analyze_doc_doc_outliers -> _store_outliers_for_graph_building -> _extract_edges_for_graph)
runs unmodified.
"""
import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SAGE_ROOT = PROJECT / "sage"
DOCS_DIR = PROJECT / "sage_hf_artifacts" / "extracted" / "docs_chunks_1"
TABLES_FILE = PROJECT / "sage_hf_artifacts" / "raw" / "table_chunks_with_metadata.json"

sys.path.insert(0, str(SAGE_ROOT))

import faiss  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

import src.pipeline  # noqa: E402,F401  (must precede src.processors: circular import in the repo)
from src.processors import embedding_service_optimized as eso  # noqa: E402
from scripts.analysis import graph_analysis_pipeline as gap  # noqa: E402

PATCH_LOG = {"P1_calls": 0, "P2_calls": 0}


def _massive_batch_cpu(self, texts):
    """P1: embedding_service_optimized.py L113-150 with CUDA-event timing replaced by perf_counter."""
    PATCH_LOG["P1_calls"] += 1
    logger.info(f" MASSIVE BATCH: Processing {len(texts):,} texts at once")
    model_to_use = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
    internal_batch_size = 256 if self.use_multi_gpu else 128
    start = time.perf_counter()
    embeddings = model_to_use.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        device=self.device,
        batch_size=internal_batch_size,
        normalize_embeddings=True,
    )
    processing_time = time.perf_counter() - start
    if embeddings.is_cuda:
        embeddings = embeddings.cpu()
    embeddings_list = embeddings.numpy().tolist()
    logger.info(f" MASSIVE BATCH COMPLETE: {len(embeddings_list):,} embeddings in {processing_time:.2f}s")
    return embeddings_list


def _calculate_similarities_for_pairs_cpu(self, unique_pairs):
    """P2: graph_analysis_pipeline.py L1587-1613 without the CUDA-only guard."""
    PATCH_LOG["P2_calls"] += 1
    logger.info("Using GPU-only similarity calculation (CPU port: CUDA guard removed)")
    gpu_calculator = gap.GPUAcceleratedSimilarityCalculator(
        chunks=self.chunks,
        embedding_service=self.embedding_service,
        batch_size=10000,
        gpu_id=0,
    )
    gpu_calculator.precompute_embeddings()
    gpu_calculator._precompute_all_similarities(unique_pairs)
    self.similarity_data = gpu_calculator.calculate_similarities_batch(unique_pairs)
    logger.info(f"GPU-only processing completed: {len(self.similarity_data)} similarities calculated")


def apply_patches():
    eso.OptimizedEmbeddingService._generate_embeddings_massive_batch = _massive_batch_cpu
    gap.GraphAnalysisPipeline._calculate_similarities_for_pairs = _calculate_similarities_for_pairs_cpu


def prepare_run_dir(run_dir: Path) -> list:
    """Create a clean cwd-relative layout SAGE expects; write indexing_filtered.csv with all doc chunk IDs."""
    if run_dir.exists():
        raise SystemExit(f"run dir already exists (refusing to reuse stale caches): {run_dir}")
    fp = run_dir / "output" / "full_pipeline"
    fp.mkdir(parents=True)
    (fp / "docs_chunks_1").symlink_to(DOCS_DIR)
    # Real table file; SAGE's own indexing filter (L1306) drops every table because none is listed (decision G1).
    (fp / "table_chunks_with_metadata.json").symlink_to(TABLES_FILE)
    ids = []
    for f in sorted(os.listdir(DOCS_DIR)):
        if f.endswith(".json"):
            with open(DOCS_DIR / f) as fh:
                ids.append(json.load(fh)["chunk_id"])
    pd.DataFrame({"chunk_id": ids}).to_csv(run_dir / "output" / "indexing_filtered.csv", index=False)
    return ids


def setup_logging(run_dir: Path):
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    logger.add(run_dir / "sage_pipeline.log", level="INFO")


def new_pipeline(single_thread_faiss=True):
    if single_thread_faiss:
        faiss.omp_set_num_threads(1)  # deterministic HNSW construction (validation V6)
    return gap.GraphAnalysisPipeline()  # defaults: ./output/full_pipeline/..., ./output/analysis_cache, 128 threads
