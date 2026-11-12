"""Graph/seed adapter: question_id + k -> top-k seed chunks and their 1-hop document-graph neighbors.

Read-only over existing artifacts (nothing in sage/, graph_reconstruction/runs or seed_retrieval/runs is written):
  seeds   seed_retrieval/runs/seeds_fusionA_top100_seed0_runA/seeds_top100.csv (+ _scores.jsonl)
  graph   graph_reconstruction/runs/docgraph_k200_q99_20261023/output/analysis_cache/knowledge_graph/unique_edges_*.json
  nodes   graph_reconstruction/runs/docgraph_k200_q99_20261023/output/indexing_filtered.csv (all 21,742, incl. isolated)
  chunks  sage_hf_artifacts/extracted/docs_chunks_1/*.json (text + LLM metadata; embeddings dropped)

Neighbor order is deterministic and NOT a relevance ranking: by the best (lowest) rank of the seeds a neighbor
is adjacent to, then chunk_id. Scoring/selection of neighbors is the controller's job (not implemented here).
"""
import ast
import json
import os
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
GRAPH_RUN = PROJECT / "graph_reconstruction" / "runs" / "docgraph_k200_q99_20261023" / "output"
DEFAULTS = {
    "seeds_csv": PROJECT / "seed_retrieval" / "runs" / "seeds_fusionA_top100_seed0_runA" / "seeds_top100.csv",
    "seed_scores_jsonl": PROJECT / "seed_retrieval" / "runs" / "seeds_fusionA_top100_seed0_runA" / "seeds_top100_scores.jsonl",
    "edges_json": GRAPH_RUN / "analysis_cache" / "knowledge_graph" / "unique_edges_20261023_053523.json",
    "nodes_csv": GRAPH_RUN / "indexing_filtered.csv",
    "docs_dir": PROJECT / "sage_hf_artifacts" / "extracted" / "docs_chunks_1",
    "cache_dir": Path(__file__).resolve().parent / "cache",
}
ORDINALS = [f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}rd" if i == 3 else f"{i}th" for i in range(1, 101)]
EDGE_FIELDS = ("reason", "content_similarity", "topic_similarity", "entity_relationship_overlap", "event_count")


class UnresolvedSeedError(ValueError):
    pass


class UnknownNodeError(KeyError):
    """A node ID that is not in the reconstructed graph (distinct from a real node with no neighbors)."""


@dataclass
class Node:
    chunk_id: str
    content: str
    source_id: str
    source_name: str
    topic: str
    entities: dict
    events: dict
    timeline: list
    keywords: list
    degree: int


@dataclass
class Seed:
    rank: int
    node: Node
    combined_score: Optional[float] = None
    hnsw_score: Optional[float] = None
    bm25_score: Optional[float] = None


@dataclass
class Neighbor:
    node: Node
    via_seeds: List[str]                        # seeds this neighbor is adjacent to (seed-rank order)
    best_seed_rank: int
    edges: Dict[str, dict] = field(default_factory=dict)   # seed_id -> released edge attributes (EDGE_FIELDS)


@dataclass
class Expansion:
    question_id: str
    question: str
    k: int
    seeds: List[Seed]
    neighbors: List[Neighbor]
    unresolved_seed_ids: List[str]
    doc_gold: List[str]
    eval_group: str

    def to_dict(self):
        return asdict(self)


class GraphSeedAdapter:
    def __init__(self, strict: bool = True, use_cache: bool = True, **paths):
        self.p = {**DEFAULTS, **{k: Path(v) for k, v in paths.items()}}
        self.strict = strict
        self._load_seeds()
        self._load_graph(use_cache)
        self._load_chunks(use_cache)

    # ---------------- loading ----------------
    def _load_seeds(self):
        df = pd.read_csv(self.p["seeds_csv"])
        self.questions = {}
        for d in df.to_dict("records"):  # not itertuples: it renames non-identifier columns like "1st"
            self.questions[d["question_id"]] = {
                "question": d["question"],
                "ranking": [d[c] for c in ORDINALS if isinstance(d.get(c), str)],
                "doc_gold": json.loads(d["doc_gold"]),
                "gold_docs": ast.literal_eval(d["gold_docs"]),
                "eval_group": d["eval_group"],
            }
        self.seed_scores = {}
        if self.p["seed_scores_jsonl"].exists():
            with open(self.p["seed_scores_jsonl"]) as f:
                for line in f:
                    o = json.loads(line)
                    self.seed_scores[o["question_id"]] = {x["chunk_id"]: x for x in o["ranking"]}

    def _cache(self, name, build, use_cache, sources):
        path = self.p["cache_dir"] / name
        stamp = [(str(s), os.path.getmtime(s), os.path.getsize(s)) for s in sources]
        if use_cache and path.exists():
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if obj.get("stamp") == stamp:
                return obj["data"]
        data = build()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"stamp": stamp, "data": data}, f, protocol=pickle.HIGHEST_PROTOCOL)
        return data

    def _load_graph(self, use_cache):
        def build():
            nodes = pd.read_csv(self.p["nodes_csv"]).chunk_id.tolist()
            adj = {n: {} for n in nodes}
            with open(self.p["edges_json"]) as f:
                edges = json.load(f)
            for e in edges:
                s, t = e["source_chunk_id"], e["target_chunk_id"]
                attrs = tuple(e.get(k) for k in EDGE_FIELDS)
                adj[s][t] = attrs  # KeyError here would mean an edge endpoint outside the node set
                adj[t][s] = attrs
            return {"adj": adj, "n_edges": len(edges)}
        g = self._cache("graph.pkl", build, use_cache, [self.p["nodes_csv"], self.p["edges_json"]])
        self.adj, self.n_edges = g["adj"], g["n_edges"]

    def _load_chunks(self, use_cache):
        def build():
            store = {}
            for fn in sorted(os.listdir(self.p["docs_dir"])):
                if not fn.endswith(".json"):
                    continue
                with open(self.p["docs_dir"] / fn) as f:
                    o = json.load(f)
                m, si = o.get("metadata") or {}, o.get("source_info") or {}
                store[o["chunk_id"]] = (o.get("content", ""), si.get("source_id", ""), si.get("source_name", ""),
                                        m.get("topic", ""), m.get("entities") or {}, m.get("events") or {},
                                        m.get("timeline") or [], o.get("keywords") or [])
            return store
        self.chunks = self._cache("chunks.pkl", build, use_cache, [self.p["docs_dir"]])
        missing = [n for n in self.adj if n not in self.chunks]
        if missing:
            raise RuntimeError(f"{len(missing)} graph nodes have no chunk file, e.g. {missing[:3]}")

    # ---------------- public API ----------------
    @property
    def question_ids(self):
        return list(self.questions)

    def in_graph(self, chunk_id: str) -> bool:
        return chunk_id in self.adj

    def node(self, chunk_id: str) -> Node:
        if not self.in_graph(chunk_id):
            raise UnknownNodeError(chunk_id)
        c = self.chunks[chunk_id]
        return Node(chunk_id, *c, degree=len(self.adj.get(chunk_id, ())))

    def get_seeds(self, question_id: str, k: int):
        """Top-k seeds (rank order) and any seed IDs absent from the graph."""
        if question_id not in self.questions:
            raise KeyError(f"unknown question_id {question_id!r}")
        ranking = self.questions[question_id]["ranking"]
        if not 1 <= k <= len(ranking):
            raise ValueError(f"k must be in [1, {len(ranking)}], got {k}")
        ids = ranking[:k]
        unresolved = [c for c in ids if not self.in_graph(c) or c not in self.chunks]
        if unresolved and self.strict:
            raise UnresolvedSeedError(f"{question_id}: seeds not in graph: {unresolved}")
        sc = self.seed_scores.get(question_id, {})
        seeds = [Seed(rank=i + 1, node=self.node(c), combined_score=sc.get(c, {}).get("combined_score"),
                      hnsw_score=sc.get(c, {}).get("hnsw_score"), bm25_score=sc.get(c, {}).get("bm25_score"))
                 for i, c in enumerate(ids) if c not in unresolved]
        return seeds, unresolved

    def get_neighbors(self, seed_ids: List[str]) -> List[Neighbor]:
        """Union of 1-hop neighbors of seed_ids, excluding the seeds themselves, deduplicated.
        Raises UnknownNodeError for an ID not in the graph (an isolated real node returns no neighbors)."""
        unknown = [s for s in seed_ids if not self.in_graph(s)]
        if unknown:
            raise UnknownNodeError(unknown)
        seed_set, rank = set(seed_ids), {s: i + 1 for i, s in enumerate(seed_ids)}
        acc: Dict[str, Neighbor] = {}
        for s in seed_ids:
            for n, attrs in self.adj.get(s, {}).items():
                if n in seed_set:
                    continue
                nb = acc.get(n)
                if nb is None:
                    nb = acc[n] = Neighbor(node=self.node(n), via_seeds=[], best_seed_rank=rank[s])
                nb.via_seeds.append(s)
                nb.edges[s] = dict(zip(EDGE_FIELDS, attrs))
        return sorted(acc.values(), key=lambda x: (x.best_seed_rank, x.node.chunk_id))

    def expand(self, question_id: str, k: int) -> Expansion:
        seeds, unresolved = self.get_seeds(question_id, k)
        q = self.questions[question_id]
        return Expansion(question_id=question_id, question=q["question"], k=k, seeds=seeds,
                         neighbors=self.get_neighbors([s.node.chunk_id for s in seeds]),
                         unresolved_seed_ids=unresolved, doc_gold=q["doc_gold"], eval_group=q["eval_group"])
