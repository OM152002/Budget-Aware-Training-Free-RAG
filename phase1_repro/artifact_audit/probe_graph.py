"""Read-only feasibility probes for the graph reconstruction plan (no SAGE modification).

argv: <sage_root> <docs_chunks_dir> <mode: emb|hnsw|fuzzy|cudaapi>
"""
import json, os, sys, time, random, resource
import numpy as np

sage, docs_dir, mode = sys.argv[1], sys.argv[2], sys.argv[3]
files = sorted(os.listdir(docs_dir))
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3  # macOS: bytes


def load_all():
    ids, embs, objs = [], [], []
    for f in files:
        o = json.load(open(os.path.join(docs_dir, f)))
        ids.append(o["chunk_id"]); embs.append(o["embedding"]); objs.append(o)
    return ids, np.asarray(embs, dtype=np.float32), objs


if mode == "emb":
    from sentence_transformers import SentenceTransformer
    ids, E, objs = load_all()
    n = np.linalg.norm(E, axis=1)
    print(f"stored norms: min {n.min():.6f} max {n.max():.6f}")
    random.seed(0); idx = random.sample(range(len(objs)), 200)
    for dev in ("cpu", "mps"):
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
        t = time.time()
        X = m.encode([objs[i]["content"] for i in idx], normalize_embeddings=True, batch_size=128, convert_to_numpy=True)
        dt = time.time() - t
        cos = (X * (E[idx] / n[idx, None])).sum(1)
        diff = np.abs(X - E[idx] / n[idx, None]).max(1)
        print(f"[{dev}] recompute vs stored on 200 chunks: cos min {cos.min():.7f} median {np.median(cos):.7f} | max|Δ| {diff.max():.2e} | {200/dt:.0f} texts/s")
        t = time.time()
        T = m.encode([o["metadata"]["topic"] for o in objs[:2000]], normalize_embeddings=True, batch_size=512, convert_to_numpy=True)
        print(f"[{dev}] topic encode 2000: {time.time()-t:.1f}s -> est. 22,716 topics {22716*(time.time()-t)/2000:.0f}s")

elif mode == "hnsw":
    import faiss
    ids, E, _ = load_all()
    faiss.normalize_L2(E)
    print(f"loaded {E.shape} rss {rss():.2f} GB; faiss threads {faiss.omp_get_max_threads()}")
    idx = faiss.IndexHNSWFlat(E.shape[1], 32)           # same as SAGE L1413-1415
    idx.hnsw.efConstruction = 1000; idx.hnsw.efSearch = 500
    t = time.time(); idx.add(E); tb = time.time() - t
    print(f"HNSW build (M=32, efC=1000): {tb:.1f}s rss {rss():.2f} GB")
    k = 201
    t = time.time(); D, I = idx.search(E, k); ts = time.time() - t   # batched; SAGE queries one at a time
    print(f"HNSW search all {len(E)} x k={k} (efS=500, batched): {ts:.1f}s rss {rss():.2f} GB")
    # one-at-a-time timing, as SAGE does (_find_k_neighbor_ids)
    t = time.time()
    for i in range(500): idx.search(E[i:i+1], k)
    print(f"single-query search: {(time.time()-t)/500*1000:.2f} ms/query -> est. {(time.time()-t)/500*len(E):.0f}s for all (1 thread)")
    # exact ground truth on a sample
    flat = faiss.IndexFlatIP(E.shape[1]); flat.add(E)
    s = np.random.RandomState(0).choice(len(E), 1000, replace=False)
    _, Ie = flat.search(E[s], k)
    rec = np.mean([len(set(I[j]) & set(Ie[r])) / k for r, j in enumerate(s)])
    print(f"HNSW recall@201 vs exact (1000 queries): {rec:.4f}")
    pairs = set()
    for i in range(len(E)):
        c = 0
        for j in I[i]:
            if j == -1 or j == i: continue
            if c == 200: break
            pairs.add((i, j) if i < j else (j, i)); c += 1
    print(f"unique undirected doc-doc candidate pairs (docs-only index, k=200): {len(pairs):,}; rss {rss():.2f} GB")
    sims = np.array([float(E[a] @ E[b]) for a, b in list(pairs)[:200000]])
    print(f"content sim sample: q99={np.quantile(sims, .99):.4f} q95={np.quantile(sims, .95):.4f} median={np.median(sims):.4f}")
    np.save("hnsw_I.npy", I)

elif mode == "fuzzy":
    sys.path.insert(0, sage); os.makedirs("output", exist_ok=True)
    from loguru import logger; logger.remove()
    import src.pipeline  # noqa: F401  (circular-import workaround, as in Phase 1)
    from scripts.analysis.graph_analysis_pipeline import GPUAcceleratedSimilarityCalculator, ChunkData
    ids, E, objs = load_all()
    I = np.load("hnsw_I.npy")
    chunks = [ChunkData(chunk_id=o["chunk_id"], chunk_type="document", content=o["content"], content_embedding=[],
                        entities=o["metadata"].get("entities") or {}, entity_embeddings={}, topic=o["metadata"]["topic"],
                        metadata=o["metadata"]) for o in objs]
    calc = GPUAcceleratedSimilarityCalculator(chunks, embedding_service=None)
    print("calculator device:", calc.device)
    rng = np.random.RandomState(0); sample = [(int(a), int(I[a, rng.randint(1, 201)])) for a in rng.randint(0, len(objs), 20000)]
    t = time.time(); ec = []
    for a, b in sample:
        cnt, _, ov = calc._fuzzy_entity_match_gpu(chunks[a].entities, chunks[b].entities, threshold=0.75); ec.append(cnt)
    te = (time.time() - t) / len(sample)
    t = time.time()
    for a, b in sample:
        e1, e2 = chunks[a].metadata.get("events", {}), chunks[b].metadata.get("events", {})
        if e1 and e2: calc._calculate_event_overlap(e1, e2)
    tv = (time.time() - t) / len(sample)
    ec = np.array(ec)
    print(f"entity fuzzy match: {te*1e6:.0f} us/pair | event overlap: {tv*1e6:.0f} us/pair (unmodified SAGE methods, CPU)")
    print(f"fraction of sampled candidate pairs with entity_count>0: {(ec > 0).mean():.3f} (mean count {ec.mean():.2f})")

elif mode == "cudaapi":
    import torch
    print("torch", torch.__version__, "cuda available", torch.cuda.is_available(), "mps", torch.backends.mps.is_available())
    for name, fn in [("torch.cuda.empty_cache()", lambda: torch.cuda.empty_cache()),
                     ("torch.cuda.memory_allocated(cpu)", lambda: torch.cuda.memory_allocated(torch.device("cpu"))),
                     ("torch.cuda.Event()", lambda: torch.cuda.Event(enable_timing=True)),
                     ("torch.cuda.synchronize()", lambda: torch.cuda.synchronize())]:
        try: r = fn(); print(f"{name}: OK -> {r}")
        except Exception as e: print(f"{name}: RAISES {type(e).__name__}: {str(e)[:90]}")
