"""Audit docs_chunks_1/*.json: key sets, types, metadata coverage, id uniqueness, link/neighbor fields."""
import json, sys, os, collections, re

d = sys.argv[1]
files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
top = collections.Counter(); si = collections.Counter(); md = collections.Counter()
types = collections.defaultdict(collections.Counter)
ent_sub = collections.Counter(); ev_sub = collections.Counter()
ids = []; srcs = set(); fname_mismatch = 0; emb_dims = collections.Counter()
empty = collections.Counter(); sl_lens = collections.Counter(); sl_examples = []
ctype = collections.Counter(); stype = collections.Counter(); chunk_idx = collections.Counter()
suspicious_keys = set(); errors = 0; contentlens = []
for f in files:
    try:
        o = json.load(open(os.path.join(d, f)))
    except Exception as e:
        errors += 1; continue
    top.update(o.keys())
    for k, v in o.items():
        types[k][type(v).__name__] += 1
    for k in o:
        if re.search(r"neigh|edge|link|graph|communit|adj|similar", k, re.I): suspicious_keys.add(("top", k))
    s = o.get("source_info", {}); si.update(s.keys())
    for k in s:
        types["source_info." + k][type(s[k]).__name__] += 1
        if re.search(r"neigh|edge|link|graph|communit|adj|similar", k, re.I): suspicious_keys.add(("source_info", k))
    m = o.get("metadata", {}) or {}; md.update(m.keys())
    for k in m:
        types["metadata." + k][type(m[k]).__name__] += 1
        if re.search(r"neigh|edge|link|graph|communit|adj|similar", k, re.I): suspicious_keys.add(("metadata", k))
    for e in (m.get("entities") or {}).values():
        if isinstance(e, dict): ent_sub.update(e.keys())
    for e in (m.get("events") or {}).values():
        if isinstance(e, dict): ev_sub.update(e.keys())
    for k in ("topic", "entities", "events", "timeline"):
        if not m.get(k): empty[k] += 1
    ids.append(o.get("chunk_id")); srcs.add(s.get("source_id"))
    if f.split("_")[-1].replace(".json", "") not in o.get("chunk_id", ""): fname_mismatch += 1
    emb_dims[len(o.get("embedding") or [])] += 1
    sl = s.get("structural_link"); sl_lens[len(sl) if isinstance(sl, list) else -1] += 1
    if sl and len(sl_examples) < 3: sl_examples.append((o["chunk_id"], sl))
    ctype[o.get("chunk_type")] += 1; stype[s.get("source_type")] += 1
    mm = re.search(r"_chunk_(\d+)_", o.get("chunk_id", "")); chunk_idx[int(mm.group(1)) if mm else -1] += 1
    contentlens.append(len(o.get("content", "")))

n = len(files)
print("files", n, "parse_errors", errors)
print("unique chunk_id", len(set(ids)), "dups", n - len(set(ids)), "unique source_id", len(srcs))
print("chunk_id prefix counts", collections.Counter(i.split("-")[0] for i in ids).most_common(5))
print("filename hash suffix not in chunk_id:", fname_mismatch)
print("top keys", dict(top)); print("source_info keys", dict(si)); print("metadata keys", dict(md))
print("types", {k: dict(v) for k, v in types.items()})
print("entity subkeys", dict(ent_sub)); print("event subkeys", dict(ev_sub))
print("empty/missing metadata", dict(empty))
print("embedding dims", dict(emb_dims))
print("structural_link list lengths", sorted(sl_lens.items())[:15], "examples", sl_examples)
print("chunk_type", dict(ctype), "source_type", dict(stype))
print("chunk index distribution", sorted(chunk_idx.items())[:10])
contentlens.sort(); print("content chars min/median/max", contentlens[0], contentlens[n // 2], contentlens[-1])
print("graph/link-like keys", suspicious_keys)
json.dump(sorted(ids), open(sys.argv[2], "w"))
