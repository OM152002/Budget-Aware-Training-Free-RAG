"""Audit table_chunks_with_metadata.json."""
import json, sys, collections

o = json.load(open(sys.argv[1]))
print("top type", type(o).__name__, "len", len(o))
items = o if isinstance(o, list) else list(o.values())
if isinstance(o, dict): print("dict keys sample", list(o)[:3])
top = collections.Counter(); md = collections.Counter(); types = collections.defaultdict(collections.Counter)
ids = []
for it in items:
    top.update(it.keys())
    for k, v in it.items(): types[k][type(v).__name__] += 1
    m = it.get("metadata") or {}
    md.update(m.keys())
    for k, v in m.items(): types["metadata." + k][type(v).__name__] += 1
    ids.append(it.get("chunk_id"))
print("top keys", dict(top)); print("metadata keys", dict(md))
print("types", {k: dict(v) for k, v in types.items()})
print("unique ids", len(set(ids)), "examples", ids[:3])
ex = items[0]
print(json.dumps({k: (v[:300] if isinstance(v, str) else v) for k, v in ex.items() if k != "metadata"}, ensure_ascii=False)[:1500])
print(json.dumps(ex.get("metadata"), ensure_ascii=False)[:1500])
json.dump(ids, open(sys.argv[2], "w"))
