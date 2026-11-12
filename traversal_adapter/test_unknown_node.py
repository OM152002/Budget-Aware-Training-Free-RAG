"""Standalone adapter test (PRD §32 "nonexistent node handling").

An ID that is not in the reconstructed graph must raise UnknownNodeError from node() and get_neighbors(),
and must not be confused with a real isolated node (which returns an empty neighbor list).

Usage: python -P -s test_unknown_node.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_seed_adapter import GraphSeedAdapter, UnknownNodeError  # noqa: E402


def expect_raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception as e:  # wrong exception type
        print(f"  unexpected {type(e).__name__}: {e}")
        return False
    return False


def main():
    A = GraphSeedAdapter()
    fake = "ottaqa-NOT_A_REAL_DOC_chunk_0_000000000000"
    real = A.question_ids and A.questions[A.question_ids[0]]["ranking"][0]
    isolated = next(n for n, v in A.adj.items() if not v)
    results = {
        "in_graph(unknown) is False": A.in_graph(fake) is False,
        "node(unknown) raises UnknownNodeError": expect_raises(lambda: A.node(fake), UnknownNodeError),
        "UnknownNodeError is a KeyError": issubclass(UnknownNodeError, KeyError),
        "get_neighbors([unknown]) raises": expect_raises(lambda: A.get_neighbors([fake]), UnknownNodeError),
        "get_neighbors([real, unknown]) raises": expect_raises(lambda: A.get_neighbors([real, fake]), UnknownNodeError),
        "get_neighbors(['']) raises": expect_raises(lambda: A.get_neighbors([""]), UnknownNodeError),
        "isolated real node: node() ok": A.node(isolated).degree == 0,
        "isolated real node: get_neighbors -> []": A.get_neighbors([isolated]) == [],
        "real node still works": len(A.get_neighbors([real])) == len(A.adj[real]),
    }
    for k, v in results.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    ok = all(results.values())
    print("ALL_PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
