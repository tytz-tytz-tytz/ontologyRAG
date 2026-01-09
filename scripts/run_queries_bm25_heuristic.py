import json
from pathlib import Path

from bm25_rag.index.builder import load_index
from bm25_rag.rag.heuristic import retrieve_heuristic, BM25HeuristicConfig

INDEX_PATH = Path("artifacts/indexes/bm25_index.pkl")
QUERIES_PATH = Path("data/eval/queries.jsonl")
OUT_DIR = Path("artifacts/bm25_rag_heuristic_results")


CFG = BM25HeuristicConfig(top_k=10)


def _iter_queries(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = load_index(INDEX_PATH)

    for q in _iter_queries(QUERIES_PATH):
        qid = q["id"]
        query = q["query"]

        out = {
            "id": qid,
            "query": query,
            "output": retrieve_heuristic(index, query, CFG),
        }

        (OUT_DIR / f"{qid}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Done. Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
