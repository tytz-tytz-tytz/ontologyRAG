import json
from pathlib import Path

from classic_rag.index.builder import load_index
from classic_rag.rag.heuristic import retrieve_heuristic, HeuristicRAGConfig

INDEX_PATH = Path("artifacts/indexes/classic_rag_index.pkl")
QUERIES_PATH = Path("data/eval/queries.jsonl")
OUT_DIR = Path("artifacts/classic_rag_heuristic_results")

# Keep the same K across methods for fair comparison.
TOP_K = 10

# Heuristics: oversample candidates, drop short chunks, dedup.
CFG = HeuristicRAGConfig(
    top_k=TOP_K,
    candidate_multiplier=6,
    min_chars=80,
    deduplicate=True,
)


def _iter_queries(path: Path):
    """Read JSONL queries: one JSON object per line."""
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
