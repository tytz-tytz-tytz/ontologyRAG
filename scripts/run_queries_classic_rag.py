import json
from pathlib import Path

from classic_rag.index.builder import load_index
from classic_rag.rag.retrieve import retrieve

INDEX_PATH = Path("artifacts/indexes/classic_rag_index.pkl")
QUERIES_PATH = Path("data/eval/queries.jsonl")
OUT_DIR = Path("artifacts/classic_rag_results")
TOP_K = 10


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = load_index(INDEX_PATH)

    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)

            out = {
                "id": q["id"],
                "query": q["query"],
                "output": retrieve(index, q["query"], TOP_K),
            }

            (OUT_DIR / f"{q['id']}.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print(f"Done. Results: {OUT_DIR}")


if __name__ == "__main__":
    main()
