from pathlib import Path

from bm25_rag.data.loaders import load_id_text_pairs
from bm25_rag.index.builder import build_bm25_index, save_index

NODES_PATH = Path("data/processed/graphrag_nodes.cleaned.json")
OUT_PATH = Path("artifacts/indexes/bm25_index.pkl")

# Standard BM25 parameters
K1 = 1.5
B = 0.75


def main() -> None:
    docs = load_id_text_pairs(NODES_PATH)
    index = build_bm25_index(docs, k1=K1, b=B)
    save_index(index, OUT_PATH)
    print(f"BM25 index saved to: {OUT_PATH}")
    print(f"Docs: {len(index.ids)} | avgdl={index.avgdl:.2f} | vocab={len(index.idf)}")


if __name__ == "__main__":
    main()
