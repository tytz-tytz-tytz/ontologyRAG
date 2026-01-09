from pathlib import Path
from classic_rag.index.builder import build_index, save_index

NODES = Path("data/processed/graphrag_nodes.cleaned.json")
OUT = Path("artifacts/indexes/classic_rag_index.pkl")


def main() -> None:
    index = build_index(NODES)
    save_index(index, OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
