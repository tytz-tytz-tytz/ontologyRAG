from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Any, List

from ontology_rag.index.store import load_index
from ontology_rag.index.embeddings import EmbeddingModel
from ontology_rag.rag.pipeline import OntologyRAGPipeline


QUERIES_PATH = Path("data/eval/queries.jsonl")
INDEX_DIR = Path("artifacts/indexes/ontology_index_dir")
OUT_DIR = Path("artifacts/ontology_rag_results")


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj or "query" not in obj:
                raise ValueError(
                    f"Line {line_no} must contain 'id' and 'query'"
                )
            yield obj


def extract_text_chunks(result: dict) -> List[str]:
    """
    Extract plain text chunks from OntologyRAG output.
    No aggregation, no reordering, no metadata leakage.
    """
    chunks: List[str] = []

    # Order is preserved as returned by the pipeline
    for key in ("section_candidates", "text_nodes"):
        items = result.get(key, [])
        if not isinstance(items, list):
            continue

        for item in items:
            if isinstance(item, dict):
                # Common field names for textual content
                for text_key in (
                    "text",
                    "content",
                    "chunk",
                    "node_text",
                    "section_text",
                ):
                    value = item.get(text_key)
                    if isinstance(value, str) and value.strip():
                        chunks.append(value.strip())
                        break

    return chunks


def main() -> None:
    if not QUERIES_PATH.exists():
        raise FileNotFoundError(f"Queries file not found: {QUERIES_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline (same setup as interactive script)
    sections, text_nodes, graph_adj = load_index(str(INDEX_DIR))
    model = EmbeddingModel(device="cpu")

    pipeline = OntologyRAGPipeline(
        sections=sections,
        text_nodes=text_nodes,
        graph_adj=graph_adj,
        embedding_model=model,
        max_graph_depth=5,
        max_graph_nodes=800,
        top_k_text=60,
    )

    for item in read_jsonl(QUERIES_PATH):
        qid = str(item["id"])
        query = str(item["query"])

        result = pipeline.run_query(query)

        if not isinstance(result, dict):
            raise TypeError(
                "run_query(query) must return a dict to extract text chunks"
            )

        output_chunks = extract_text_chunks(result)

        out_obj = {
            "id": qid,
            "query": query,
            "output": output_chunks,
        }

        out_path = OUT_DIR / f"{qid}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)

        print(f"[OK] {qid} -> {out_path}")


if __name__ == "__main__":
    main()
