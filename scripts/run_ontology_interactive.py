# run_ontology_interactive.py

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ontology_rag.index.store import load_index
from ontology_rag.index.embeddings import EmbeddingModel
from ontology_rag.rag.pipeline import OntologyRAGPipeline


DEFAULT_INDEX_DIR = Path("artifacts/indexes/ontology_index_dir")
DEFAULT_CONFIG_PATH = Path("configs/ontology_rag.json")


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise TypeError("Config must be a JSON object (dict) at the top level.")
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive OntologyRAG runner (offline).")
    p.add_argument(
        "--index_dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Path to the built ontology index directory.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to OntologyRAG config JSON (e.g., configs/ontology_rag.json).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Embedding model device (e.g., cpu, cuda).",
    )
    return p.parse_args()


def run() -> None:
    args = parse_args()

    print("=== Loading offline index ===")
    sections, text_nodes, graph_adj = load_index(str(args.index_dir))

    print("=== Loading config ===")
    config = load_config(args.config)

    print("=== Initializing embedding model ===")
    model = EmbeddingModel(device=args.device)

    pipeline = OntologyRAGPipeline(
        sections=sections,
        text_nodes=text_nodes,
        graph_adj=graph_adj,
        embedding_model=model,
        config=config,
    )

    while True:
        query = input("\nEnter a query (or 'exit'): ").strip()
        if query.lower() in ("exit", "quit"):
            break

        # Run the OntologyRAG pipeline
        result = pipeline.run_query(query)

        # Prepare an "LLM-ready" payload
        llm_input = {
            "query": result["query"],
            "section_candidates": result["section_candidates"],
            # Optionally, you may also include:
            # "text_nodes": result["text_nodes"],
            # "graph_context": result["graph_context"],
        }

        print("\n=== PIPELINE OUTPUT (LLM-ready) ===\n")
        print(json.dumps(llm_input, ensure_ascii=False, indent=2))
        print("\n=== END ===")


if __name__ == "__main__":
    run()
