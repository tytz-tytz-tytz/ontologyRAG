# main.py

import json

from src.index.store import load_index
from src.index.embeddings import EmbeddingModel
from src.rag.pipeline import OntologyRAGPipeline


def run():
    print("=== Загрузка оффлайн-индекса ===")
    sections, text_nodes, graph_adj = load_index("index")

    print("=== Инициализация embedding-модели ===")
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

    while True:
        query = input("\nВведите запрос (или 'exit'): ").strip()
        if query.lower() in ("exit", "quit"):
            break

        # Запускаем RAG-пайплайн
        result = pipeline.run_query(query)

        # Ожидаем, что pipeline.run_query возвращает структуру вида:
        # {
        #   "query": str,
        #   "section_candidates": [ { section_id, title, text, score, node_ids }, ... ],
        #   "text_nodes": [ ... ],
        #   "graph_context": { "nodes": [...], "edges": [...] },
        #   (опционально) "flat_text": ...
        # }

        # Готовим "LLM-ready" формат — то, что дальше пойдёт в OntologyRAG Retriever Agent
        llm_input = {
            "query": result["query"],
            "section_candidates": result["section_candidates"],
            # если хочешь, можно прокинуть и это:
            # "text_nodes": result["text_nodes"],
            # "graph_context": result["graph_context"],
        }

        print("\n=== PIPELINE OUTPUT (LLM-ready) ===\n")
        print(
            json.dumps(
                llm_input,
                ensure_ascii=False,
                indent=2,
            )
        )
        print("\n=== END ===")


if __name__ == "__main__":
    run()
