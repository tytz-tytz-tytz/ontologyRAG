# main.py

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

        result = pipeline.run_query(query)

        print("\n=== FLAT TEXT (LLM-ready) ===\n")
        for sec in result["flat_text"]:
            print(f"### {sec['title']}")
            print(sec["text"])
            print("\n")

        print("=== END ===")


if __name__ == "__main__":
    run()
