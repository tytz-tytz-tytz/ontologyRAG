from src.index.store import load_index
from src.index.embeddings import EmbeddingModel
from src.rag.pipeline import OntologyRAGPipeline

print("=== Load index ===")
sections, text_nodes, graph_adj = load_index("ontology_index.pkl")

print("Sections:", len(sections))
print("Text nodes:", len(text_nodes))

print("=== Init model ===")
model = EmbeddingModel("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

pipeline = OntologyRAGPipeline(
    sections,
    text_nodes,
    graph_adj,
    model,
    max_graph_nodes=200,
    top_k_text=10,
)

print("=== Run query ===")
res = pipeline.run_query("Как работает функциональная структура Maxbot?")

print("Text context:")
for item in res["text_context"]:
    print(" -", item["node_id"], "→", item["text"][:60])

print("Graph nodes:", len(res["graph_context"]["nodes"]))
print("Graph edges:", len(res["graph_context"]["edges"]))
