# build_index.py

from src.data.loaders import load_ontology
from src.ontology.hierarchy import build_hierarchy
from src.index.embeddings import EmbeddingModel
from src.index.section_index import SectionIndex
from src.index.text_index import TextIndex
from src.index.store import save_index


print("=== 1. Load ontology ===")
sections, text_nodes, graph_adj = load_ontology(
    "graphrag_nodes.json",
    "graphrag_edges.json"
)

print("=== 2. Build hierarchy ===")
build_hierarchy(sections, text_nodes)

print("=== 3. Init embedding model ===")
model = EmbeddingModel(device="cpu")

print("=== 4. Compute section embeddings ===")
sec_index = SectionIndex(model)
sec_index.compute_section_embeddings(sections)

print("=== 5. Compute text node embeddings ===")
txt_index = TextIndex(model)
txt_index.compute_textnode_embeddings(text_nodes)

print("=== 6. Save index ===")
save_index("index", sections, text_nodes, graph_adj)

print("\n=== DONE. Index saved to /index ===")
