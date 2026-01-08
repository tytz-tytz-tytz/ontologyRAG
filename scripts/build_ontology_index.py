# build_index.py

from ontology_rag.data.loaders import load_ontology
from ontology_rag.ontology.hierarchy import build_hierarchy
from ontology_rag.index.embeddings import EmbeddingModel
from ontology_rag.index.section_index import SectionIndex
from ontology_rag.index.text_index import TextIndex
from ontology_rag.index.store import save_index


print("=== 1. Load ontology ===")
sections, text_nodes, graph_adj = load_ontology(
    "data/processed/graphrag_nodes.cleaned.json",
    "data/processed/graphrag_edges.cleaned.json"
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
save_index("artifacts/indexes/ontology_index_dir", sections, text_nodes, graph_adj)

print("\n=== DONE. Index saved to artifacts/indexes/ontology_index_dir ===")
