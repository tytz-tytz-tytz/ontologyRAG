# test_expand_sanity.py

from src.data.loaders import load_ontology
from src.ontology.hierarchy import build_hierarchy
from src.index.embeddings import EmbeddingModel
from src.index.section_index import SectionIndex
from src.rag.drill import DrillSelector, DrillConfig
from src.rag.expand import GraphExpander

print("=== 1. Load ontology ===")
sections, text_nodes, graph_adj = load_ontology(
    "graphrag_nodes.json",
    "graphrag_edges.json"
)
sections, text_nodes = build_hierarchy(sections, text_nodes)

print("Sections:", len(sections))
print("Text nodes:", len(text_nodes))


print("\n=== 2. Compute section embeddings ===")
model = EmbeddingModel("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
sec_idx = SectionIndex(model)
sections = sec_idx.compute_section_embeddings(sections)


print("\n=== 3. Select seeds ===")
selector = DrillSelector(sections, DrillConfig())
query = "Как работает функциональная структура Maxbot?"
q_emb = model.encode(query)
seeds = selector.select_seeds(q_emb, top_r=3)

print("Seeds:", seeds)


print("\n=== 4. Expand graph ===")
expander = GraphExpander(graph_adj, max_depth=3, max_nodes=200)
all_nodes, all_edges, dist = expander.expand(seeds)

print("Expanded nodes:", len(all_nodes))
print("Expanded edges:", len(all_edges))

# sanity checks
assert len(all_nodes) > 0, "Expansion returned empty node set!"
assert all(seed in all_nodes for seed in seeds), "Seeds must be in result!"
assert all(d >= 0 for d in dist.values()), "Distances must be >= 0"

print("\n=== EXPAND TEST PASSED ===")
