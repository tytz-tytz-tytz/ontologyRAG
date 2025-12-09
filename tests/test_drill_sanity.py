# test_drill_sanity.py

from src.data.loaders import load_ontology
from src.ontology.hierarchy import build_hierarchy
from src.index.embeddings import EmbeddingModel
from src.index.section_index import SectionIndex
from src.rag.drill import DrillSelector, DrillConfig
import numpy as np


print("=== 1. Load ontology ===")
sections, text_nodes, graph_adj = load_ontology(
    "graphrag_nodes.json",
    "graphrag_edges.json"
)
sections, text_nodes = build_hierarchy(sections, text_nodes)


print("\n=== 2. Load embeddings ===")
model = EmbeddingModel("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
sec_index = SectionIndex(model)
sections = sec_index.compute_section_embeddings(sections)


print("\n=== 3. Prepare query ===")
query = "Как работает функциональная структура Maxbot?"
q_emb = model.encode(query)

print("\n=== 4. Drill selection ===")
selector = DrillSelector(sections, DrillConfig())
seeds = selector.select_seeds(q_emb, top_r=3)

print("Seeds:", seeds)
print("Seed count:", len(seeds))

# Simple invariants
assert len(seeds) > 0, "Drill returned no seeds!"
for sid in seeds:
    assert sid in sections, f"Seed {sid} not in sections!"

print("\n=== DRILL TEST PASSED ===")
