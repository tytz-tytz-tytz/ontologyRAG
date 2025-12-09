# test_section_index_sanity.py

from src.data.loaders import load_ontology
from src.ontology.hierarchy import build_hierarchy
from src.index.embeddings import EmbeddingModel
from src.index.section_index import SectionIndex

import numpy as np


print("\n=== 1. Load ontology ===")
sections, text_nodes, graph_adj = load_ontology(
    "graphrag_nodes.json",
    "graphrag_edges.json"
)

sections, text_nodes = build_hierarchy(sections, text_nodes)


print("\n=== 2. Load embedding model ===")
model = EmbeddingModel("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
index = SectionIndex(model)


print("\n=== 3. Compute embeddings ===")
sections = index.compute_section_embeddings(sections)


# ---------------------------------------------------------
# TEST 1 — no missing embeddings for subtree
# ---------------------------------------------------------
missing_subtree = [sid for sid, s in sections.items() if s.subtree_text.strip() and s.E_subtree is None]
print("Missing subtree embeddings:", missing_subtree)
assert len(missing_subtree) == 0, "ERROR: subtree embeddings missing!"


# ---------------------------------------------------------
# TEST 2 — subtree and local embedding similarity sanity
# ---------------------------------------------------------
def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

example = next(iter(sections.values()))
if example.E_local is not None:
    sim = cos_sim(example.E_local, example.E_subtree)
    print("Similarity(local, subtree):", sim)
    assert sim > 0.3, "Local and subtree should have moderate similarity"


# ---------------------------------------------------------
# TEST 3 — embedding consistency: same text → same vector
# ---------------------------------------------------------
text = "Привет мир"
v1 = model.encode(text)
v2 = model.encode(text)
diff = np.abs(v1 - v2).sum()

print("Identical text diff:", diff)
assert diff < 1e-6, "Embedding model should be deterministic"


print("\n=== ALL SECTION INDEX TESTS PASSED ===")
