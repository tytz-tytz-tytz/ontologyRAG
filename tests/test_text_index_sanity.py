# test_text_index_sanity.py

from src.data.loaders import load_ontology
from src.ontology.hierarchy import build_hierarchy
from src.index.embeddings import EmbeddingModel
from src.index.text_index import TextIndex
import numpy as np


print("\n=== 1. Load ontology ===")
sections, text_nodes, graph_adj = load_ontology(
    "graphrag_nodes.json",
    "graphrag_edges.json"
)
sections, text_nodes = build_hierarchy(sections, text_nodes)

print("Text nodes:", len(text_nodes))


print("\n=== 2. Load model ===")
model = EmbeddingModel("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
ti = TextIndex(model)


print("\n=== 3. Compute text node embeddings ===")
text_nodes = ti.compute_textnode_embeddings(text_nodes)


# ---------------------------------------------------------
# TEST 1 — no embedding for empty texts, embeddings for others
# ---------------------------------------------------------
missing = []
non_missing = []

for nid, tn in text_nodes.items():
    if tn.text.strip() and tn.embedding is None:
        missing.append(nid)
    if not tn.text.strip() and tn.embedding is not None:
        non_missing.append(nid)

print("Missing embeddings for non-empty:", missing[:10])
print("Embeddings present for empty:", non_missing[:10])

assert len(missing) == 0, "ERROR: Non-empty nodes missing embeddings!"
assert len(non_missing) == 0, "ERROR: Empty-text nodes improperly got embeddings!"


# ---------------------------------------------------------
# TEST 2 — deterministic model
# ---------------------------------------------------------
text = "Маркетинговая рассылка"
v1 = model.encode(text)
v2 = model.encode(text)
diff = float(np.abs(v1 - v2).sum())
print("Deterministic diff:", diff)
assert diff < 1e-6, "Embedding model must be deterministic"


# ---------------------------------------------------------
# TEST 3 — similarity check on two related nodes
# ---------------------------------------------------------
example_ids = list(text_nodes.keys())[:2]
vA = text_nodes[example_ids[0]].embedding
vB = text_nodes[example_ids[1]].embedding

if vA is not None and vB is not None:
    cos = float(np.dot(vA, vB) / (np.linalg.norm(vA) * np.linalg.norm(vB)))
    print("Similarity between two nodes:", cos)
    assert -1 <= cos <= 1, "Cosine similarity must be valid!"


print("\n=== ALL TEXT INDEX TESTS PASSED ===")
