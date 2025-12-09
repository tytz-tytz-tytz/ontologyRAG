from src.index.store import load_index

sections, text_nodes, graph_adj = load_index("ontology_index.pkl")

print("Sections:", len(sections))
print("Text nodes:", len(text_nodes))
print("Adjacency nodes:", len(graph_adj))

# Быстрый sanity-check
any_emb = next(iter(text_nodes.values())).embedding
print("Sample embedding shape:", None if any_emb is None else any_emb.shape)
