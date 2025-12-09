# test_score_sanity.py

from src.data.loaders import load_ontology
from src.ontology.hierarchy import build_hierarchy
from src.index.embeddings import EmbeddingModel
from src.index.section_index import SectionIndex
from src.index.text_index import TextIndex
from src.rag.drill import DrillSelector, DrillConfig
from src.rag.expand import GraphExpander
from src.rag.score import NodeScorer, ScoreConfig

print("=== 1. Load ontology ===")
sections, text_nodes, graph_adj = load_ontology(
    "graphrag_nodes.json",
    "graphrag_edges.json"
)
sections, text_nodes = build_hierarchy(sections, text_nodes)

print("=== 2. Embeddings ===")
model = EmbeddingModel("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

sec_idx = SectionIndex(model)
sections = sec_idx.compute_section_embeddings(sections)

txt_idx = TextIndex(model)
text_nodes = txt_idx.compute_textnode_embeddings(text_nodes)

print("=== 3. Drill ===")
selector = DrillSelector(sections, DrillConfig())
query = "Как работает функциональная структура Maxbot?"
q_emb = model.encode(query)
seeds = selector.select_seeds(q_emb, top_r=3)

print("Seeds:", seeds)

print("=== 4. Expand ===")
expander = GraphExpander(graph_adj, max_depth=3, max_nodes=200)
all_nodes, all_edges, dist = expander.expand(seeds)
print("Candidate nodes:", len(all_nodes))

print("=== 5. Score ===")
scorer = NodeScorer(sections, text_nodes, ScoreConfig())
ranked = scorer.score_all(
    query_emb=q_emb,
    dist_to_seed=dist,
    candidate_node_ids=list(all_nodes),
    top_k=10,
)

print("Top ranked nodes:")
for nid, score in ranked:
    tn = text_nodes[nid]
    print(f"{nid} ({tn.node_type}, lvl={sections[tn.section_id].level}) → {score:.3f}")
    print("  ", tn.text[:80], "...")
