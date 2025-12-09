# src/rag/pipeline.py

from typing import Dict, List
import numpy as np

from ..data.models import TextNode, Section, Edge
from ..index.embeddings import EmbeddingModel
from .drill import DrillSelector, DrillConfig
from .expand import GraphExpander
from .score import NodeScorer, ScoreConfig


class OntologyRAGPipeline:
    """
    ONLINE RAG-пайплайн.
    Работает с предварительно построенным оффлайн-индексом.
    """

    def __init__(
        self,
        sections: Dict[str, Section],
        text_nodes: Dict[str, TextNode],
        graph_adj: Dict[str, List[Edge]],
        embedding_model: EmbeddingModel,
        drill_cfg: DrillConfig = DrillConfig(),
        score_cfg: ScoreConfig = ScoreConfig(),
        max_graph_depth: int = 3,
        max_graph_nodes: int = 200,
        top_k_text: int = 20,
    ):
        self.sections = sections
        self.text_nodes = text_nodes
        self.graph_adj = graph_adj
        self.model = embedding_model

        self.drill_cfg = drill_cfg
        self.score_cfg = score_cfg
        self.max_graph_depth = max_graph_depth
        self.max_graph_nodes = max_graph_nodes
        self.top_k_text = top_k_text

    # =============================================================
    # FULL SECTION MODE (LLM-ready)
    # =============================================================
    def build_full_sections(self, ranked_nodes: List[dict]):
        """
        Если секция содержит хотя бы один релевантный узел —
        мы берём ВЕСЬ текст секции, последовательно.
        """

        # 1. Определяем релевантные секции
        relevant_sections = set()
        for item in ranked_nodes:
            sid = item["section_id"]
            if sid is not None:
                relevant_sections.add(sid)

        output = []

        # 2. Для каждой секции формируем полный текст
        for sid in relevant_sections:
            sec = self.sections[sid]

            # Заголовок секции
            title = ""
            if sec.local_text:
                title = sec.local_text.split("\n")[0].strip()

            # Собрать все chunk-и этой секции
            chunks = []
            for nid, tn in self.text_nodes.items():
                if tn.section_id == sid:
                    chunks.append((nid, tn.text))

            # Сортировка по реальному порядку chunk_chXXXX
            def sort_key(x):
                nid = x[0]
                try:
                    return int(nid.split("ch")[1])
                except:
                    return 999999

            chunks_sorted = sorted(chunks, key=sort_key)
            full_text = "\n".join(t for _, t in chunks_sorted).strip()

            output.append({
                "section_id": sid,
                "title": title,
                "text": full_text,
            })

        # Стабильная сортировка секций по номеру
        return sorted(
            output,
            key=lambda x: int(x["section_id"].split("ch")[1])
        )

    # =============================================================
    # MAIN PIPELINE METHOD
    # =============================================================
    def run_query(self, query: str) -> Dict:
        # 1. Embed query
        q_emb = self.model.encode(query)

        # 2. Drill: choose seed sections
        selector = DrillSelector(self.sections, self.drill_cfg)
        seed_ids = selector.select_seeds(q_emb, top_r=3)

        # 3. Expand graph (BFS)
        expander = GraphExpander(
            self.graph_adj,
            max_depth=self.max_graph_depth,
            max_nodes=self.max_graph_nodes,
        )
        all_nodes, all_edges, dist = expander.expand(seed_ids)

        # 4. Score text nodes
        scorer = NodeScorer(self.sections, self.text_nodes, self.score_cfg)
        ranked = scorer.score_all(
            query_emb=q_emb,
            dist_to_seed=dist,
            candidate_node_ids=list(all_nodes),
            top_k=self.top_k_text,
        )

        # 5. Build detail list for debug / interpretability
        text_context = []
        for nid, score in ranked:
            tn = self.text_nodes[nid]
            text_context.append({
                "node_id": nid,
                "section_id": tn.section_id,
                "type": tn.node_type,
                "text": tn.text,
                "score": float(score),
            })

        # 6. NEW: full-section extraction
        flat_text = self.build_full_sections(text_context)

        # 7. Graph context (optional, for visualization)
        graph_nodes = list(all_nodes)
        graph_edges = [
            {"from": e.from_id, "to": e.to_id, "type": e.relation_type}
            for e in all_edges
            if e.from_id in all_nodes and e.to_id in all_nodes
        ]

        return {
            "text_context": text_context,
            "flat_text": flat_text,
            "graph_context": {
                "nodes": graph_nodes,
                "edges": graph_edges,
            },
        }
