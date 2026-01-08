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
    def build_full_sections(self, ranked_nodes: List[dict]) -> List[dict]:
        """
        Превращает ranked text-nodes в список секций-кандидатов для LLM.

        - Группирует text_nodes по section_id.
        - Для каждой секции собирает полный текст (как раньше).
        - Считает агрегированный score (max по узлам секции).
        - Сохраняет node_ids, которые подсветили секцию.

        Формат элемента результата:
        {
            "section_id": str,
            "title": str,
            "text": str,
            "score": float,
            "node_ids": List[str],
        }
        """

        # 1. Группируем узлы по секции
        section_to_nodes: Dict[str, List[dict]] = {}
        for item in ranked_nodes:
            sid = item["section_id"]
            if sid is None:
                continue
            section_to_nodes.setdefault(sid, []).append(item)

        output = []

        # 2. Для каждой секции формируем полный текст и агрегированный score
        for sid, nodes in section_to_nodes.items():
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
                except Exception:
                    return 999_999

            chunks_sorted = sorted(chunks, key=sort_key)
            full_text = "\n".join(t for _, t in chunks_sorted).strip()

            # Агрегированный score: берём максимум по узлам секции
            section_score = float(
                max(node["score"] for node in nodes)
            )

            node_ids = [node["node_id"] for node in nodes]

            output.append({
                "section_id": sid,
                "title": title,
                "text": full_text,
                "score": section_score,
                "node_ids": node_ids,
            })

        # 3. Сортируем секции для стабильности:
        #    сначала по убыванию score, потом по номеру секции
        def section_sort_key(item: dict):
            sid = item["section_id"]
            try:
                order = int(sid.split("ch")[1])
            except Exception:
                order = 999_999
            return (-item["score"], order)

        return sorted(output, key=section_sort_key)

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

        # 5. Детализированный список текстовых узлов (для интерпретации / отладки)
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

        # 6. Секции-кандидаты для LLM-агента (LLM-ready)
        section_candidates = self.build_full_sections(text_context)

        # 7. Графовый контекст (для визуализации / глубокой логики)
        graph_nodes = list(all_nodes)
        graph_edges = [
            {"from": e.from_id, "to": e.to_id, "type": e.relation_type}
            for e in all_edges
            if e.from_id in all_nodes and e.to_id in all_nodes
        ]

        # 8. Итоговый формат, удобный для дальнейшего LLM-агента
        return {
            "query": query,
            "section_candidates": section_candidates,
            "text_nodes": text_context,
            "graph_context": {
                "nodes": graph_nodes,
                "edges": graph_edges,
            },
        }
