# src/rag/score.py

from typing import Dict, List, Tuple
import numpy as np

from ..data.models import TextNode, Section


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


class ScoreConfig:
    """
    Конфигурация весов финального скоринга.
    """

    def __init__(
        self,
        w_text: float = 1.0,
        w_type: float = 0.3,
        w_level: float = 0.15,
        w_dist: float = 0.2,
    ):
        self.w_text = w_text
        self.w_type = w_type
        self.w_level = w_level
        self.w_dist = w_dist

        # бонусы за тип
        self.type_bonus = {
            "list_item": 1.0,
            "chunk": 0.6,
            "section_title": 0.8,
            "caption": 0.2,
        }

        # бонусы за уровень раздела (можно подбирать)
        self.level_bonus = {
            1: 0.1,
            2: 0.2,
            3: 0.3,
        }


class NodeScorer:
    """
    Рассчитывает итоговый score для каждого текстового узла
    в candidate_node_ids (узлы из BFS-графа).
    """

    def __init__(
        self,
        sections: Dict[str, Section],
        text_nodes: Dict[str, TextNode],
        config: ScoreConfig,
    ):
        self.sections = sections
        self.text_nodes = text_nodes
        self.cfg = config

    # -------------------------------------------------------------
    # score_one()
    # -------------------------------------------------------------
    def score_one(
        self,
        node_id: str,
        query_emb: np.ndarray,
        dist_to_seed: Dict[str, int],
    ) -> float:

        tn = self.text_nodes.get(node_id)
        if tn is None:
            return -999.0  # узел не текстовый

        # нет embedding → нерелевантно
        sim = cosine_sim(query_emb, tn.embedding)

        # бонус за тип
        bonus_type = self.cfg.type_bonus.get(tn.node_type, 0.0)

        # уровень секции
        sec = self.sections.get(tn.section_id)
        lvl = sec.level if sec else 1
        bonus_level = self.cfg.level_bonus.get(lvl, 0.0)

        # расстояние
        dist = dist_to_seed.get(node_id, 999)

        # итоговый score
        score = (
            self.cfg.w_text * sim
            + self.cfg.w_type * bonus_type
            + self.cfg.w_level * bonus_level
            - self.cfg.w_dist * dist
        )

        return score

    # -------------------------------------------------------------
    # score_all()
    # -------------------------------------------------------------
    def score_all(
        self,
        query_emb: np.ndarray,
        dist_to_seed: Dict[str, int],
        candidate_node_ids: List[str],
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Возвращает top-K узлов по score.
        """

        scored = []
        for nid in candidate_node_ids:
            if nid not in self.text_nodes:
                continue  # не текстовый узел → не ранжируем

            s = self.score_one(nid, query_emb, dist_to_seed)
            scored.append((nid, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
