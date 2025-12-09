# src/rag/drill.py

from typing import Dict, List, Optional, Set
import numpy as np

from ..data.models import Section


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Безопасная косинусная близость."""
    if a is None or b is None:
        return -1.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


class DrillConfig:
    """
    Параметры алгоритма drill()

    tau_local  — порог релевантности локального текста
    tau_child  — порог релевантности дочерних subtree
    margin     — насколько local может быть хуже лучшего child
    top_k      — сколько лучших детей спускать
    """

    def __init__(
        self,
        tau_local: float = 0.35,
        tau_child: float = 0.45,
        margin: float = 0.05,
        top_k: int = 2,
    ):
        self.tau_local = tau_local
        self.tau_child = tau_child
        self.margin = margin
        self.top_k = top_k


class DrillSelector:
    """
    Основной класс, реализующий алгоритм выбора seed-секций.
    """

    def __init__(self, sections: Dict[str, Section], config: DrillConfig):
        self.sections = sections
        self.cfg = config

    # -------------------------------------------------------------
    # STEP 1 — Score only by subtree similarity
    # -------------------------------------------------------------
    def rank_l1_sections(self, query_emb: np.ndarray) -> List[Section]:
        """Сортирует секции уровня 1 по sim(query, subtree)."""
        lvl1 = [s for s in self.sections.values() if s.level == 1]
        scored = [
            (s, cosine_sim(query_emb, s.E_subtree))
            for s in lvl1
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, score in scored]

    # -------------------------------------------------------------
    # STEP 2 — Recursive drill
    # -------------------------------------------------------------
    def drill_section(self, sec: Section, query_emb: np.ndarray, seeds: Set[str]):
        """
        Рекурсивный выбор seed-секций.
        Добавляет seed_id в seeds.
        """

        cfg = self.cfg
        children = [self.sections[cid] for cid in sec.children_ids]

        score_local = cosine_sim(query_emb, sec.E_local)
        child_scores = [(c, cosine_sim(query_emb, c.E_subtree)) for c in children]

        score_best_child = max([sc for _, sc in child_scores], default=-1.0)

        # ---------------------------------------------------------
        # CASE 1 — нет локального текста (или пустой)
        # ---------------------------------------------------------
        if not sec.local_text.strip():
            if score_best_child < cfg.tau_child:
                return  # ветка нерелевантна
            # иначе идём в лучших детей
            child_scores.sort(key=lambda x: x[1], reverse=True)
            for c, sc in child_scores[: cfg.top_k]:
                self.drill_section(c, query_emb, seeds)
            return

        # ---------------------------------------------------------
        # CASE 2 — есть локальный текст
        # ---------------------------------------------------------
        cond_seed = (
            score_local >= score_best_child - cfg.margin
            and score_local >= cfg.tau_local
        )

        if cond_seed:
            seeds.add(sec.id)
            return

        # иначе: если subtree релевантно → идём в детей
        if score_best_child >= cfg.tau_child:
            child_scores.sort(key=lambda x: x[1], reverse=True)
            for c, sc in child_scores[: cfg.top_k]:
                self.drill_section(c, query_emb, seeds)

        # если нет — просто завершаем эту ветку
        return

    # -------------------------------------------------------------
    # TOP-LEVEL ENTRY
    # -------------------------------------------------------------
    def select_seeds(self, query_emb: np.ndarray, top_r: int = 3) -> List[str]:
        """
        Полный алгоритм:
        1) ранжируем Level-1 секции
        2) берём top-R веток
        3) запускаем drill()
        4) возвращаем список seed_ids
        """

        lvl1_ranked = self.rank_l1_sections(query_emb)
        roots = lvl1_ranked[:top_r]

        seeds: Set[str] = set()
        for root in roots:
            self.drill_section(root, query_emb, seeds)

        return list(seeds)
