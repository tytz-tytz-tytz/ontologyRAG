# src/rag/expand.py

from typing import Dict, Set, List
import collections

from ..data.models import Edge


ALLOWED_RELATIONS = {
    "HAS_SUBSECTION",
    "HAS_CHUNK",
    "HAS_ITEM",
    "CAPTIONS",
    "LINKS_TO",
}


class GraphExpander:
    """
    Ограниченный BFS по онтологическому графу
    от множества seed-узлов.
    """

    def __init__(self,
                 graph_adj: Dict[str, List[Edge]],
                 max_depth: int = 4,
                 max_nodes: int = 500):
        self.graph_adj = graph_adj
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    # -------------------------------------------------------------
    # Основной метод
    # -------------------------------------------------------------
    def expand(self, seed_ids: List[str]):
        """
        BFS от seed_ids.

        Возвращает:
            all_nodes: Set[node_id]
            all_edges: List[Edge]
            dist_to_seed: Dict[node_id, int]
        """

        all_nodes: Set[str] = set()
        all_edges: List[Edge] = []
        dist_to_seed: Dict[str, int] = {}

        q = collections.deque()

        # Инициализация очереди
        for sid in seed_ids:
            all_nodes.add(sid)
            dist_to_seed[sid] = 0
            q.append((sid, 0))

        # BFS
        while q and len(all_nodes) < self.max_nodes:
            node, depth = q.popleft()

            if depth >= self.max_depth:
                continue

            for e in self.graph_adj.get(node, []):
                if e.relation_type not in ALLOWED_RELATIONS:
                    continue

                tgt = e.to_id

                # Добавляем вершину и ребро
                all_edges.append(e)

                # Если ты хочешь исключить повторяющиеся ребра — можно делать set,
                # но для RAG это не критично.

                if tgt not in all_nodes:
                    all_nodes.add(tgt)
                    dist_to_seed[tgt] = depth + 1

                    if len(all_nodes) >= self.max_nodes:
                        break

                    q.append((tgt, depth + 1))

        return all_nodes, all_edges, dist_to_seed
