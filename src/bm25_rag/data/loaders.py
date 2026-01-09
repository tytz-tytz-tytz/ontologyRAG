import json
from pathlib import Path
from typing import List, Tuple


def load_id_text_pairs(nodes_path: Path) -> List[Tuple[str, str]]:
    """
    Load nodes from graphrag_nodes.cleaned.json and return (id, text) pairs.
    Only uses 'id' and 'text' fields.
    """
    data = json.loads(nodes_path.read_text(encoding="utf-8"))
    out: List[Tuple[str, str]] = []

    for obj in data:
        cid = obj.get("id")
        text = obj.get("text", "")
        if not cid or not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        out.append((cid, text))

    return out
