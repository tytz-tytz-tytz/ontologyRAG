# test_loaders_sanity.py

from src.data.loaders import load_ontology, load_json
from src.ontology.hierarchy import build_hierarchy
import collections
import numpy as np


print("\n=== 1. Load raw JSON ===")
raw_nodes = load_json("graphrag_nodes.json")
raw_sections = [n for n in raw_nodes if n["type"] == "Section"]

print("Raw Section nodes in JSON:", len(raw_sections))

sections_with_level = [
    n for n in raw_sections
    if n.get("attributes", {}).get("level") is not None
]
sections_without_level = [
    n for n in raw_sections
    if n.get("attributes", {}).get("level") is None
]

print("Document Sections (with level):", len(sections_with_level))
print("Non-document Sections (no level):", len(sections_without_level))


print("\n=== 2. Load ontology via loaders ===")
sections, text_nodes, graph_adj = load_ontology(
    "graphrag_nodes.json",
    "graphrag_edges.json"
)

print("Sections loaded by loader:", len(sections))
print("Example section IDs:", list(sections.keys())[:10])

# --------------------------------------------------------
# CHECK: All loaded sections MUST have level != None
# --------------------------------------------------------
invalid_sec = [sid for sid, s in sections.items() if s.level is None]
print("Sections with no level inside loader:", invalid_sec)
assert len(invalid_sec) == 0, "ERROR: Non-document Section leaked into sections dict!"


# --------------------------------------------------------
# CHECK: No 'phantom' Section IDs with no level exist inside sections
# --------------------------------------------------------
phantom_ids = [n["id"] for n in sections_without_level]
phantoms_inside = [pid for pid in phantom_ids if pid in sections]

print("Phantom sections present inside loader sections:", phantoms_inside)
assert len(phantoms_inside) == 0, "ERROR: Phantom Sections leaked into sections dict!"


# --------------------------------------------------------
# 3. Check text nodes
# --------------------------------------------------------
print("\n=== 3. Text nodes ===")
print("TextNodes:", len(text_nodes))

sample_tn = next(iter(text_nodes.values()))
print("Example TextNode:", sample_tn.id, sample_tn.node_type, sample_tn.text[:50])


# --------------------------------------------------------
# 4. Check hierarchy reconstruction
# --------------------------------------------------------
print("\n=== 4. Hierarchy ===")
sections, text_nodes = build_hierarchy(sections, text_nodes)

levels = {}
for s in sections.values():
    levels.setdefault(s.level, 0)
    levels[s.level] += 1

print("Levels distribution:", levels)

# debug: sample with children
for s in sections.values():
    if s.children_ids:
        print("Example section with children:", s.id, "->", s.children_ids[:5])
        break

# no orphans
orphans = [
    sid for sid, s in sections.items()
    if s.level != 1 and s.parent_id is None
]
print("Orphan sections:", orphans)
assert len(orphans) == 0, "ERROR: Orphan sections exist!"


# --------------------------------------------------------
# 5. CAPTIONS
# --------------------------------------------------------
print("\n=== 5. Captions ===")
caps = [tn for tn in text_nodes.values() if tn.node_type == "caption"]
print("Found captions:", len(caps))
if caps:
    print("Example caption:", caps[0].id, caps[0].text[:80])

caption_edges = [
    e for edges in graph_adj.values() for e in edges if e.relation_type == "CAPTIONS"
]
print("CAPTIONS edges:", len(caption_edges))
assert len(caption_edges) == len(caps), "Mismatch caption edges vs caption nodes!"


# --------------------------------------------------------
# 6. Graph reachability test
# --------------------------------------------------------
print("\n=== 6. Graph adjacency & reachability ===")
for nid, edges in list(graph_adj.items())[:5]:
    print(nid, "->", [(e.to_id, e.relation_type) for e in edges[:5]])

def bfs(start, depth=2):
    visited = {start}
    q = collections.deque([(start, 0)])
    while q:
        n, d = q.popleft()
        if d >= depth:
            continue
        for e in graph_adj.get(n, []):
            if e.to_id not in visited:
                visited.add(e.to_id)
                q.append((e.to_id, d+1))
    return visited

root = next(iter(sections.keys()))
reachable = bfs(root)
print("Reachable from", root, ":", len(reachable))
assert len(reachable) > 3, "Graph seems disconnected!"


print("\n=== ALL TESTS PASSED ===")
