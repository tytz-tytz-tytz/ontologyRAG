from src.data.loaders import load_ontology
from src.ontology.hierarchy import build_hierarchy, get_root_sections


print("\n=== 1. Load ontology ===")
sections, text_nodes, graph_adj = load_ontology(
    "graphrag_nodes.json",
    "graphrag_edges.json"
)

print("Sections:", len(sections))
print("TextNodes:", len(text_nodes))

# baseline sanity
assert len(sections) > 0, "No sections loaded!"
assert len(text_nodes) > 0, "No text nodes loaded!"


print("\n=== 2. Build hierarchy ===")
sections, text_nodes = build_hierarchy(sections, text_nodes)

# Quick distribution check
levels = {}
for s in sections.values():
    levels.setdefault(s.level, 0)
    levels[s.level] += 1

print("Levels:", levels)

root_sections = get_root_sections(sections)
print("Root sections:", [s.id for s in root_sections[:5]], "... total:", len(root_sections))
assert len(root_sections) > 0, "No level-1 root sections detected!"


print("\n=== 3. Test local_text ===")
# Find first non-empty local text section
sample_local = None
for s in sections.values():
    if s.local_text:
        sample_local = s
        break

assert sample_local is not None, "All local_texts appear empty — unexpected!"
print("Example local section:", sample_local.id)
print("Local text sample:", sample_local.local_text[:120].replace("\n", " ") + "...")


print("\n=== 4. Test subtree_text correctness ===")
# Subtree text must be >= local text length
for sid, s in sections.items():
    if s.children_ids:
        assert len(s.subtree_text) >= len(s.local_text), f"Subtree text is smaller than local text for {sid}"

print("Subtree text length check: OK")

# Check that subtree contains child local text
example_parent = None
example_child = None

for s in sections.values():
    if s.children_ids:
        example_parent = s
        example_child = sections[s.children_ids[0]]
        break

assert example_parent, "No parent-child example found!"

child_snippet = example_child.local_text[:50].strip()
print("Child snippet:", child_snippet)

if child_snippet:
    found = child_snippet in example_parent.subtree_text
    print("Found in subtree:", found)
    assert found, f"Child text not found in parent's subtree for {example_parent.id}"


print("\n=== 5. Ensure no empty subtree text for roots ===")
for r in root_sections:
    assert r.subtree_text.strip(), f"Root section {r.id} has empty subtree_text!"

print("Root subtree text check: OK")


print("\n=== ALL HIERARCHY TESTS PASSED ===")
