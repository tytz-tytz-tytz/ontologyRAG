# OntologyRAG

This repository contains a research implementation of an ontology-based retrieval pipeline (OntologyRAG) and supporting scripts for data preprocessing and query execution.

The codebase is intended for controlled experimental evaluation and comparison with other retrieval approaches (e.g., classic RAG, graph-based RAG) under identical input conditions.

---

## What this repository does

- Cleans raw document graph data (nodes and edges)
- Builds an ontology-aware retrieval index from cleaned data
- Executes a fixed set of queries against the index
- Saves retrieval outputs in a neutral JSON format suitable for blind evaluation

The repository does **not** include:
- serving or API code
- user interfaces
- online systems
- training procedures

All pipelines are executed offline.

---

## Repository structure

```
ontologyRAG/
├── artifacts/
│   ├── indexes/
│   │   ├── ontology_index_dir/
│   │   └── ontology_index.pkl
│   ├── ontology_rag_results/
│   └── reports/
├── configs/
├── data/
│   ├── raw/
│   │   ├── graphrag_nodes.json
│   │   └── graphrag_edges.json
│   ├── processed/
│   │   ├── graphrag_nodes.cleaned.json
│   │   └── graphrag_edges.cleaned.json
│   └── eval/
│       └── queries.jsonl
├── scripts/
│   ├── preprocess_graph_data.py
│   ├── build_ontology_index.py
│   ├── run_ontology_interactive.py
│   └── run_queries_ontology.py
├── src/
│   └── ontology_rag/
├── tests/
├── pyproject.toml
└── README.md
```

---

## Setup

### Python version
- Python 3.12

### Installation

Create and activate a virtual environment, then install the project in editable mode:

```bash
pip install -e .
```

All dependencies are specified in `pyproject.toml`.

---

## Data preprocessing

Clean raw graph data before building any index:

```bash
python scripts/preprocess_graph_data.py
```

This step:
- removes non-informative text fragments (e.g., page numbers, punctuation-only nodes)
- normalizes text
- removes dangling edges

Cleaned data is written to:

```
data/processed/
```

All downstream steps must use the processed data.

---

## Building the ontology index

Build the ontology-aware retrieval index:

```bash
python scripts/build_ontology_index.py
```

The index is stored in:

```
artifacts/indexes/ontology_index_dir/
```

---

## Running queries (batch mode)

Execute a fixed set of evaluation queries:

```bash
python scripts/run_queries_ontology.py
```

- Queries are read from `data/eval/queries.jsonl`
- One JSON file is written per query
- Results are stored in:

```
artifacts/ontology_rag_results/
```

Each result file has the format:

```json
{
  "id": "Q001",
  "query": "...",
  "output": [
    "...text fragment 1...",
    "...text fragment 2..."
  ]
}
```

The output contains **only retrieved text fragments**, without metadata or method identifiers.

---

## Interactive mode (optional)

For manual inspection:

```bash
python scripts/run_ontology_interactive.py
```