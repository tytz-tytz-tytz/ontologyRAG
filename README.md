# OntologyRAG

This repository contains a research implementation of **OntologyRAG** — an ontology-aware retrieval pipeline — together with multiple baseline retrieval methods (BM25, dense RAG, and heuristic variants) and tooling for **controlled, blind evaluation** using LLM-as-a-judge.

The codebase is designed for **offline experimental comparison** of retrieval strategies under identical input conditions.

---

## Conceptual overview

OntologyRAG differs from classic retrieval-augmented generation (RAG) pipelines in several key aspects:

- Documents are treated as **graph-structured entities** rather than flat text chunks  
- Retrieval operates over **ontology- and graph-level relations**, not only surface text similarity  
- Structural traversal (subtrees, neighbors, distances) is explicitly modeled and controlled  
- Retrieval logic is separated from generation and evaluation concerns  
- Multiple retrieval strategies can be compared under a **shared evaluation protocol**

The goal is to study how structural knowledge (ontology + graph topology) affects retrieval quality compared to standard lexical and dense baselines.

---

## What this repository does

- Preprocesses raw document graph data (nodes and edges)
- Builds multiple retrieval indexes:
  - Ontology-aware index (OntologyRAG)
  - BM25 index
  - Dense (embedding-based) index
- Executes the same query set across all retrieval methods
- Produces **method-agnostic JSON artifacts** suitable for blind evaluation
- Prepares normalized judge payloads for LLM-as-judge experiments

The repository does **not** include:
- serving or API code
- user interfaces
- online systems
- training or fine-tuning procedures

All pipelines are executed **offline**.

---

## Repository structure

```
ontologyRAG/
├── artifacts/                     # Generated artifacts (not committed)
│   ├── bm25_rag_results/           # BM25 retrieval outputs
│   ├── bm25_rag_heuristic_results/ # BM25 + heuristics outputs
│   ├── classic_rag_results/        # Classic dense RAG outputs
│   ├── classic_rag_heuristic_results/
│   ├── ontology_rag_results/       # OntologyRAG outputs
│   ├── indexes/                    # Serialized retrieval indexes
│   │   ├── bm25_index.pkl
│   │   ├── classic_rag_index.pkl
│   │   ├── ontology_index.pkl
│   │   └── ontology_index_dir/
│   ├── judge_payloads/             # Prepared inputs for LLM-as-judge
│   └── reports/                    # Aggregated evaluation reports
│
├── configs/
│   └── judge_prep.json              # Configuration for judge payload generation
│
├── data/
│   ├── raw/                         # Raw graph data
│   │   ├── graphrag_nodes.json
│   │   └── graphrag_edges.json
│   ├── processed/                   # Cleaned and normalized graph data
│   │   ├── graphrag_nodes.cleaned.json
│   │   └── graphrag_edges.cleaned.json
│   └── eval/
│       └── queries.jsonl            # Fixed evaluation query set
│
├── scripts/                         # Entry-point scripts (offline pipelines)
│   ├── preprocess_graph_data.py
│   ├── build_bm25_index.py
│   ├── build_classic_rag_index.py
│   ├── build_ontology_index.py
│   ├── run_queries_bm25.py
│   ├── run_queries_bm25_heuristic.py
│   ├── run_queries_classic_rag.py
│   ├── run_queries_classic_rag_heuristic.py
│   ├── run_queries_ontology.py
│   ├── run_ontology_interactive.py
│   └── build_judge_payloads.py
│
├── src/
│   ├── bm25_rag/                    # BM25 retrieval implementation
│   ├── classic_rag/                 # Dense vector RAG implementation
│   ├── ontology_rag/                # OntologyRAG (main method)
│   │   ├── ontology/                # Ontology structures and relations
│   │   ├── index/                   # Ontology-aware index logic
│   │   └── rag/                     # Retrieval pipeline
│   └── judge_prep/                  # Cleaning & truncation logic for evaluation
│       └── clean_cap.py
│
├── pyproject.toml
├── README.md
└── .gitignore
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
- removes non-informative text fragments
- normalizes text
- removes dangling or invalid edges

Cleaned data is written to:

```
data/processed/
```

All downstream steps must use the processed data.

---

## Retrieval methods

The following retrieval pipelines are implemented:

- **OntologyRAG** — ontology- and graph-aware retrieval (main proposed method)
- **BM25** — sparse lexical baseline
- **BM25 + heuristics** — BM25 with structural post-filtering
- **Classic dense RAG** — embedding-based similarity retrieval
- **Classic dense RAG + heuristics** — dense retrieval with structural heuristics

All methods:
- operate on the same corpus
- use the same query set
- differ *only* in retrieval strategy

---

## Running queries (batch mode)

Each retrieval method has a dedicated script, for example:

```bash
python scripts/run_queries_ontology.py
python scripts/run_queries_bm25.py
python scripts/run_queries_classic_rag.py
```

- Queries are read from `data/eval/queries.jsonl`
- One JSON file is written per query
- Results are stored in method-specific folders under `artifacts/`

Example output format:

```json
{
  "id": "Q001",
  "query": "...",
  "output": [
    "...retrieved text fragment 1...",
    "...retrieved text fragment 2..."
  ]
}
```

The output contains **only retrieved text**, without method identifiers or scores.

---

## Output artifacts

Each retrieval method produces comparable artifacts:

- One JSON file per query
- Each file contains:
  - query identifier
  - query text
  - ordered list of retrieved text fragments

These artifacts are the **unit of comparison** in evaluation.

For LLM-as-judge experiments, cleaned and normalized payloads are generated using:

```bash
python scripts/build_judge_payloads.py
```

Judge-ready files are written to:

```
artifacts/judge_payloads/
```

Each payload:
- contains multiple anonymized retrieval outputs (A, B, C, …)
- uses a fixed token budget per method
- includes a private mapping for result decoding

---

## Reproducibility notes

- All retrieval pipelines are **deterministic** given fixed inputs and configurations
- No stochastic components (sampling, randomness) are used during retrieval
- Results do not depend on execution order
- Randomization is applied **only** at the judge-payload construction stage for blind evaluation

This design enables reproducible comparison across methods.

---

## Interactive mode (optional)

For manual inspection of OntologyRAG behavior:

```bash
python scripts/run_ontology_interactive.py
```
