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
- Builds **LLM-as-a-judge prompts** in a fixed, strict format
- Aggregates and analyzes judge outputs according to predefined metrics

The repository does **not** include:
- serving or API code
- user interfaces
- online systems
- training or fine-tuning procedures
- scripts for directly calling LLM APIs

All pipelines are executed **offline**.  
LLM inference (judge calls) is expected to be performed externally.

---

## Repository structure

```text
ontologyRAG/
├── artifacts/                      # Generated artifacts (not committed)
│   ├── bm25_rag_results/            # BM25 retrieval outputs
│   ├── bm25_rag_heuristic_results/  # BM25 + heuristics outputs
│   ├── classic_rag_results/         # Classic dense RAG outputs
│   ├── classic_rag_heuristic_results/
│   ├── ontology_rag_results/        # OntologyRAG outputs
│   ├── indexes/                     # Serialized retrieval indexes
│   │   ├── bm25_index.pkl
│   │   ├── classic_rag_index.pkl
│   │   ├── ontology_index.pkl
│   │   └── ontology_index_dir/
│   ├── judge_payloads/              # Anonymized retrieval results per query
│   ├── judge_prompts/               # Fully built LLM-as-judge prompts
│   ├── judge_outputs/               # Raw LLM judge responses (external)
│   └── reports/                     # Aggregated evaluation reports
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
│   ├── build_judge_payloads.py
│   ├── build_judge_prompts.py
│   └── build_judge_reports.py
│
├── src/
│   ├── bm25_rag/                    # BM25 retrieval implementation
│   ├── classic_rag/                 # Dense vector RAG implementation
│   ├── ontology_rag/                # OntologyRAG (main method)
│   │   ├── ontology/                # Ontology structures and relations
│   │   ├── index/                   # Ontology-aware index logic
│   │   └── rag/                     # Retrieval pipeline
│   └── judge_prep/                  # Prompt building & parsing logic
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

```text
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
- differ only in retrieval strategy

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

The output contains only retrieved text, without method identifiers or scores.

---

## LLM-as-judge evaluation pipeline

### Judge payloads

For blind evaluation, retrieval outputs from different methods are merged and anonymized:

```bash
python scripts/build_judge_payloads.py
```

Outputs are written to:

```text
artifacts/judge_payloads/
```

Each payload:
- contains the user query
- includes multiple candidate contexts labeled A–E
- includes a private mapping from letters to retrieval methods
- is method-agnostic and suitable for blind judging

### Judge prompts

LLM-as-judge prompts are built from payloads using a fixed instruction template:

```bash
python scripts/build_judge_prompts.py
```

Prompts can be exported as:
- Markdown (`.md`)
- Chat-style messages (`.messages.json`)
- JSONL for batch inference

The prompt specifies:
- evaluation task and constraints
- required metrics (relevance, answerability, noise, overall)
- strict JSON output schema

**Important:**  
This repository does not perform LLM inference.  
Users are expected to submit prompts to an external LLM and save raw responses manually.

### Judge outputs (external)

Raw LLM responses must be saved to:

```text
artifacts/judge_outputs/<model_name>/Qxxx_<replica>.json
```

Each file is expected to contain a JSON object with a `judge_response` field
matching the required schema defined in the prompt.

Multiple replicas per query are supported.

### Aggregation and reports

Judge outputs are parsed and aggregated using:

```bash
python scripts/build_judge_reports.py \
  --judge_outputs_dir artifacts/judge_outputs \
  --judge_payloads_dir artifacts/judge_payloads \
  --reports_dir artifacts/reports
```

This produces multiple analysis-ready tables, including:
- per-candidate metric scores (long format)
- per-run judge decisions
- aggregated method-level summaries
- winner statistics and confidence measures

All reports are written to:

```text
artifacts/reports/
```

---

## Reproducibility notes

- All retrieval pipelines are deterministic given fixed inputs
- No stochastic components are used during retrieval
- Randomization is applied only during:
  - candidate shuffling in judge payloads
  - external LLM judge inference
- Multiple judge replicas are supported and explicitly tracked

This design enables controlled and reproducible comparison across retrieval methods.

---

## Interactive mode (optional)

For manual inspection of OntologyRAG behavior:

```bash
python scripts/run_ontology_interactive.py
```
