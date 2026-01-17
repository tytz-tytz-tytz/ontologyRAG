# OntologyRAG

This repository contains a research implementation of **OntologyRAG** — an ontology-aware retrieval pipeline — together with multiple baseline retrieval methods (BM25, dense RAG, and heuristic variants) and tooling for **controlled, blind evaluation** using LLM-as-a-judge.

The codebase is designed for **offline experimental comparison** of retrieval strategies under identical input conditions, with a strong focus on **ablation-based analysis** of ontology-aware retrieval components.

All pipelines are executed **offline**.  
LLM inference (judge calls) is expected to be performed externally.

---

## Conceptual overview

OntologyRAG differs from classic retrieval-augmented generation (RAG) pipelines in several key aspects:

- Documents are treated as **graph-structured entities** rather than flat text chunks
- Retrieval operates over **ontology- and graph-level relations**, not only surface text similarity
- Structural traversal (subtrees, neighbors, distances) is explicitly modeled and parameterized
- Retrieval logic is separated from generation and evaluation concerns
- All structural assumptions are exposed via **configuration files** and validated via systematic ablations

The primary research goal is to study **how and when structural knowledge (ontology + graph topology) improves retrieval quality**, compared to purely lexical or dense baselines.

---

## What this repository does

- Preprocesses raw document graph data (nodes and edges)
- Builds multiple retrieval indexes:
  - Ontology-aware index (OntologyRAG)
  - BM25 index
  - Dense (embedding-based) index
- Executes a fixed query set across all retrieval methods
- Produces **method-agnostic JSON artifacts** suitable for blind evaluation
- Supports **systematic ablations** of OntologyRAG via configuration variants
- Builds **LLM-as-a-judge prompts** in a strict, reproducible format
- Aggregates and analyzes judge outputs for **multi-way** comparisons and **diagnostic pairwise** comparisons

The repository does **not** include:
- serving or API code
- user interfaces
- online systems
- training or fine-tuning procedures
- scripts for directly calling LLM APIs

---

## High-level pipeline (ASCII diagram)

```text
                 ┌────────────────────┐
                 │  Raw graph data     │
                 │  (nodes, edges)     │
                 │  data/raw/          │
                 └─────────┬──────────┘
                           │
                           ▼
            ┌────────────────────────────┐
            │ Preprocessing               │
            │ scripts/preprocess_*.py    │
            └─────────┬──────────────────┘
                      │
                      ▼
        ┌────────────────────────────────────┐
        │ Cleaned graph data                  │
        │ data/processed/                     │
        └─────────┬──────────────────────────┘
                  │
                  ▼
     ┌───────────────────────────────────────────┐
     │ Index construction                         │
     │  - BM25                                   │
     │  - Dense (classic RAG)                    │
     │  - OntologyRAG                            │
     │ scripts/build_*_index.py                  │
     └─────────┬─────────────────────────────────┘
               │
               ▼
 ┌────────────────────────────────────────────────────┐
 │ Retrieval runs (same queries, different methods)    │
 │ scripts/run_queries_*.py                            │
 └─────────┬──────────────────────────────────────────┘
           │
           ▼
 ┌────────────────────────────────────────────────────┐
 │ Method-agnostic retrieval outputs                   │
 │ artifacts/*_results/Qxxx.json                       │
 └─────────┬──────────────────────────────────────────┘
           │
           ▼
 ┌────────────────────────────────────────────────────┐
 │ Judge payload construction (blind merging)          │
 │ scripts/build_judge_payloads.py                     │
 └─────────┬──────────────────────────────────────────┘
           │
           ▼
 ┌────────────────────────────────────────────────────┐
 │ LLM-as-a-judge prompts                              │
 │ scripts/build_judge_prompts.py                      │
 └─────────┬──────────────────────────────────────────┘
           │
           ▼
 ┌────────────────────────────────────────────────────┐
 │ External LLM inference (manual)                     │
 │ artifacts/judge_outputs/                            │
 └─────────┬──────────────────────────────────────────┘
           │
           ▼
 ┌────────────────────────────────────────────────────┐
 │ Aggregation & reports                               │
 │ scripts/build_judge_reports.py                      │
 │ artifacts/reports/                                 │
 └────────────────────────────────────────────────────┘
```

---

## Repository structure

```text
ontologyRAG/
├── artifacts/                      # Generated artifacts (not committed)
│   ├── bm25_rag_results/
│   ├── bm25_rag_heuristic_results/
│   ├── classic_rag_results/
│   ├── classic_rag_heuristic_results/
│   ├── ontology_rag_results/
│   │   └── param_experiments/       # OntologyRAG ablation runs
│   ├── indexes/
│   ├── judge_payloads/              # Blind judge payloads
│   │   ├── rag_5way/
│   │   └── ablation_pairs/
│   ├── judge_prompts/
│   ├── judge_outputs/
│   │   ├── rag_5way/
│   │   └── ablation_pairs/
│   └── reports/
│       ├── rag_5way/
│       └── ablation_pairs/
│
├── configs/
│   ├── judge_prep/                  # Judge preparation configs
│   └── ablations/                   # OntologyRAG ablation configs
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── eval/
│       └── queries.jsonl
│
├── scripts/
├── src/
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## Setup

### Python version
- Python 3.12

### Installation

```bash
pip install -e .
```

---

## End-to-end experimental workflow

This section describes the **complete, linear experimental protocol** required to reproduce results produced by this repository.

All steps must be executed **in the order specified below**.

---

### Step 1 — Data preprocessing

```bash
python scripts/preprocess_graph_data.py
```

---

### Step 2 — Build retrieval indexes

```bash
python scripts/build_bm25_index.py
python scripts/build_classic_rag_index.py
python scripts/build_ontology_index.py
```

Indexes are written to:

```text
artifacts/indexes/
```

---

### Step 3 — Run retrieval pipelines

```bash
python scripts/run_queries_ontology.py --config configs/ablations/stable_baseline.json
python scripts/run_queries_bm25.py
python scripts/run_queries_bm25_heuristic.py
python scripts/run_queries_classic_rag.py
python scripts/run_queries_classic_rag_heuristic.py
```

Each run:
- reads queries from `data/eval/queries.jsonl`
- writes one JSON file per query
- stores results under `artifacts/`

---

### Retrieval output format

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

## LLM-as-a-judge evaluation

### Evaluation modes (important)

This repository distinguishes **two evaluation modes**:

1. **Multi-way evaluation (A–E)**  
   Fully implemented end-to-end. Used for comparing different retrieval methods
   (e.g., OntologyRAG vs BM25 vs dense RAG).

2. **Pairwise evaluation (A vs B)**  
   Used for **diagnostic validation of OntologyRAG ablations**.  
   Pairwise judging validates *retrieval regimes* (e.g. structural vs collapsed),
   not parameter tuning or configuration search.  
   Aggregation and reporting for pairwise evaluation are intended for diagnostic analysis and hypothesis validation, rather than final leaderboard-style comparison.


#### Pairwise evaluation semantics (important)
In pairwise ablation evaluation (A vs B), the following outcomes are explicitly distinguished and handled during aggregation:
- **A wins**: the judge explicitly prefers candidate A.
- **B wins**: the judge explicitly prefers candidate B.
- **Tie / no preference**: the judge determines that both contexts are equally useful and does not select a winner.
- **Skipped identical**: the judge run is intentionally skipped when both retrieved
contexts are textually identical.

Tie decisions and skipped-identical cases are **not treated as errors** and **do not contribute a win** to either method. They are preserved in reports to avoid artificially inflating win rates in ablation comparisons.

This design ensures that ablation experiments measure *genuine retrieval differences*, rather than forcing arbitrary preferences when evidence is insufficient.

These semantics are enforced during both judge payload construction and report aggregation.

---

### Step 4 — Build judge payloads (blind)

```bash
python scripts/build_judge_payloads.py
```

Payloads are written under:

```text
artifacts/judge_payloads/
├── rag_5way/Qxxx.json
└── ablation_pairs/<pair_name>/Qxxx.json
```

Each payload contains:
- the query
- anonymized candidate contexts
- a private label-to-method mapping (hidden from the judge)

---

### Step 5 — Build judge prompts

```bash
python scripts/build_judge_prompts.py
```

Prompts can be exported as:
- Markdown (`Qxxx.md`)
- Chat messages (`Qxxx.messages.json`)
- JSONL (`prompts.jsonl`)

For **pairwise payloads**, identical A/B contexts may be **skipped automatically**
to avoid unnecessary LLM calls.

---

### Step 6 — External LLM-as-a-judge inference (manual)

Judge responses must be saved to:

```text
artifacts/judge_outputs/<group>/<model_name>/Qxxx_<replica>.json
```

Multiple replicas per query are supported.

---

### Step 7 — Aggregate judge outputs and build reports

Reports are built **separately for each evaluation group**.

```bash
python scripts/build_judge_reports.py   --judge_outputs_dir artifacts/judge_outputs/rag_5way   --judge_payloads_dir artifacts/judge_payloads/rag_5way   --reports_dir artifacts/reports/rag_5way
```

```bash
python scripts/build_judge_reports.py   --judge_outputs_dir artifacts/judge_outputs/ablation_pairs   --judge_payloads_dir artifacts/judge_payloads/ablation_pairs   --reports_dir artifacts/reports/ablation_pairs
```

### Judge output post-processing and recovery

During pairwise evaluation, some LLM judges may return semantically valid decisions
(e.g. explicit tie judgments) that do not conform to the strict expected schema
used during inference-time validation.

To avoid unnecessary re-running of expensive judge calls, the aggregation step
(`build_judge_reports.py`) performs **post-hoc recovery** of such outputs:

- Raw judge responses are re-parsed during report construction.
- Valid pairwise decisions (A / B / tie) are recovered when possible.
- Recovered runs are marked explicitly in reports and treated as valid observations.
- No preference inversion or score adjustment is performed during recovery.

This recovery step affects **only aggregation**, never retrieval or judge prompting,
and preserves experimental integrity while improving robustness.


Multi-way reports include per-method scores and winner statistics.  
Pairwise reports are intended for hypothesis validation and may use different summaries.

---

## Reproducibility notes

- All retrieval pipelines are deterministic given fixed inputs.
- No stochastic components are used during retrieval.
- Randomization is applied only during:
  - candidate shuffling in judge payloads
  - external LLM judge inference
- All configurations are frozen prior to evaluation.

### Interpretation of missing judge outputs in ablations

In ablation experiments, some query–pair combinations may intentionally have
no corresponding judge output. This occurs when both retrieval variants produce
identical contexts and running a judge would be redundant.

Such cases are treated as **neutral (no preference / skipped-identical)** during aggregation and are explicitly reported as skipped. They do not count as wins or losses for any method.

---

## Interactive mode (optional)

```bash
python scripts/run_ontology_interactive.py
```

