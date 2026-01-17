# LLM-as-Judge report

- Total run files scanned: **30**
- Parsed OK: **30**
- Parsed ERROR: **0**
- Fiveway runs (files): **30**
- Pairwise runs (files): **0**

## Fiveway summary (A–E)

- Total OK fiveway runs: **30**
- Tie/empty winners: **5** (0.1667)

| method | wins | win_rate |
| --- | --- | --- |
| ontology_stable_baseline | 17 | 0.5667 |
| classic_dense_heuristic | 5 | 0.1667 |
| bm25_heuristic | 3 | 0.1 |

### Fiveway overall metric summary (pooled over runs)

| method | n | relevance_mean | relevance_std | answerability_mean | answerability_std | noise_mean | noise_std | overall_mean | overall_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 30 | 3.3 | 1.1788363637137231 | 2.2333333333333334 | 1.3565507307349296 | 3.2666666666666666 | 0.9444331755018487 | 2.033333333333333 | 1.2726115785600325 |
| bm25_heuristic | 30 | 3.566666666666667 | 1.304721752165845 | 2.8 | 1.4239333576037019 | 2.2666666666666666 | 0.980265035707122 | 2.6666666666666665 | 1.3978637231524922 |
| classic_dense | 30 | 2.2333333333333334 | 1.072648457158112 | 1.1 | 0.9948141396330237 | 3.7666666666666666 | 0.9352607356658147 | 1.1333333333333333 | 1.0080138659874618 |
| classic_dense_heuristic | 30 | 2.966666666666667 | 1.629117242099459 | 2.2333333333333334 | 1.5687318254659988 | 2.433333333333333 | 1.3308885632599343 | 2.2333333333333334 | 1.5013403972802926 |
| ontology_stable_baseline | 30 | 3.6666666666666665 | 1.8445321495096563 | 3.1333333333333333 | 2.046583921349534 | 1.9333333333333333 | 1.6386144974533723 | 3.3 | 2.151983848163973 |

### Fiveway per-query winner breakdown

| qid | top_winner_method | top_winner_share | winners_breakdown |
| --- | --- | --- | --- |
| Q001 | ontology_stable_baseline | 1.0 | ontology_stable_baseline:3 |
| Q002 | bm25_heuristic | 0.6666666666666666 | bm25_heuristic:2;__TIE_OR_EMPTY__:1 |
| Q003 | ontology_stable_baseline | 0.6666666666666666 | ontology_stable_baseline:2;__TIE_OR_EMPTY__:1 |
| Q004 | classic_dense_heuristic | 1.0 | classic_dense_heuristic:3 |
| Q005 | ontology_stable_baseline | 1.0 | ontology_stable_baseline:3 |
| Q006 | classic_dense_heuristic | 0.6666666666666666 | classic_dense_heuristic:2;__TIE_OR_EMPTY__:1 |
| Q007 | ontology_stable_baseline | 1.0 | ontology_stable_baseline:3 |
| Q008 | ontology_stable_baseline | 1.0 | ontology_stable_baseline:3 |
| Q009 |  | 0.6666666666666666 | __TIE_OR_EMPTY__:2;bm25_heuristic:1 |
| Q010 | ontology_stable_baseline | 1.0 | ontology_stable_baseline:3 |
