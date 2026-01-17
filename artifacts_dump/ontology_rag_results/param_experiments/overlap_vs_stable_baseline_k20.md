# Overlap Report (vs baseline)

Root: `artifacts\ontology_rag_results\param_experiments`  
Baseline: `stable_baseline`  
Metrics: overlap@20, Jaccard@20 (exact string match on retrieved chunks)

## Summary (each run compared to baseline)

| Run | Common queries | Mean overlap@k | Median overlap@k | Mean Jaccard@k | Median Jaccard@k |
|---|---:|---:|---:|---:|---:|
| `stable_baseline` | 10 | 0.9400 | 0.9500 | 1.0000 | 1.0000 |
| `baseline_debug_depth2_nodes800` | 10 | 0.9400 | 0.9500 | 1.0000 | 1.0000 |
| `baseline_debug_nodes800` | 10 | 0.9400 | 0.9500 | 1.0000 | 1.0000 |
| `stable_drill_narrow` | 10 | 0.2900 | 0.2500 | 0.2295 | 0.2056 |
| `stable_drill_wide` | 10 | 0.6000 | 0.5750 | 0.4902 | 0.4607 |
| `stable_level_bonus_off` | 10 | 0.9150 | 0.9500 | 0.9488 | 0.9524 |
| `stable_level_high` | 10 | 0.9100 | 0.9000 | 0.9398 | 0.9524 |
| `stable_no_dist` | 10 | 0.6450 | 0.6750 | 0.5430 | 0.5625 |
| `stable_no_level` | 10 | 0.9150 | 0.9500 | 0.9488 | 0.9524 |
| `stable_no_links` | 10 | 0.9400 | 0.9500 | 1.0000 | 1.0000 |
| `stable_no_type` | 10 | 0.8200 | 0.8000 | 0.7798 | 0.7636 |
| `stable_tau_child_high` | 10 | 0.8500 | 0.9000 | 0.8759 | 1.0000 |
| `stable_tau_child_low` | 10 | 0.9400 | 0.9500 | 1.0000 | 1.0000 |
| `stable_tau_local_high` | 10 | 0.8300 | 0.9000 | 0.8731 | 1.0000 |
| `stable_tau_local_low` | 10 | 0.9200 | 0.9250 | 0.9696 | 1.0000 |
| `stable_text_only` | 10 | 0.6550 | 0.6750 | 0.5413 | 0.5407 |
| `stable_type_high` | 10 | 0.9000 | 0.9000 | 0.9221 | 0.9024 |

## Interpretation (diagnostic, not quality)
- overlap@20 close to **1.0** vs baseline ⇒ retrieval output is largely unchanged by that config.
- overlap@20 noticeably lower ⇒ config changes what gets retrieved (structure/text weights likely matter).
- Jaccard@20 helps distinguish whether differences are small reorderings vs truly different sets.
