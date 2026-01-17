# LLM-as-Judge report

- Total run files scanned: **36**
- Parsed OK: **36**
- Parsed ERROR: **0**
- Fiveway runs (files): **0**
- Pairwise runs (files): **36**

## Pairwise summary (A/B)

- Total OK pairwise runs: **36**
- Tie/empty decisions (from runs): **4** (0.1111)
- Skipped identical payloads (no runs): **6**

| method | wins | win_rate |
| --- | --- | --- |
| stable_baseline | 27 | 0.75 |
| stable_text_only | 5 | 0.1389 |

### Pairwise per-query breakdown (includes SKIPPED_IDENTICAL)

| qid | top_decision_method | top_decision_share | decisions_breakdown |
| --- | --- | --- | --- |
| Q001 |  | 1.0 | __SKIPPED_IDENTICAL__:1 |
| Q002 |  | 1.0 | __SKIPPED_IDENTICAL__:1 |
| Q003 | stable_text_only | 0.5555555555555556 | stable_text_only:5;__TIE_OR_EMPTY__:4 |
| Q004 |  | 1.0 | __SKIPPED_IDENTICAL__:1 |
| Q005 |  | 1.0 | __SKIPPED_IDENTICAL__:1 |
| Q006 |  | 1.0 | __SKIPPED_IDENTICAL__:1 |
| Q007 | stable_baseline | 1.0 | stable_baseline:9 |
| Q008 |  | 1.0 | __SKIPPED_IDENTICAL__:1 |
| Q009 | stable_baseline | 1.0 | stable_baseline:9 |
| Q010 | stable_baseline | 1.0 | stable_baseline:9 |
