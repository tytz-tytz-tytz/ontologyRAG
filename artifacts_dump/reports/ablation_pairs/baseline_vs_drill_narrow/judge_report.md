# LLM-as-Judge report

- Total run files scanned: **72**
- Parsed OK: **72**
- Parsed ERROR: **0**
- Fiveway runs (files): **0**
- Pairwise runs (files): **72**

## Pairwise summary (A/B)

- Total OK pairwise runs: **72**
- Tie/empty decisions (from runs): **20** (0.2778)
- Skipped identical payloads (no runs): **2**

| method | wins | win_rate |
| --- | --- | --- |
| stable_baseline | 40 | 0.5556 |
| stable_drill_narrow | 12 | 0.1667 |

### Pairwise per-query breakdown (includes SKIPPED_IDENTICAL)

| qid | top_decision_method | top_decision_share | decisions_breakdown |
| --- | --- | --- | --- |
| Q001 | stable_baseline | 1.0 | stable_baseline:9 |
| Q002 |  | 0.5555555555555556 | __TIE_OR_EMPTY__:5;stable_baseline:4 |
| Q003 |  | 0.6666666666666666 | __TIE_OR_EMPTY__:6;stable_drill_narrow:3 |
| Q004 |  | 1.0 | __TIE_OR_EMPTY__:9 |
| Q005 |  | 1.0 | __SKIPPED_IDENTICAL__:1 |
| Q006 | stable_drill_narrow | 1.0 | stable_drill_narrow:9 |
| Q007 | stable_baseline | 1.0 | stable_baseline:9 |
| Q008 |  | 1.0 | __SKIPPED_IDENTICAL__:1 |
| Q009 | stable_baseline | 1.0 | stable_baseline:9 |
| Q010 | stable_baseline | 1.0 | stable_baseline:9 |
