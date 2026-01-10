from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

LETTERS: List[str] = ["A", "B", "C", "D", "E"]


# -----------------------------
# Helpers
# -----------------------------
def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _is_blank(s: str) -> bool:
    return len(s.strip()) == 0


def _extract_contexts_for_judge(payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract candidate contexts A–E from payload in a robust way.

    Priority order:
    1) contexts_for_judge (your current format)
    2) retrieved_contexts (common legacy)
    3) contexts (legacy)
    4) candidates (legacy)

    If nothing is found, returns empty strings for all letters.
    """
    raw = payload.get("contexts_for_judge")
    if isinstance(raw, dict):
        return {k: _as_str(raw.get(k, "")) for k in LETTERS}

    for key in ("retrieved_contexts", "contexts", "candidates"):
        raw2 = payload.get(key)
        if isinstance(raw2, dict):
            return {k: _as_str(raw2.get(k, "")) for k in LETTERS}

    return {k: "" for k in LETTERS}


# -----------------------------
# Prompt templates
# -----------------------------
SYSTEM_PROMPT = (
    "You are an impartial judge evaluating retrieval results.\n"
    "Your task is to evaluate how well each retrieved context supports answering the given user query.\n\n"
    "The query and all retrieved contexts are written in Russian.\n"
    "Evaluate them as-is. Do not translate, rewrite, or summarize them.\n\n"
    "You are evaluating retrieval quality, not text generation.\n"
)

USER_INSTRUCTIONS = """Task
Given:
- A user query
- Several candidate retrieved contexts labeled A–E

Evaluate how well each context supports answering the query.
Judge only retrieval usefulness.

Important rules
- Do NOT answer the query.
- Do NOT reward writing style, fluency, or phrasing.
- Do NOT reward longer contexts or a larger number of chunks. More text is not better.
- Penalize irrelevant content, repetitions, and topic drift.
- Preserve the idea of top-k retrieval: if necessary information is not present in the provided context,
  the candidate is worse, even if such information might exist elsewhere.
- If a context includes both relevant and irrelevant parts, evaluate based on how easily a user could
  answer using ONLY that context.
- IMPORTANT: Inside a candidate context you may see lines like '---'. Treat them as part of the
  retrieved content (chunk separators), NOT as separators between candidates.
- If a candidate contains the literal marker '[NO RELEVANT CONTEXT FOUND]' OR is empty/whitespace-only,
  it does not support the query. In that case, set all metrics for that candidate to 0 and include it
  in failure_letters.

Metrics (0–5)
Score EACH candidate A–E on FOUR metrics:

1) relevance (0–5)
0 = Completely off-topic
5 = Directly about the same topic as the query

2) answerability (0–5)
0 = Cannot answer the query using only this context
5 = Can answer precisely and completely using only this context

3) noise (0–5)
0 = Almost no noise; minimal irrelevant/repeated content
5 = Almost all noise; mostly irrelevant/repeated/distracting content

4) overall (0–5)
Your overall usefulness judgment as a trade-off of relevance/answerability/noise.

Aggregates to compute
- winner: the single letter with the highest overall. If there is a tie for highest overall, set winner
  to an empty string "".
- ranking: all letters sorted by overall descending. Break ties alphabetically (A before B before C ...).
- failure_letters: all letters where the candidate is empty OR contains '[NO RELEVANT CONTEXT FOUND]'.
- confidence (0–5): your confidence in the overall ranking and winner.
  0 = very unsure / ambiguous, 5 = very sure / clear separation.

Output format (STRICT)
Return a single JSON object with EXACTLY this schema and keys:
{
  "relevance": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0},
  "answerability": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0},
  "noise": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0},
  "overall": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0},
  "winner": "<single letter A-E or empty string>",
  "ranking": ["A", "B", "C", "D", "E"],
  "failure_letters": ["<zero or more letters among A-E, sorted alphabetically>"],
  "confidence": 0,
  "rationales": {"A": "<short>", "B": "<short>", "C": "<short>", "D": "<short>", "E": "<short>"}
}

Keep rationales short (1–3 sentences each).
Do not add any other keys.
Do not wrap the JSON in markdown code fences.

---
"""


# -----------------------------
# Data model
# -----------------------------
@dataclass(frozen=True)
class JudgePrompt:
    """
    Internal representation of a single judge prompt.

    id: payload id (e.g., Q001)
    query: Russian query string
    candidates: dict A–E -> context text
    """
    id: str
    query: str
    candidates: Dict[str, str]


# -----------------------------
# Builders
# -----------------------------
def build_prompts(payload: Any) -> List[JudgePrompt]:
    """
    Build prompts from:
    - a single payload dict
    - OR a list of payload dicts

    Supported payload shape (recommended):
    {
      "id": "Q001",
      "query": "...",
      "contexts_for_judge": {"A": "...", ..., "E": "..."},
      "private_mapping": {"A": "...", ...}  # ignored here on purpose
    }
    """
    items: List[Dict[str, Any]] = []

    if isinstance(payload, list):
        # List of payload dicts
        for it in payload:
            if isinstance(it, dict):
                items.append(it)
            else:
                raise TypeError(f"Expected list[dict] payload, got element type: {type(it)}")
    elif isinstance(payload, dict):
        items = [payload]
    else:
        raise TypeError(f"Expected payload dict or list[dict], got: {type(payload)}")

    prompts: List[JudgePrompt] = []
    for item in items:
        pid = _as_str(item.get("id", "")).strip() or "UNKNOWN"
        query = _as_str(item.get("query", "")).strip()

        candidates = _extract_contexts_for_judge(item)
        # Ensure all letters exist and are strings
        candidates = {k: _as_str(candidates.get(k, "")) for k in LETTERS}

        prompts.append(JudgePrompt(id=pid, query=query, candidates=candidates))

    return prompts


def _render_candidates_block(candidates: Dict[str, str]) -> str:
    parts: List[str] = []
    for letter in LETTERS:
        ctx = _as_str(candidates.get(letter, ""))
        parts.append(f"===== CANDIDATE {letter} =====\n{ctx}\n===== END CANDIDATE {letter} =====\n")
    return "\n".join(parts).rstrip() + "\n"


def prompts_to_markdown(prompts: Sequence[JudgePrompt], *, title: Optional[str] = None) -> str:
    """
    Render one or many prompts into a Markdown file for inspection.

    Note: The model prompt itself is not wrapped in code fences on purpose,
    because you later embed the same content into chat messages.
    """
    lines: List[str] = []
    if title:
        lines.append(f"# {title}\n")

    for p in prompts:
        lines.append(f"## {p.id}\n")
        lines.append(SYSTEM_PROMPT.strip() + "\n")
        lines.append(USER_INSTRUCTIONS.rstrip())
        lines.append("User query (RU)\n" + p.query + "\n\n")
        lines.append("Retrieved contexts\n")
        lines.append(_render_candidates_block(p.candidates))
        lines.append("\n---\n")

    return "\n".join(lines).rstrip() + "\n"


def prompts_to_messages(prompts: Sequence[JudgePrompt]) -> List[Dict[str, str]]:
    """
    Convert prompts into OpenAI Chat Completions-style messages.

    If you pass multiple prompts, they will be concatenated into a single user message.
    For your pipeline, you usually want one payload per file anyway.
    """
    user_chunks: List[str] = []
    for p in prompts:
        chunk = (
            USER_INSTRUCTIONS
            + "User query (RU)\n"
            + p.query
            + "\n\nRetrieved contexts\n"
            + _render_candidates_block(p.candidates)
        )
        user_chunks.append(chunk.rstrip())

    user_content = "\n\n".join(user_chunks).rstrip() + "\n"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def dumps_messages_json(id_: str, prompts: Sequence[JudgePrompt]) -> str:
    """
    Convenience helper to produce {"id": ..., "messages": [...]} as JSON text.
    """
    obj = {"id": id_, "messages": prompts_to_messages(prompts)}
    return json.dumps(obj, ensure_ascii=False, indent=2)
