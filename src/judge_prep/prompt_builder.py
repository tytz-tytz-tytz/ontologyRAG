from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Legacy letters (multi-way)
LETTERS: List[str] = ["A", "B", "C", "D", "E"]
PAIRWISE_LETTERS: List[str] = ["A", "B"]

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


def _normalize_contexts_dict(raw: Dict[str, Any]) -> Dict[str, str]:
    """Normalize contexts dict values to strings."""
    out: Dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str):
            out[k] = _as_str(v)
    return out


def _extract_contexts_raw(payload: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Return raw contexts dict if present (any of known keys).
    Does NOT pad to A–E; returns exactly what payload provides.
    """
    raw = payload.get("contexts_for_judge")
    if isinstance(raw, dict):
        return _normalize_contexts_dict(raw)

    for key in ("retrieved_contexts", "contexts", "candidates"):
        raw2 = payload.get(key)
        if isinstance(raw2, dict):
            return _normalize_contexts_dict(raw2)

    return None


def _detect_mode_and_letters(payload: Dict[str, Any], mode: str) -> Tuple[str, List[str]]:
    """
    Decide whether to use multi-way (A–E) or pairwise (A–B).
    - mode: "auto" | "multiway" | "pairwise"
    """
    mode = (mode or "auto").strip().lower()
    if mode not in ("auto", "multiway", "pairwise"):
        raise ValueError("mode must be one of: auto | multiway | pairwise")

    if mode == "multiway":
        return "multiway", LETTERS
    if mode == "pairwise":
        return "pairwise", PAIRWISE_LETTERS

    # auto
    raw = _extract_contexts_raw(payload)
    if raw is None:
        # fall back to legacy multi-way
        return "multiway", LETTERS

    present = [k for k in raw.keys() if k in LETTERS]
    # If payload has exactly A and B (or generally 2 among A–E), treat as pairwise.
    if len(present) == 2 and set(present) == set(PAIRWISE_LETTERS):
        return "pairwise", PAIRWISE_LETTERS

    # Default: multi-way
    return "multiway", LETTERS


def _extract_contexts_for_letters(payload: Dict[str, Any], letters: List[str]) -> Dict[str, str]:
    """
    Extract contexts for specified letters.
    Pads missing letters with "" (only within requested letters).
    """
    raw = _extract_contexts_raw(payload)
    if raw is None:
        return {k: "" for k in letters}

    return {k: _as_str(raw.get(k, "")) for k in letters}


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

# ---- Legacy multi-way (A–E) instructions (UNCHANGED SEMANTICS) ----
USER_INSTRUCTIONS_MULTIWAY = """Task
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

# ---- New pairwise (A–B) instructions ----
USER_INSTRUCTIONS_PAIRWISE = """Task
Given:
- A user query
- Two candidate retrieved contexts labeled A and B (top-k retrieval results)

Choose which context better supports answering the user query using ONLY the information contained in that context.

Important rules
- Do NOT answer the query.
- Do NOT reward writing style, fluency, or phrasing.
- Do NOT reward longer contexts. More text is not better.
- Do NOT use any external knowledge. Judge only what is present in the contexts.
- Penalize irrelevant content, repetitions, and topic drift.
- Preserve the idea of top-k retrieval: if necessary information is not present in the provided context,
  that candidate is worse, even if such information might exist elsewhere.
- If both contexts are equally useful OR equally useless for answering the query, you MUST output Tie.
- IMPORTANT: Inside a candidate context you may see lines like '---'. Treat them as part of the
  retrieved content (chunk separators), NOT as separators between candidates.
- If a candidate contains the literal marker '[NO RELEVANT CONTEXT FOUND]' OR is empty/whitespace-only,
  treat it as providing no support.

Decision
Return one of:
- "A"  (A is better)
- "B"  (B is better)
- "Tie" (no clear advantage)

Output format (STRICT)
Return a single JSON object with EXACTLY this schema and keys:
{
  "decision": "A" | "B" | "Tie",
  "reason": "<1-3 sentences, content-based; do not invent facts; no spoilers>"
}

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
    candidates: dict letter -> context text (letters depends on mode)
    mode: "multiway" | "pairwise"
    letters: list of letters used in this prompt
    """
    id: str
    query: str
    candidates: Dict[str, str]
    mode: str
    letters: List[str]


# -----------------------------
# Builders
# -----------------------------
def build_prompts(payload: Any, mode: str = "auto") -> List[JudgePrompt]:
    """
    Build prompts from:
    - a single payload dict
    - OR a list of payload dicts

    Supported payload shape (recommended):
    {
      "id": "Q001",
      "query": "...",
      "contexts_for_judge": {"A": "...", ..., "E": "..."} OR {"A": "...", "B": "..."},
      "private_mapping": {"A": "...", ...}  # ignored here on purpose
    }

    mode:
      - "auto": choose by number of candidates present in payload
      - "multiway": force A–E
      - "pairwise": force A–B
    """
    items: List[Dict[str, Any]] = []

    if isinstance(payload, list):
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

        detected_mode, letters = _detect_mode_and_letters(item, mode=mode)
        candidates = _extract_contexts_for_letters(item, letters)

        # Ensure all requested letters exist and are strings
        candidates = {k: _as_str(candidates.get(k, "")) for k in letters}

        prompts.append(
            JudgePrompt(
                id=pid,
                query=query,
                candidates=candidates,
                mode=detected_mode,
                letters=list(letters),
            )
        )

    return prompts


def _render_candidates_block(candidates: Dict[str, str], letters: List[str]) -> str:
    parts: List[str] = []
    for letter in letters:
        ctx = _as_str(candidates.get(letter, ""))
        parts.append(
            f"===== CANDIDATE {letter} =====\n{ctx}\n===== END CANDIDATE {letter} =====\n"
        )
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

        if p.mode == "pairwise":
            lines.append(USER_INSTRUCTIONS_PAIRWISE.rstrip())
        else:
            lines.append(USER_INSTRUCTIONS_MULTIWAY.rstrip())

        lines.append("User query (RU)\n" + p.query + "\n\n")
        lines.append("Retrieved contexts\n")
        lines.append(_render_candidates_block(p.candidates, p.letters))
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
        instructions = USER_INSTRUCTIONS_PAIRWISE if p.mode == "pairwise" else USER_INSTRUCTIONS_MULTIWAY
        chunk = (
            instructions
            + "User query (RU)\n"
            + p.query
            + "\n\nRetrieved contexts\n"
            + _render_candidates_block(p.candidates, p.letters)
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
