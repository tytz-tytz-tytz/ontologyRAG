from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Set

from classic_rag.index.store import ClassicRAGIndex
from classic_rag.rag.retrieve import retrieve_with_scores


@dataclass(frozen=True)
class HeuristicRAGConfig:
    # Final number of chunks returned (context budget).
    top_k: int = 10

    # Retrieve more candidates first, then filter down to top_k.
    candidate_multiplier: int = 6

    # Filter out very short chunks (headings/captions/noise).
    min_chars: int = 80

    # Deduplicate texts after normalization.
    deduplicate: bool = True

    # Drop common caption-like chunks (e.g., "Рисунок 51 — ...", "Table 3 - ...").
    drop_captions: bool = True

    # Drop chunks dominated by bullets / list formatting.
    drop_bullets: bool = True

    # Drop fragments ending with ":" (often incomplete / followed by a list).
    drop_colon_trailing: bool = True

    # Drop table-like header rows (many short "column name" tokens).
    drop_table_like: bool = True

    # Drop common incomplete phrases ("см. подробнее...", "в разделе", etc.).
    drop_incomplete_phrases: bool = True

    # Drop short heading-like chunks (e.g., "Редактирование событий").
    drop_headings: bool = True

    # In fallback, still keep some minimum length to avoid tiny junk.
    fallback_min_chars: int = 40


def _normalize_text_for_dedup(text: str) -> str:
    """
    Normalize text for deduplication:
    - lowercase
    - collapse whitespace
    - trim punctuation at ends
    """
    t = (text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" \t\n\r.,;:!—-")
    return t


def _looks_like_caption(text: str) -> bool:
    """
    Heuristic: detect figure/table captions using only the chunk text.
    Examples:
      - "Рисунок 51 — ..."
      - "Table 3 - ..."
      - "Figure 2: ..."
    """
    t = (text or "").strip().lower()
    if not t:
        return False

    prefixes = ("рисунок", "таблица", "figure", "table", "диаграмма", "схема", "листинг")
    return t.startswith(prefixes)


def _looks_like_bullet_list(text: str) -> bool:
    """
    Heuristic: detect chunks dominated by bullets/lists based on raw text.
    This helps avoid retrieving mostly formatting rather than explanatory prose.
    """
    t = (text or "").strip()
    if not t:
        return False

    bullet_markers = ["•", "—", "-", "*", "·"]
    bullet_count = sum(t.count(m) for m in bullet_markers)
    newline_count = t.count("\n")

    if bullet_count >= 3:
        return True
    if t.lstrip().startswith("•") and newline_count >= 1:
        return True
    if newline_count >= 2 and bullet_count >= 1:
        return True

    return False


def _looks_like_table_header(text: str) -> bool:
    """
    Heuristic: detect table-like header rows such as:
      'Состояние Иконка Изменение сценария Изменение условий ...'

    Approximation:
    - many tokens
    - almost no sentence punctuation
    - many Title Case tokens (start with uppercase)
    """
    t = (text or "").strip()
    if not t:
        return False

    # If there is sentence punctuation, it's probably not a header row.
    if any(ch in t for ch in ".!?;"):
        return False

    tokens = [x for x in t.split() if x]
    if len(tokens) < 5:
        return False

    upper_initial = sum(1 for tok in tokens if tok[:1].isupper())
    ratio = upper_initial / max(1, len(tokens))

    # Catch typical column-name rows, including mixed-case tokens.
    if ratio >= 0.5 and len(tokens) >= 5:
        return True

    # Also catch very long token rows regardless of capitalization.
    if len(tokens) >= 10:
        return True

    return False


def _looks_like_heading(text: str) -> bool:
    """
    Heuristic: detect short heading-like chunks.
    Typical properties:
    - short text
    - no sentence punctuation
    - mostly letters/spaces
    """
    t = (text or "").strip()
    if not t:
        return False

    # Very short lines are often headings.
    if len(t) > 80:
        return False

    # Headings usually have no sentence punctuation.
    if any(ch in t for ch in ".!?"):
        return False

    letters_spaces = sum(1 for ch in t if ch.isalpha() or ch.isspace())
    if letters_spaces / max(1, len(t)) >= 0.9 and t[:1].isupper():
        return True

    return False


_INCOMPLETE_PATTERNS = [
    r"\(см\.\s*$",                   # ends with "(см."
    r"см\.\s*рисунок\s*\d+\)\s*:$",   # ends with "... (см. Рисунок 180):"
    r"см\.\s*подробнее.*$",           # "см. подробнее ..."
    r"в\s+разделе\s*$",               # ends with "в разделе"
    r"в\s+разделе\s*\(?$",            # ends with "в разделе ("
]


def _looks_incomplete(text: str) -> bool:
    """
    Heuristic: detect common incomplete fragments that refer to missing continuation.
    """
    t = (text or "").strip().lower()
    if not t:
        return False

    for pat in _INCOMPLETE_PATTERNS:
        if re.search(pat, t):
            return True

    return False


def retrieve_heuristic(
    index: ClassicRAGIndex,
    query: str,
    cfg: HeuristicRAGConfig = HeuristicRAGConfig(),
) -> List[str]:
    """
    Heuristic-enhanced dense retrieval without using any structure/metadata.

    Steps:
    1) Oversample candidates with dense retrieval.
    2) Filter obvious non-informative chunks (captions/lists/table headers/headings/incomplete fragments).
    3) Apply min length and dedup.
    4) Fallback to fill top_k if too strict (still avoid obvious junk and duplicates).
    """
    cand_k = max(cfg.top_k * cfg.candidate_multiplier, cfg.top_k)
    candidates: List[Tuple[str, str, float]] = retrieve_with_scores(index, query, top_k=cand_k)

    out: List[str] = []
    seen: Set[str] = set()

    def _accept_strict(txt: str) -> bool:
        if not txt:
            return False

        if cfg.drop_captions and _looks_like_caption(txt):
            return False

        if cfg.drop_bullets and _looks_like_bullet_list(txt):
            return False

        if cfg.drop_table_like and _looks_like_table_header(txt):
            return False

        if cfg.drop_headings and _looks_like_heading(txt):
            return False

        if cfg.drop_incomplete_phrases and _looks_incomplete(txt):
            return False

        if cfg.drop_colon_trailing and txt.endswith(":"):
            return False

        if len(txt) < cfg.min_chars:
            return False

        return True

    # Pass 1: strict filtering
    for _cid, text, _score in candidates:
        txt = (text or "").strip()

        if not _accept_strict(txt):
            continue

        if cfg.deduplicate:
            key = _normalize_text_for_dedup(txt)
            if key in seen:
                continue
            seen.add(key)

        out.append(txt)
        if len(out) >= cfg.top_k:
            break

    # Pass 2: fallback if filters were too strict.
    # Relax most constraints, but keep minimal hygiene and avoid tiny fragments.
    if len(out) < cfg.top_k:
        for _cid, text, _score in candidates:
            txt = (text or "").strip()
            if not txt:
                continue

            if cfg.drop_captions and _looks_like_caption(txt):
                continue

            if cfg.drop_table_like and _looks_like_table_header(txt):
                continue

            if cfg.drop_headings and _looks_like_heading(txt):
                continue

            if len(txt) < cfg.fallback_min_chars:
                continue

            if cfg.deduplicate:
                key = _normalize_text_for_dedup(txt)
                if key in seen:
                    continue
                seen.add(key)

            out.append(txt)
            if len(out) >= cfg.top_k:
                break

    return out
