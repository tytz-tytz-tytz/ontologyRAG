# src/judge_prep/clean_cap.py
"""
Utilities for cleaning and capping retrieval contexts before LLM-as-judge.

This module MUST:
- apply identical clean+cap logic for all retrieval methods;
- keep retrieval order (rank) as-is (no extra optimization);
- cap by the same token_budget_per_method for all methods;
- never cut mid-sentence (prefer sentence boundary);
- avoid empty contexts: if no valid text -> "[NO RELEVANT CONTEXT FOUND]".

Design goals:
- conservative lexical heuristics (no semantic models);
- stable, reproducible behavior.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence


# =========================
# Config and result models
# =========================

@dataclass(frozen=True)
class CleanCapConfig:
    # Minimum acceptable output length in characters (after cleanup).
    min_chars: int = 20

    # Token budget per method (same for all methods).
    token_budget_per_method: int = 350

    # Optional: tiktoken encoding name (used if tiktoken is installed).
    encoding_name: str = "cl100k_base"

    # Drop caption-like lines ("Figure 12 ...", "Рисунок 12 ...").
    drop_captions: bool = True

    # Drop standalone headings without an explanatory paragraph after them.
    drop_headings: bool = True

    # Drop table headers / schema-like lines.
    drop_table_headers: bool = True

    # Drop lines that end with ":" and look like an unfinished UI fragment (optional).
    drop_trailing_colon_fragments: bool = False

    # Heading heuristics.
    heading_max_words: int = 8
    heading_max_chars: int = 80

    # Table header heuristics.
    table_header_min_words: int = 6
    table_header_max_punct: int = 1
    table_header_uppercase_ratio: float = 0.35

    # Optional: limit number of chunks to consider (None = no limit).
    max_chunks: Optional[int] = None

    # Joiner for chunks (usually "\n").
    joiner: str = "\n"


@dataclass(frozen=True)
class CleanCapStats:
    in_chunks: int
    kept_chunks: int
    dropped_empty: int
    dropped_dedup: int
    dropped_caption: int
    dropped_heading: int
    dropped_table_header: int
    dropped_trailing_colon: int
    truncated: bool


@dataclass(frozen=True)
class CleanCapResult:
    text: str
    tokens: int
    stats: CleanCapStats


# =========================
# Public API
# =========================

_NO_CONTEXT = "[NO RELEVANT CONTEXT FOUND]"


def clean_and_cap(chunks: List[str], config: CleanCapConfig, query: str = "") -> CleanCapResult:
    """
    Clean and cap a list of retrieved chunks (already ordered by retrieval rank).

    Compatible with build_judge_payloads.py call style:
      - clean_and_cap(chunks, config)
      - clean_and_cap(chunks, config, query="...")

    Returns:
      CleanCapResult with .text, .tokens and .stats fields expected by debug output.
    """
    query = (query or "").strip()
    in_chunks = len(chunks)

    # Optionally enforce max_chunks.
    if config.max_chunks is not None:
        chunks = chunks[: max(0, int(config.max_chunks))]

    # 1) Normalize chunks and split into paragraphs while keeping order.
    paragraphs: List[str] = []
    dropped_empty = 0
    for ch in chunks:
        ch = _normalize_text(ch)
        if not ch:
            dropped_empty += 1
            continue
        paragraphs.extend(_split_into_paragraphs(ch))

    # 2) Drop obvious junk paragraphs and apply format normalization.
    dropped_caption = 0
    dropped_table_header = 0
    dropped_trailing_colon = 0

    filtered: List[str] = []
    for p in paragraphs:
        original = p
        p2, cap_drop, tbl_drop, colon_drop = _clean_paragraph(
            p,
            cfg=config,
        )
        dropped_caption += cap_drop
        dropped_table_header += tbl_drop
        dropped_trailing_colon += colon_drop

        if not p2:
            continue

        # Drop very short lines (explicit requirement: < 40 chars).
        # Apply after cleanup, so we do not keep UI crumbs.
        if len(p2) < 40:
            continue

        # Drop "list fragments" that are just bullet items without meaning.
        if _is_list_fragment(p2):
            continue

        # Drop paragraphs with extremely high stopword ratio and low content.
        if _looks_like_stopword_noise(p2):
            continue

        filtered.append(p2)

    # 3) Drop standalone headings (requires look-ahead).
    dropped_heading = 0
    if config.drop_headings:
        filtered2: List[str] = []
        i = 0
        while i < len(filtered):
            p = filtered[i]
            if _looks_like_heading(p, cfg=config):
                # If next paragraph exists and looks like a real paragraph, keep heading.
                if i + 1 < len(filtered) and _looks_like_body_paragraph(filtered[i + 1]):
                    filtered2.append(p)
                else:
                    dropped_heading += 1
                i += 1
                continue
            filtered2.append(p)
            i += 1
        filtered = filtered2

    # 4) De-dup consecutive identical paragraphs.
    dropped_dedup = 0
    deduped: List[str] = []
    prev_key: Optional[str] = None
    for p in filtered:
        key = p.strip().lower()
        if key == prev_key:
            dropped_dedup += 1
            continue
        deduped.append(p)
        prev_key = key
    filtered = deduped

    # 5) Topic drift control: paragraph must contain at least one keyword from query.
    # Requirement: lexical overlap only, no semantics.
    if query:
        kws = _extract_keywords(query)
        if kws:
            filtered = [p for p in filtered if _contains_any_keyword(p, kws)]

    # 6) Cap by tokens, preserve order. Never cut mid-sentence.
    budget = max(0, int(config.token_budget_per_method))
    text, truncated = _cap_paragraphs_to_budget(filtered, budget, cfg=config)

    # 7) Final formatting rules:
    # - paragraphs joined by "\n\n"
    # - no more than 2 newlines in a row
    text = _final_format(text)

    tokens = _count_tokens(text, cfg=config)

    # 8) Empty / too short handling.
    if not text.strip() or len(text.strip()) < max(0, int(config.min_chars)):
        text = _NO_CONTEXT
        tokens = _count_tokens(text, cfg=config)
        truncated = False  # placeholder is not a truncation of useful text

    kept_chunks_est = _estimate_kept_chunks(filtered, cfg=config)

    return CleanCapResult(
        text=text,
        tokens=tokens,
        stats=CleanCapStats(
            in_chunks=in_chunks,
            kept_chunks=kept_chunks_est,
            dropped_empty=dropped_empty,
            dropped_dedup=dropped_dedup,
            dropped_caption=dropped_caption,
            dropped_heading=dropped_heading,
            dropped_table_header=dropped_table_header,
            dropped_trailing_colon=dropped_trailing_colon,
            truncated=truncated,
        ),
    )


# =========================
# Paragraph processing
# =========================

_CAPTION_RE = re.compile(r"^\s*(рисунок|таблица|figure|table)\s*\d+\b", re.IGNORECASE)
_CAPTION_DASH_RE = re.compile(r"^\s*(рисунок|таблица)\s*\d+\s*[—-]\s*.+$", re.IGNORECASE)
_SUCCESSFUL_RE = re.compile(r"\bSuccessful change of state\b", re.IGNORECASE)

# Simple schema/table field patterns.
_SCHEMA_LINE_RE = re.compile(
    r"^\s*[A-Za-z_][A-Za-z0-9_]*(\[\])?\s+(string|str|boolean|bool|object|array|integer|int|number|float|decimal|date)\b",
    re.IGNORECASE,
)
_SCHEMA_COLON_RE = re.compile(
    r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*:\s*(string|str|boolean|bool|object|array|integer|int|number|float|decimal|date)\b",
    re.IGNORECASE,
)

# Table header like: "Filter  Description" or multiple spaced columns.
_MULTI_COL_RE = re.compile(r"^\s*\S+(?:\s{2,}\S+){1,}\s*$")

# Trailing colon fragments.
_TRAILING_COLON_RE = re.compile(r"^\s*.+:\s*$")

# Bullet detection.
_BULLET_RE = re.compile(r"^\s*(?:•|\-|\*|\d+\.)\s+")
_BULLET_FRAGMENT_RE = re.compile(r"^\s*(?:•|\-|\*|\d+\.)\s+[^.?!]{0,80}[;:]?\s*$")


def _clean_paragraph(p: str, cfg: CleanCapConfig) -> tuple[str, int, int, int]:
    """
    Clean one paragraph. Returns:
      (cleaned_paragraph, dropped_caption_count, dropped_table_header_count, dropped_trailing_colon_count)

    Note: we do not drop short lines here; that is handled by the caller (< 40 chars rule).
    """
    dropped_caption = 0
    dropped_table_header = 0
    dropped_trailing_colon = 0

    # Hard removals: known junk string (explicit examples).
    if _SUCCESSFUL_RE.search(p):
        return "", 0, 1, 0

    # Remove inline figure references but keep surrounding text.
    p = re.sub(r"\(\s*см\.\s*рис(?:унок)?\s*\d+\s*\)", "", p, flags=re.IGNORECASE)
    p = re.sub(r"\(\s*see\s*(?:figure|fig\.)\s*\d+\s*\)", "", p, flags=re.IGNORECASE)

    # Drop captions.
    if cfg.drop_captions:
        # Drop lines like "Рисунок 12 — ..." if they do not carry meaning (caption-only).
        if _CAPTION_RE.match(p) or _CAPTION_DASH_RE.match(p):
            dropped_caption += 1
            return "", dropped_caption, 0, 0

    # Drop schema/table lines.
    if cfg.drop_table_headers:
        if _SCHEMA_LINE_RE.match(p) or _SCHEMA_COLON_RE.match(p):
            dropped_table_header += 1
            return "", 0, dropped_table_header, 0

        # Drop multi-column headers with too many columns / few punct.
        if _looks_like_table_header(p, cfg):
            dropped_table_header += 1
            return "", 0, dropped_table_header, 0

        # Drop obvious "Header Header" pairs like "Фильтр Описание", "Поле Описание".
        if _looks_like_two_col_header(p):
            dropped_table_header += 1
            return "", 0, dropped_table_header, 0

    # Drop trailing colon fragments (optional requirement).
    if cfg.drop_trailing_colon_fragments and _TRAILING_COLON_RE.match(p):
        dropped_trailing_colon += 1
        return "", 0, 0, dropped_trailing_colon

    # Normalize whitespace.
    p = _normalize_text(p)

    return p, dropped_caption, dropped_table_header, dropped_trailing_colon


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Split a chunk into paragraphs.
    Rules:
    - treat 2+ newlines as paragraph separators;
    - preserve order.
    """
    t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not t:
        return []
    parts = [p.strip() for p in re.split(r"\n{2,}", t) if p.strip()]
    # If no blank lines exist, keep as one paragraph.
    return parts if parts else [t]


def _is_list_fragment(p: str) -> bool:
    """
    Identify "list fragments" such as:
      • события Start;
      • блоков действий;
    without a meaningful sentence.
    """
    # If the whole paragraph is only 1-3 bullet lines and each is fragment-like -> drop.
    lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
    if not lines:
        return True
    if len(lines) <= 3 and all(_BULLET_FRAGMENT_RE.match(ln) for ln in lines):
        return True
    return False


def _looks_like_heading(p: str, cfg: CleanCapConfig) -> bool:
    """
    Heuristic heading:
    - single line
    - short length and limited words
    - lacks sentence-ending punctuation
    """
    if "\n" in p:
        return False
    if len(p) > cfg.heading_max_chars:
        return False
    words = re.findall(r"\w+", p, flags=re.UNICODE)
    if len(words) == 0 or len(words) > cfg.heading_max_words:
        return False
    # No strong sentence punctuation at end.
    if re.search(r"[.!?…]$", p):
        return False
    return True


def _looks_like_body_paragraph(p: str) -> bool:
    """
    Minimal check that a paragraph looks like a real explanatory text.
    """
    if len(p) < 80:
        return False
    if re.search(r"[.!?…]", p):
        return True
    # Long paragraph without punctuation can still be meaningful, but rare.
    return len(re.findall(r"\w+", p, flags=re.UNICODE)) >= 12


def _looks_like_two_col_header(p: str) -> bool:
    """
    Drop common two-column UI/table headers such as:
      "Фильтр Описание"
      "Поле Описание"
    """
    low = p.strip().lower()
    if low in {"фильтр описание", "поле описание", "parameter description", "field description"}:
        return True
    return False


def _looks_like_table_header(p: str, cfg: CleanCapConfig) -> bool:
    """
    Heuristic "table header" detection.
    """
    if not _MULTI_COL_RE.match(p):
        return False

    words = re.findall(r"\w+", p, flags=re.UNICODE)
    if len(words) < cfg.table_header_min_words:
        return False

    punct = re.findall(r"[.,;:!?…]", p)
    if len(punct) > cfg.table_header_max_punct:
        return False

    letters = re.findall(r"[A-Za-zА-ЯЁа-яё]", p)
    if not letters:
        return False
    upper = [ch for ch in letters if ch.isupper()]
    ratio = len(upper) / max(1, len(letters))
    if ratio > cfg.table_header_uppercase_ratio:
        return True

    # Multi-column line with low punctuation is often a header.
    return True


# =========================
# Topic keyword overlap
# =========================

_RU_STOPWORDS = {
    "и", "в", "во", "на", "по", "о", "об", "от", "до", "для", "к", "ко", "из",
    "с", "со", "у", "при", "как", "что", "это", "то", "а", "но", "или", "ли",
    "же", "не", "ни", "без", "над", "под", "про", "между", "после", "перед",
}
_EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by", "at", "from", "as",
    "is", "are", "was", "were", "be", "been", "it", "this", "that",
}


def _extract_keywords(query: str) -> List[str]:
    """
    Extract keywords from query using basic normalization + stopword removal.
    No lemmatization.
    """
    q = (query or "").lower()
    q = q.replace("\u00a0", " ")
    q = re.sub(r"[^\w\s\-]", " ", q, flags=re.UNICODE)
    parts = [p for p in re.split(r"\s+", q) if p]

    kws: List[str] = []
    for p in parts:
        if p in _RU_STOPWORDS or p in _EN_STOPWORDS:
            continue
        if len(p) <= 2:
            continue
        kws.append(p)

    # Deduplicate preserving order.
    seen = set()
    out: List[str] = []
    for k in kws:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def _contains_any_keyword(text: str, keywords: Sequence[str]) -> bool:
    """
    True if text contains at least one keyword (lexical overlap).
    Uses word boundary match when possible; otherwise substring.
    """
    t = text.lower()
    for kw in keywords:
        if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", t, flags=re.UNICODE):
            return True
        if kw in t:
            return True
        # Light prefix match for inflections (RU).
        if len(kw) >= 5:
            stem = kw[:4]
            if re.search(rf"(?<!\w){re.escape(stem)}\w{{1,10}}", t, flags=re.UNICODE):
                return True
    return False


# =========================
# Stopword-noise detection
# =========================

def _looks_like_stopword_noise(p: str) -> bool:
    """
    Drop paragraphs with a very high stopword ratio and little content.
    This is conservative: only triggers when content is clearly low-value.
    """
    words = [w.lower() for w in re.findall(r"\w+", p, flags=re.UNICODE)]
    if len(words) < 6:
        return False

    stop = 0
    content = 0
    for w in words:
        if w in _RU_STOPWORDS or w in _EN_STOPWORDS:
            stop += 1
        else:
            content += 1

    ratio = stop / max(1, len(words))
    # "high share of service words without meaningful text"
    return ratio >= 0.75 and content <= 2


# =========================
# Capping logic (sentence-safe)
# =========================

_SENT_BOUNDARY_RE = re.compile(r"[.!?…]+(?:[\"'”»)\]]+)?\s*$", re.UNICODE)


def _cap_paragraphs_to_budget(paragraphs: Sequence[str], budget: int, cfg: CleanCapConfig) -> tuple[str, bool]:
    """
    Append paragraphs in order until token budget is reached.
    If a paragraph overflows, trim it to the nearest sentence boundary.
    """
    if budget <= 0:
        return "", False

    out: List[str] = []
    used = 0
    truncated = False

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        p_tokens = _count_tokens(p, cfg=cfg)
        if used + p_tokens <= budget:
            out.append(p)
            used += p_tokens
            continue

        # Overflow: we need to trim the last paragraph safely.
        remaining = budget - used
        if remaining <= 0:
            truncated = True
            break

        trimmed = _safe_sentence_trim(p, remaining, cfg=cfg).strip()
        if trimmed:
            out.append(trimmed)
        truncated = True
        break

    return "\n\n".join(out).strip(), truncated


def _safe_sentence_trim(text: str, token_limit: int, cfg: CleanCapConfig) -> str:
    """
    Trim text to <= token_limit while avoiding cutting mid-sentence.

    Strategy:
    - take prefix that fits token_limit,
    - then search backwards for a sentence boundary (. ! ? …),
    - if none found, fall back to last whitespace.
    """
    if not text or token_limit <= 0:
        return ""

    # Fast path: already within limit.
    if _count_tokens(text, cfg=cfg) <= token_limit:
        return text

    # We need a token-aware prefix. Prefer tiktoken if available, else fallback.
    prefix = _prefix_by_tokens(text, token_limit, cfg=cfg).rstrip()
    if not prefix:
        return ""

    # Find the last sentence terminator in the prefix.
    last = None
    for m in re.finditer(r"[.!?…]+", prefix):
        last = m.end()

    if last is not None and last >= max(20, int(0.3 * len(prefix))):
        candidate = prefix[:last].rstrip()
        if _SENT_BOUNDARY_RE.search(candidate + " "):
            return candidate

    # Fallback: cut at last whitespace.
    ws = prefix.rfind(" ")
    if ws > 0:
        return prefix[:ws].rstrip()

    return prefix.strip()


# =========================
# Token counting (tiktoken optional)
# =========================

def _count_tokens(text: str, cfg: CleanCapConfig) -> int:
    if not text:
        return 0
    # Try to use tiktoken if installed.
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding(cfg.encoding_name)
        return len(enc.encode(text))
    except Exception:
        # Fallback: rough token estimate (word/punct pieces).
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _prefix_by_tokens(text: str, token_limit: int, cfg: CleanCapConfig) -> str:
    """
    Return a prefix of `text` that is within `token_limit` tokens.
    Uses tiktoken when available, otherwise approximates by regex tokens.
    """
    if not text or token_limit <= 0:
        return ""

    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding(cfg.encoding_name)
        toks = enc.encode(text)
        toks = toks[:token_limit]
        return enc.decode(toks)
    except Exception:
        # Fallback: approximate via regex tokens and join by character spans.
        matches = list(re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE))
        if len(matches) <= token_limit:
            return text
        end = matches[token_limit - 1].end()
        return text[:end]


# =========================
# Final formatting
# =========================

def _final_format(text: str) -> str:
    t = (text or "").replace("\u00a0", " ")
    # Normalize spaces.
    t = re.sub(r"[ \t]+", " ", t)
    # Normalize newlines: no more than 2.
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    # Remove excessive spaces.
    s = re.sub(r"[ \t]+", " ", s)
    # Trim each line.
    lines = [ln.strip() for ln in s.split("\n")]
    # Drop empty lines at ends but keep internal newlines (later normalized in final_format).
    s2 = "\n".join([ln for ln in lines if ln != ""]).strip()
    return s2


# =========================
# Stats helpers
# =========================

def _estimate_kept_chunks(paragraphs: Sequence[str], cfg: CleanCapConfig) -> int:
    """
    Kept chunks in debug are used as a rough indicator.
    We estimate it as the number of paragraphs that survive (bounded by original chunk count).
    """
    # This is an approximation; build_judge_payloads.py uses it only for debugging.
    return len(paragraphs)
