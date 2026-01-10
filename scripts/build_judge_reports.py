#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build aggregated LLM-as-judge reports.

Expected judge outputs layout:
  artifacts/judge_outputs/<MODEL_OR_ANY_SUBDIR?>/Q001_1.json
  artifacts/judge_outputs/Q001_2.json
  ...

We will recursively scan --judge_outputs_dir for *.json and parse:
- qid and replica from filename: Q###_N.json
- model from JSON field if present

Judge output schema supported:
A) Wrapped:
   {
     "id": "Q002",
     "replica": 2,
     "model": "...",
     "judge_response": { <the strict judge schema> }
   }

B) Flat (fallback):
   { <the strict judge schema>, "id": "Q002", ... }

We also scan --judge_payloads_dir for payloads with:
  {
    "id": "Q002",
    "query": "...",
    "private_mapping": { "A": "OntologyRAG", ... }
  }

Outputs (CSV):
- <prefix>_long.csv    : per (qid, replica, letter)
- <prefix>_runs.csv    : per (qid, replica)
- <prefix>_summary.csv : per (qid, method) aggregated over replicas (mean + std)
- <prefix>_winners.csv : per (qid) winner stats over replicas
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LETTERS = ["A", "B", "C", "D", "E"]


# ---------------------------
# Helpers
# ---------------------------

def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def safe_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    return str(x)


def mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return None
    return sum(xs) / len(xs)


def std(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if len(xs) < 2:
        return 0.0 if xs else None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def dump_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def parse_qid_replica_from_filename(p: Path) -> Tuple[Optional[str], Optional[int]]:
    """
    Supports:
      Q002_2.json -> ("Q002", 2)
      Q002.json   -> ("Q002", None)  (not recommended but supported)
    """
    m = re.match(r"^(Q\d{3})(?:_(\d+))?\.json$", p.name)
    if not m:
        return None, None
    qid = m.group(1)
    replica = safe_int(m.group(2), None)
    return qid, replica


def get_judge_block(raw: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns (judge_dict, error_string)
    """
    if isinstance(raw.get("judge_response"), dict):
        return raw["judge_response"], None
    # fallback: maybe stored flat (already strict schema)
    # minimal check: has required keys
    required = ["relevance", "answerability", "noise", "overall", "winner", "ranking", "failure_letters", "confidence", "rationales"]
    if all(k in raw for k in required):
        return raw, None
    return None, "missing judge_response (and not a flat strict schema)"


def normalize_metric_map(d: Any) -> Dict[str, Optional[int]]:
    """
    Expect {"A": 0..5, ...}. Returns {letter: int|None}.
    """
    out: Dict[str, Optional[int]] = {L: None for L in LETTERS}
    if not isinstance(d, dict):
        return out
    for L in LETTERS:
        v = d.get(L)
        out[L] = safe_int(v, None)
    return out


def list_json_files_recursive(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*.json") if p.is_file()])


# ---------------------------
# Data
# ---------------------------

@dataclass
class PayloadInfo:
    qid: str
    query: str
    mapping: Dict[str, str]  # letter -> method


@dataclass
class RunRecord:
    qid: str
    replica: Optional[int]
    model: str
    file_path: str
    parse_ok: bool
    parse_error: str

    query: str

    # from judge_response
    relevance: Dict[str, Optional[int]]
    answerability: Dict[str, Optional[int]]
    noise: Dict[str, Optional[int]]
    overall: Dict[str, Optional[int]]

    winner_letter: str
    ranking_letters: List[str]
    failure_letters: List[str]
    confidence: Optional[int]
    rationales: Dict[str, str]

    # from payload mapping
    letter_to_method: Dict[str, str]


# ---------------------------
# Build reports
# ---------------------------

def load_payloads(payload_dir: Path) -> Dict[str, PayloadInfo]:
    payloads: Dict[str, PayloadInfo] = {}
    for p in sorted(payload_dir.glob("Q*.json")):
        raw, err = load_json(p)
        if raw is None:
            continue
        qid = safe_str(raw.get("id"))
        if not re.match(r"^Q\d{3}$", qid):
            continue
        query = safe_str(raw.get("query"))
        mapping_raw = raw.get("private_mapping") or {}
        mapping: Dict[str, str] = {}
        if isinstance(mapping_raw, dict):
            for L in LETTERS:
                if L in mapping_raw:
                    mapping[L] = safe_str(mapping_raw[L])
        payloads[qid] = PayloadInfo(qid=qid, query=query, mapping=mapping)
    return payloads


def load_runs(judge_outputs_dir: Path, payloads: Dict[str, PayloadInfo]) -> List[RunRecord]:
    runs: List[RunRecord] = []
    for p in list_json_files_recursive(judge_outputs_dir):
        qid, replica = parse_qid_replica_from_filename(p)
        if qid is None:
            continue

        raw, err = load_json(p)
        if raw is None:
            # still produce a run row (parse fail)
            payload = payloads.get(qid)
            runs.append(
                RunRecord(
                    qid=qid,
                    replica=replica,
                    model="",
                    file_path=str(p),
                    parse_ok=False,
                    parse_error=err or "unknown json error",
                    query=payload.query if payload else "",
                    relevance={L: None for L in LETTERS},
                    answerability={L: None for L in LETTERS},
                    noise={L: None for L in LETTERS},
                    overall={L: None for L in LETTERS},
                    winner_letter="",
                    ranking_letters=[],
                    failure_letters=[],
                    confidence=None,
                    rationales={L: "" for L in LETTERS},
                    letter_to_method=(payload.mapping if payload else {}),
                )
            )
            continue

        payload = payloads.get(qid)
        judge, jerr = get_judge_block(raw)

        model = safe_str(raw.get("model"), "")
        # also accept older fields
        if not model:
            model = safe_str(raw.get("judge_model"), "")

        if judge is None:
            runs.append(
                RunRecord(
                    qid=qid,
                    replica=replica,
                    model=model,
                    file_path=str(p),
                    parse_ok=False,
                    parse_error=jerr or "unknown schema error",
                    query=payload.query if payload else "",
                    relevance={L: None for L in LETTERS},
                    answerability={L: None for L in LETTERS},
                    noise={L: None for L in LETTERS},
                    overall={L: None for L in LETTERS},
                    winner_letter="",
                    ranking_letters=[],
                    failure_letters=[],
                    confidence=None,
                    rationales={L: "" for L in LETTERS},
                    letter_to_method=(payload.mapping if payload else {}),
                )
            )
            continue

        rel = normalize_metric_map(judge.get("relevance"))
        ans = normalize_metric_map(judge.get("answerability"))
        noi = normalize_metric_map(judge.get("noise"))
        ovl = normalize_metric_map(judge.get("overall"))

        winner_letter = safe_str(judge.get("winner"), "")
        ranking_letters = judge.get("ranking") if isinstance(judge.get("ranking"), list) else []
        ranking_letters = [safe_str(x) for x in ranking_letters if safe_str(x) in LETTERS]

        failure_letters = judge.get("failure_letters") if isinstance(judge.get("failure_letters"), list) else []
        failure_letters = sorted({safe_str(x) for x in failure_letters if safe_str(x) in LETTERS})

        confidence = safe_int(judge.get("confidence"), None)

        rats_raw = judge.get("rationales") if isinstance(judge.get("rationales"), dict) else {}
        rationales = {L: safe_str(rats_raw.get(L), "") for L in LETTERS}

        runs.append(
            RunRecord(
                qid=qid,
                replica=replica,
                model=model,
                file_path=str(p),
                parse_ok=True,
                parse_error="",
                query=payload.query if payload else "",
                relevance=rel,
                answerability=ans,
                noise=noi,
                overall=ovl,
                winner_letter=winner_letter if winner_letter in LETTERS else "",
                ranking_letters=ranking_letters,
                failure_letters=failure_letters,
                confidence=confidence,
                rationales=rationales,
                letter_to_method=(payload.mapping if payload else {}),
            )
        )

    # stable ordering: qid then replica (None last)
    def key(r: RunRecord):
        rep = r.replica if r.replica is not None else 10**9
        return (r.qid, rep, r.file_path)

    return sorted(runs, key=key)


def build_long_rows(runs: List[RunRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in runs:
        for L in LETTERS:
            method = r.letter_to_method.get(L, "")
            rows.append(
                {
                    "qid": r.qid,
                    "replica": r.replica if r.replica is not None else "",
                    "model": r.model,
                    "query": r.query,
                    "letter": L,
                    "method": method,
                    "parse_ok": int(r.parse_ok),
                    "parse_error": r.parse_error,
                    "is_failure_letter": int(L in set(r.failure_letters)),
                    "relevance": r.relevance.get(L),
                    "answerability": r.answerability.get(L),
                    "noise": r.noise.get(L),
                    "overall": r.overall.get(L),
                    "rationale": r.rationales.get(L, ""),
                    "source_file": r.file_path,
                }
            )
    return rows


def build_runs_rows(runs: List[RunRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in runs:
        # Map winner/ranking to methods via payload mapping
        winner_method = r.letter_to_method.get(r.winner_letter, "") if r.winner_letter else ""
        ranking_methods = [r.letter_to_method.get(L, "") for L in r.ranking_letters]
        rows.append(
            {
                "qid": r.qid,
                "replica": r.replica if r.replica is not None else "",
                "model": r.model,
                "query": r.query,
                "parse_ok": int(r.parse_ok),
                "parse_error": r.parse_error,
                "winner_letter": r.winner_letter,
                "winner_method": winner_method,
                "confidence": r.confidence,
                "failure_letters": ",".join(r.failure_letters),
                "ranking_letters": ",".join(r.ranking_letters),
                "ranking_methods": ",".join(ranking_methods),
                # helpful: overall per letter
                "overall_A": r.overall.get("A"),
                "overall_B": r.overall.get("B"),
                "overall_C": r.overall.get("C"),
                "overall_D": r.overall.get("D"),
                "overall_E": r.overall.get("E"),
                "source_file": r.file_path,
            }
        )
    return rows


def build_summary_rows(long_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate over replicas for each (qid, method).
    We only use rows with parse_ok=1 and with non-empty method.
    """
    # bucket: (qid, method) -> metric lists
    bucket: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    query_by_qid: Dict[str, str] = {}

    for row in long_rows:
        qid = safe_str(row.get("qid"))
        query_by_qid[qid] = safe_str(row.get("query"))
        if safe_int(row.get("parse_ok"), 0) != 1:
            continue
        method = safe_str(row.get("method"))
        if not method:
            continue
        key = (qid, method)
        bucket.setdefault(key, {"relevance": [], "answerability": [], "noise": [], "overall": []})
        for m in ["relevance", "answerability", "noise", "overall"]:
            v = row.get(m)
            if v is None or v == "":
                continue
            try:
                bucket[key][m].append(float(v))
            except Exception:
                pass

    rows: List[Dict[str, Any]] = []
    for (qid, method), metrics in sorted(bucket.items()):
        rows.append(
            {
                "qid": qid,
                "query": query_by_qid.get(qid, ""),
                "method": method,
                "n": max(len(metrics["overall"]), len(metrics["relevance"]), len(metrics["answerability"]), len(metrics["noise"])),
                "relevance_mean": mean(metrics["relevance"]),
                "relevance_std": std(metrics["relevance"]),
                "answerability_mean": mean(metrics["answerability"]),
                "answerability_std": std(metrics["answerability"]),
                "noise_mean": mean(metrics["noise"]),
                "noise_std": std(metrics["noise"]),
                "overall_mean": mean(metrics["overall"]),
                "overall_std": std(metrics["overall"]),
            }
        )
    return rows


def build_winners_rows(runs_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Winner distribution per qid (across replicas).
    Uses winner_method when available; counts ties as empty winner_letter (winner_method empty).
    """
    # qid -> method -> count
    counts: Dict[str, Dict[str, int]] = {}
    totals: Dict[str, int] = {}
    queries: Dict[str, str] = {}

    for rr in runs_rows:
        qid = safe_str(rr.get("qid"))
        queries[qid] = safe_str(rr.get("query"))
        if safe_int(rr.get("parse_ok"), 0) != 1:
            continue
        totals[qid] = totals.get(qid, 0) + 1
        m = safe_str(rr.get("winner_method"))
        if not m:
            m = "__TIE_OR_EMPTY__"
        counts.setdefault(qid, {})
        counts[qid][m] = counts[qid].get(m, 0) + 1

    rows: List[Dict[str, Any]] = []
    for qid in sorted(totals.keys()):
        total = totals[qid]
        # build a compact representation
        items = sorted(counts.get(qid, {}).items(), key=lambda kv: (-kv[1], kv[0]))
        top_method, top_count = items[0] if items else ("", 0)
        rows.append(
            {
                "qid": qid,
                "query": queries.get(qid, ""),
                "n_runs": total,
                "top_winner_method": "" if top_method == "__TIE_OR_EMPTY__" else top_method,
                "top_winner_count": top_count,
                "top_winner_share": (top_count / total) if total else None,
                "winners_breakdown": ";".join([f"{k}:{v}" for k, v in items]),
            }
        )
    return rows


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_outputs_dir", type=str, required=True)
    ap.add_argument("--judge_payloads_dir", type=str, required=True)
    ap.add_argument("--reports_dir", type=str, required=True)
    ap.add_argument("--prefix", type=str, default="judge")
    ap.add_argument("--fmt", type=str, default="csv", choices=["csv"])
    args = ap.parse_args()

    judge_outputs_dir = Path(args.judge_outputs_dir)
    judge_payloads_dir = Path(args.judge_payloads_dir)
    reports_dir = Path(args.reports_dir)
    prefix = args.prefix

    payloads = load_payloads(judge_payloads_dir)
    runs = load_runs(judge_outputs_dir, payloads)

    long_rows = build_long_rows(runs)
    runs_rows = build_runs_rows(runs)
    summary_rows = build_summary_rows(long_rows)
    winners_rows = build_winners_rows(runs_rows)

    # Write CSVs
    dump_csv(
        reports_dir / f"{prefix}_long.csv",
        long_rows,
        [
            "qid", "replica", "model", "query", "letter", "method",
            "parse_ok", "parse_error", "is_failure_letter",
            "relevance", "answerability", "noise", "overall",
            "rationale", "source_file",
        ],
    )

    dump_csv(
        reports_dir / f"{prefix}_runs.csv",
        runs_rows,
        [
            "qid", "replica", "model", "query",
            "parse_ok", "parse_error",
            "winner_letter", "winner_method", "confidence",
            "failure_letters", "ranking_letters", "ranking_methods",
            "overall_A", "overall_B", "overall_C", "overall_D", "overall_E",
            "source_file",
        ],
    )

    dump_csv(
        reports_dir / f"{prefix}_summary.csv",
        summary_rows,
        [
            "qid", "query", "method", "n",
            "relevance_mean", "relevance_std",
            "answerability_mean", "answerability_std",
            "noise_mean", "noise_std",
            "overall_mean", "overall_std",
        ],
    )

    dump_csv(
        reports_dir / f"{prefix}_winners.csv",
        winners_rows,
        [
            "qid", "query", "n_runs",
            "top_winner_method", "top_winner_count", "top_winner_share",
            "winners_breakdown",
        ],
    )

    print(f"Wrote reports to: {reports_dir.resolve()}")
    print(f" - {prefix}_long.csv")
    print(f" - {prefix}_runs.csv")
    print(f" - {prefix}_summary.csv")
    print(f" - {prefix}_winners.csv")


if __name__ == "__main__":
    main()
