from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from judge_prep.clean_cap import CleanCapConfig, clean_and_cap


@dataclass(frozen=True)
class MethodSpec:
    method: str
    dir: Path


@dataclass(frozen=True)
class ShuffleConfig:
    enabled: bool
    seed: int
    keys: List[str]
    mode: str  # "seed_plus_queryid_hash" | "seed_plus_index"


@dataclass(frozen=True)
class DebugConfig:
    enabled: bool
    output_dir: Path
    include_meta: bool


@dataclass(frozen=True)
class AppConfig:
    queries_file: Optional[Path]
    output_dir: Path
    methods: List[MethodSpec]
    clean_cap: CleanCapConfig
    shuffle: ShuffleConfig
    debug: DebugConfig


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _stable_u32_from_str(s: str) -> int:
    # Stable across runs and platforms, unlike Python's built-in hash()
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _load_config(config_path: Path) -> AppConfig:
    cfg_raw = _read_json(config_path)
    base = config_path.parent

    queries_file_raw = cfg_raw.get("queries_file")
    queries_file = (base / queries_file_raw).resolve() if queries_file_raw else None

    output_dir = (base / cfg_raw.get("output_dir", "artifacts/judge_payloads")).resolve()

    methods_raw = cfg_raw.get("methods", [])
    if not isinstance(methods_raw, list) or len(methods_raw) != 5:
        raise ValueError("Config must contain exactly 5 methods under 'methods'.")

    methods: List[MethodSpec] = []
    for m in methods_raw:
        if "method" not in m or "dir" not in m:
            raise ValueError("Each methods[] entry must have 'method' and 'dir'.")
        methods.append(
            MethodSpec(
                method=str(m["method"]),
                dir=(base / str(m["dir"])).resolve(),
            )
        )

    # Clean+cap config
    cc = cfg_raw.get("clean_cap", {})
    clean_cfg = CleanCapConfig(
        min_chars=int(cc.get("min_chars", 20)),
        token_budget_per_method=int(cc.get("token_budget_per_method", 350)),
        encoding_name=str(cc.get("encoding_name", "cl100k_base")),
        drop_captions=bool(cc.get("drop_captions", True)),
        drop_headings=bool(cc.get("drop_headings", True)),
        drop_table_headers=bool(cc.get("drop_table_headers", True)),
        drop_trailing_colon_fragments=bool(cc.get("drop_trailing_colon_fragments", False)),
        heading_max_words=int(cc.get("heading_max_words", 8)),
        heading_max_chars=int(cc.get("heading_max_chars", 80)),
        table_header_min_words=int(cc.get("table_header_min_words", 6)),
        table_header_max_punct=int(cc.get("table_header_max_punct", 1)),
        table_header_uppercase_ratio=float(cc.get("table_header_uppercase_ratio", 0.35)),
        max_chunks=cc.get("max_chunks", None),
        joiner=str(cc.get("joiner", "\n")),
    )

    # Shuffle config
    sh = cfg_raw.get("shuffle", {})
    shuffle_cfg = ShuffleConfig(
        enabled=bool(sh.get("enabled", True)),
        seed=int(sh.get("seed", 42)),
        keys=[str(k) for k in sh.get("keys", ["A", "B", "C", "D", "E"])],
        mode=str(sh.get("mode", "seed_plus_queryid_hash")),
    )
    if len(shuffle_cfg.keys) != 5:
        raise ValueError("shuffle.keys must contain exactly 5 labels (e.g. A..E).")
    if len(set(shuffle_cfg.keys)) != 5:
        raise ValueError("shuffle.keys must be unique.")
    if shuffle_cfg.mode not in ("seed_plus_queryid_hash", "seed_plus_index"):
        raise ValueError("shuffle.mode must be 'seed_plus_queryid_hash' or 'seed_plus_index'.")

    # Debug config (optional)
    dbg = cfg_raw.get("debug", {}) if isinstance(cfg_raw.get("debug", {}), dict) else {}
    debug_cfg = DebugConfig(
        enabled=bool(dbg.get("enabled", False)),
        output_dir=(base / str(dbg.get("output_dir", "artifacts/judge_payloads_debug"))).resolve(),
        include_meta=bool(dbg.get("include_meta", True)),
    )

    return AppConfig(
        queries_file=queries_file,
        output_dir=output_dir,
        methods=methods,
        clean_cap=clean_cfg,
        shuffle=shuffle_cfg,
        debug=debug_cfg,
    )


def _list_query_ids_from_methods(methods: List[MethodSpec]) -> List[str]:
    # Use intersection of filenames across methods to ensure every method has Qxxx.json
    sets = []
    for m in methods:
        files = {p.stem for p in m.dir.glob("Q*.json") if p.is_file()}
        sets.append(files)
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


def _read_query_text(queries_file: Optional[Path], qid: str, fallback_query: str) -> str:
    # If queries_file is absent, use the query from method files.
    if queries_file is None:
        return fallback_query

    path = queries_file
    if not path.exists():
        return fallback_query

    # queries.jsonl format: {"id": "...", "query": "..."}
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if str(obj.get("id")) == qid:
                    return str(obj.get("query", fallback_query))
        return fallback_query

    # queries.json format: list or dict
    obj = _read_json(path)
    if isinstance(obj, list):
        for it in obj:
            if str(it.get("id")) == qid:
                return str(it.get("query", fallback_query))
        return fallback_query
    if isinstance(obj, dict):
        if qid in obj:
            return str(obj[qid])
    return fallback_query


def _load_method_result(spec: MethodSpec, qid: str) -> Tuple[str, List[str]]:
    # Returns (query_text, chunks)
    in_path = spec.dir / f"{qid}.json"
    data = _read_json(in_path)

    file_qid = str(data.get("id", "")).strip()
    if file_qid and file_qid != qid:
        raise ValueError(f"ID mismatch in {in_path}: expected {qid}, got {file_qid}")

    query = str(data.get("query", "")).strip()

    output = data.get("output", [])
    if not isinstance(output, list):
        raise ValueError(f"Invalid 'output' field in {in_path} (expected list).")

    chunks = [str(x) for x in output]
    return query, chunks


def _make_rng_for_query(app_cfg: AppConfig, qid: str, index: int) -> random.Random:
    if not app_cfg.shuffle.enabled:
        # Fixed mapping: keep methods order as provided by config, keys order as provided.
        return random.Random(0)

    base_seed = int(app_cfg.shuffle.seed)
    if app_cfg.shuffle.mode == "seed_plus_queryid_hash":
        per_q = _stable_u32_from_str(qid)
        seed = (base_seed + per_q) & 0xFFFFFFFF
        return random.Random(seed)

    # seed_plus_index
    seed = (base_seed + int(index)) & 0xFFFFFFFF
    return random.Random(seed)


def build_payload_for_query(app_cfg: AppConfig, qid: str, index: int) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    # 1) Load and clean all methods
    cleaned_variants: List[Tuple[str, str]] = []  # (method_name, cleaned_text)
    debug_variants: List[Dict[str, Any]] = []

    query_text_fallback: Optional[str] = None

    for spec in app_cfg.methods:
        query, chunks = _load_method_result(spec, qid)
        if query_text_fallback is None:
            query_text_fallback = query

        res = clean_and_cap(chunks, app_cfg.clean_cap, query=query_text_fallback or query)
        cleaned_variants.append((spec.method, res.text))

        if app_cfg.debug.enabled:
            debug_variants.append(
                {
                    "method": spec.method,
                    "tokens": res.tokens,
                    "stats": {
                        "in_chunks": res.stats.in_chunks,
                        "kept_chunks": res.stats.kept_chunks,
                        "dropped_empty": res.stats.dropped_empty,
                        "dropped_dedup": res.stats.dropped_dedup,
                        "dropped_caption": res.stats.dropped_caption,
                        "dropped_heading": res.stats.dropped_heading,
                        "dropped_table_header": res.stats.dropped_table_header,
                        "dropped_trailing_colon": res.stats.dropped_trailing_colon,
                        "truncated": res.stats.truncated,
                    },
                }
            )

    if query_text_fallback is None:
        query_text_fallback = ""

    query_text = _read_query_text(app_cfg.queries_file, qid, query_text_fallback)

    # 2) Shuffle and assign to keys
    keys = list(app_cfg.shuffle.keys)
    variants = list(cleaned_variants)

    if app_cfg.shuffle.enabled:
        rng = _make_rng_for_query(app_cfg, qid, index)
        rng.shuffle(variants)
    else:
        # No shuffle: keep config order, but still assign keys in order.
        pass

    contexts_for_judge: Dict[str, str] = {}
    private_mapping: Dict[str, str] = {}

    for k, (method_name, text) in zip(keys, variants, strict=True):
        contexts_for_judge[k] = text
        private_mapping[k] = method_name

    payload: Dict[str, Any] = {
        "id": qid,
        "query": query_text,
        "contexts_for_judge": contexts_for_judge,
        "private_mapping": private_mapping,
    }

    debug_payload: Optional[Dict[str, Any]] = None
    if app_cfg.debug.enabled:
        debug_payload = {
            "id": qid,
            "query": query_text,
            "shuffle": {
                "enabled": app_cfg.shuffle.enabled,
                "seed": app_cfg.shuffle.seed,
                "mode": app_cfg.shuffle.mode,
                "keys": keys,
                "private_mapping": private_mapping,
            },
            "clean_cap": {
                "token_budget_per_method": app_cfg.clean_cap.token_budget_per_method,
                "encoding_name": app_cfg.clean_cap.encoding_name,
            },
            "variants": debug_variants,
        }
        if app_cfg.debug.include_meta:
            debug_payload["meta"] = {
                "note": "Debug file is not intended for judges.",
            }

    return payload, debug_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build offline blind judge payloads with deterministic shuffling.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/judge_prep.json",
        help="Path to JSON config (default: configs/judge_prep.json).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Optional: build only one query id, e.g. Q010.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: build only first N queries (after sorting).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    app_cfg = _load_config(cfg_path)

    qids = _list_query_ids_from_methods(app_cfg.methods)
    if args.only:
        if args.only not in qids:
            raise ValueError(f"Query id {args.only} not found in all method folders.")
        qids = [args.only]

    if args.limit is not None:
        qids = qids[: max(0, int(args.limit))]

    if not qids:
        print("No common Q*.json files found across all method folders.")
        return 1

    app_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if app_cfg.debug.enabled:
        app_cfg.debug.output_dir.mkdir(parents=True, exist_ok=True)

    built = 0
    for idx, qid in enumerate(qids):
        payload, debug_payload = build_payload_for_query(app_cfg, qid, idx)

        out_path = app_cfg.output_dir / f"{qid}.json"
        _write_json(out_path, payload)

        if app_cfg.debug.enabled and debug_payload is not None:
            dbg_path = app_cfg.debug.output_dir / f"{qid}.debug.json"
            _write_json(dbg_path, debug_payload)

        built += 1
        print(f"[OK] {qid} -> {out_path}")

    print(f"Done. Built {built} payload(s) into: {app_cfg.output_dir}")
    if app_cfg.debug.enabled:
        print(f"Debug files saved into: {app_cfg.debug.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
