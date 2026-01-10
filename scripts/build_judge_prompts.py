from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from judge_prep.prompt_builder import (
    build_prompts,
    prompts_to_markdown,
    prompts_to_messages,
)


def read_json(path: Path) -> Dict[str, Any]:
    """Read a UTF-8 encoded JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    """Write text to a file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write a JSON file with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LLM-as-judge prompts from artifacts/judge_payloads/*.json"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="artifacts/judge_payloads",
        help="Directory containing Qxxx.json judge payloads",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/judge_prompts",
        help="Directory to write generated judge prompts",
    )
    parser.add_argument(
        "--write_md",
        action="store_true",
        help="Write one Markdown file per query (Qxxx.md)",
    )
    parser.add_argument(
        "--write_messages_json",
        action="store_true",
        help="Write one messages JSON per query (Qxxx.messages.json)",
    )
    parser.add_argument(
        "--write_jsonl",
        action="store_true",
        help="Write prompts.jsonl with one {id, messages} per line",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if not in_dir.exists():
        raise SystemExit(f"Input directory does not exist: {in_dir}")

    payload_paths = sorted(in_dir.glob("Q*.json"))
    if not payload_paths:
        raise SystemExit(f"No payloads found in {in_dir} (expected Q*.json)")

    jsonl_lines: List[str] = []

    for payload_path in payload_paths:
        payload = read_json(payload_path)

        # Build prompts (usually returns a list of length 1)
        prompts = build_prompts(payload)

        # Determine query id
        qid = (
            payload.get("id")
            if isinstance(payload, dict)
            else payload_path.stem
        )
        if not qid:
            qid = payload_path.stem

        # Write Markdown prompt (human-readable)
        if args.write_md:
            md = prompts_to_markdown(prompts, title=qid)
            write_text(out_dir / f"{qid}.md", md)

        # Write per-query messages JSON
        if args.write_messages_json:
            msg_obj = {
                "id": qid,
                "messages": prompts_to_messages(prompts),
            }
            write_json(out_dir / f"{qid}.messages.json", msg_obj)

        # Accumulate JSONL lines
        if args.write_jsonl:
            msg_obj = {
                "id": qid,
                "messages": prompts_to_messages(prompts),
            }
            jsonl_lines.append(json.dumps(msg_obj, ensure_ascii=False))

    # Write aggregated JSONL file
    if args.write_jsonl:
        write_text(
            out_dir / "prompts.jsonl",
            "\n".join(jsonl_lines) + "\n",
        )

    print(
        f"Done. Processed {len(payload_paths)} payload(s). "
        f"Output written to {out_dir}"
    )


if __name__ == "__main__":
    main()
