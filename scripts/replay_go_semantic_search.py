#!/usr/bin/env python3
"""Replay GO semantic search against saved enrichment fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.services.go_semantic_search import (  # noqa: E402
    GoSemanticStrategy,
    load_enrich_table,
    load_snapshot_metadata,
    overlap_at_k,
    parse_expected_go_ids,
    rank_go_terms,
)

DEFAULT_EXPECTED = REPO_ROOT / "tests/fixtures/IRF8_UC_top_GO_terms.md(1).md"
DEFAULT_SNAPSHOT = (
    REPO_ROOT / "tests/fixtures/snapshots/rs16940186_IRF8_enrich_tbl.csv"
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_input_table(args: argparse.Namespace):
    input_path = Path(args.input)
    enrich_tbl = load_enrich_table(str(input_path))
    phenotype = args.phenotype
    causal_gene = args.causal_gene
    meta_path = args.metadata
    if meta_path is None and input_path.name.endswith("_enrich_tbl.csv"):
        candidate = input_path.parent / input_path.name.replace(
            "_enrich_tbl.csv", "_metadata.json"
        )
        if candidate.is_file():
            meta_path = candidate
    if meta_path:
        meta = load_snapshot_metadata(str(meta_path))
        phenotype = phenotype or meta.get("phenotype")
        causal_gene = causal_gene or meta.get("causal_gene")
    return enrich_tbl, phenotype, causal_gene


def cmd_run(args: argparse.Namespace) -> int:
    load_dotenv(REPO_ROOT / ".env")
    strategy = GoSemanticStrategy(args.strategy)
    enrich_tbl, phenotype, causal_gene = _load_input_table(args)
    if not phenotype:
        raise SystemExit("--phenotype is required (or pass --metadata / snapshot CSV)")

    results = rank_go_terms(
        phenotype=phenotype,
        enrich_tbl=enrich_tbl,
        k=args.k,
        strategy=strategy,
        causal_gene=causal_gene,
    )
    output = args.output or Path(f"results/go_semantic_{strategy.value}.json")
    _write_json(output, results)

    print(f"Strategy: {strategy.value}")
    print(f"Phenotype: {phenotype}")
    print(f"Causal gene: {causal_gene or '(none)'}")
    print(f"Input rows: {len(enrich_tbl)}")
    print(f"Wrote {len(results)} ranked terms to {output}\n")
    print("| Rank | GO ID | Term | p-value | score |")
    print("| --- | --- | --- | --- | --- |")
    for row in results:
        score = row.get("score", row.get("similarity", 0.0))
        print(
            f"| {row['rank']} | {row['id']} | {row['name']} | {row['p']:.2e} | {score:.4f} |"
        )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    candidate = _load_results(args.candidate)
    expected_ids = parse_expected_go_ids(str(args.expected))
    candidate_ids = [row["id"] for row in candidate]
    expected_set = set(expected_ids)

    if args.baseline:
        baseline_overlap = overlap_at_k(
            [row["id"] for row in _load_results(args.baseline)], expected_ids, k=args.k
        )
        print(f"Baseline overlap@{args.k}: {baseline_overlap}")
    print(f"Candidate overlap@{args.k}: {overlap_at_k(candidate_ids, expected_ids, k=args.k)}")
    print()
    print("| Rank | GO ID | Term | in gold? |")
    print("| --- | --- | --- | --- |")
    for row in candidate[: args.k]:
        print(f"| {row['rank']} | {row['id']} | {row['name']} | {'yes' if row['id'] in expected_set else 'no'} |")
    return 0


def cmd_ab(args: argparse.Namespace) -> int:
    load_dotenv(REPO_ROOT / ".env")
    enrich_tbl, phenotype, causal_gene = _load_input_table(args)
    if not phenotype:
        raise SystemExit("--phenotype is required (or pass snapshot CSV with metadata)")

    expected_ids = parse_expected_go_ids(str(args.expected))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.output_dir / "baseline.json"
    improved_path = args.output_dir / "improved.json"

    for strategy, path in (
        (GoSemanticStrategy.BASELINE, baseline_path),
        (GoSemanticStrategy.IMPROVED, improved_path),
    ):
        results = rank_go_terms(
            phenotype=phenotype,
            enrich_tbl=enrich_tbl,
            k=args.k,
            strategy=strategy,
            causal_gene=causal_gene,
        )
        _write_json(path, results)

    baseline_overlap = overlap_at_k(
        [row["id"] for row in _load_results(baseline_path)], expected_ids, k=args.k
    )
    improved_overlap = overlap_at_k(
        [row["id"] for row in _load_results(improved_path)], expected_ids, k=args.k
    )
    print(f"Phenotype: {phenotype}")
    print(f"Causal gene: {causal_gene or '(none)'}")
    print(f"Input rows: {len(enrich_tbl)}")
    print(f"Baseline overlap@{args.k}: {baseline_overlap}")
    print(f"Improved overlap@{args.k}: {improved_overlap}")
    print(f"Wrote {baseline_path}")
    print(f"Wrote {improved_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run semantic search on a fixture")
    run.add_argument("--input", type=Path, default=DEFAULT_SNAPSHOT)
    run.add_argument("--metadata", type=Path, default=None)
    run.add_argument("--phenotype", default=None)
    run.add_argument("--causal-gene", default=None)
    run.add_argument("--strategy", choices=["baseline", "improved"], default="improved")
    run.add_argument("--k", type=int, default=10)
    run.add_argument("--output", type=Path, default=None)
    run.set_defaults(func=cmd_run)

    compare = sub.add_parser("compare", help="Compare results against expected gold terms")
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    compare.add_argument("--baseline", type=Path, default=None)
    compare.add_argument("--k", type=int, default=10)
    compare.set_defaults(func=cmd_compare)

    ab = sub.add_parser("ab", help="Run baseline vs improved and report overlap")
    ab.add_argument("--input", type=Path, default=DEFAULT_SNAPSHOT)
    ab.add_argument("--metadata", type=Path, default=None)
    ab.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    ab.add_argument("--phenotype", default=None)
    ab.add_argument("--causal-gene", default=None)
    ab.add_argument("--k", type=int, default=10)
    ab.add_argument("--output-dir", type=Path, default=Path("results/go_semantic_ab"))
    ab.set_defaults(func=cmd_ab)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
