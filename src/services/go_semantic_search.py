"""Semantic ranking of enriched GO terms against a GWAS phenotype (CLI / A-B testing)."""

from __future__ import annotations

import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import openai
import pandas as pd
import scipy.spatial
from loguru import logger

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_HYBRID_ALPHA = 0.7
MAX_GENES_IN_DOC = 20

_FIXTURE_LINE_RE = re.compile(
    r"^(?P<term>.+?)\s+\((?P<go_id>GO:\d+)\)\s+\|\s+adj_p=(?P<adj_p>[\d.eE+-]+)\s+\|\s+genes:\s*(?P<genes>.+)$"
)
_GO_ID_RE = re.compile(r"GO:\d+")


class GoSemanticStrategy(str, Enum):
    BASELINE = "baseline"
    IMPROVED = "improved"


def build_go_search_query(
    phenotype: str,
    causal_gene: Optional[str] = None,
    strategy: GoSemanticStrategy = GoSemanticStrategy.IMPROVED,
) -> str:
    if strategy == GoSemanticStrategy.BASELINE:
        return phenotype.strip()
    parts = [f"GWAS phenotype: {phenotype.strip()}."]
    if causal_gene:
        parts.append(f"Causal gene at locus: {causal_gene.strip()}.")
    parts.append(
        "Identify GO biological processes most relevant to this disease mechanism."
    )
    return " ".join(parts)


def build_document_text(row: pd.Series, strategy: GoSemanticStrategy) -> str:
    term = str(row["Term"]).strip()
    desc = str(row.get("Desc", "")).strip()
    if not desc or desc == "NA" or desc == "GO":
        desc = term
    text = f"{term} [SEP] {desc}"
    if strategy == GoSemanticStrategy.IMPROVED:
        genes = str(row.get("Genes", "")).strip()
        if genes:
            gene_list = genes.replace(",", ";").split(";")[:MAX_GENES_IN_DOC]
            gene_text = "; ".join(g.strip() for g in gene_list if g.strip())
            if gene_text:
                text += f" [SEP] Genes: {gene_text}"
    return text


def normalize_neg_log_p(pvalues: pd.Series) -> pd.Series:
    p = pvalues.astype(float).clip(lower=1e-300)
    neg_log = -np.log10(p)
    min_v, max_v = neg_log.min(), neg_log.max()
    if max_v == min_v:
        return pd.Series(1.0, index=pvalues.index)
    return (neg_log - min_v) / (max_v - min_v)


def hybrid_score(
    similarity: pd.Series,
    pvalues: pd.Series,
    alpha: float = DEFAULT_HYBRID_ALPHA,
) -> pd.Series:
    return alpha * similarity + (1.0 - alpha) * normalize_neg_log_p(pvalues)


def _embed_texts(texts: list[str], model: str) -> list[list[float]]:
    client = openai.Client()
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def _rows_to_results(ranked: pd.DataFrame, score_column: str) -> list[dict]:
    results = []
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        genes_raw = str(row.get("Genes", ""))
        genes = [g.strip() for g in genes_raw.replace(",", ";").split(";") if g.strip()]
        entry = {
            "id": str(row["ID"]).strip(),
            "name": str(row["Term"]).strip(),
            "genes": genes,
            "p": float(row["Adjusted P-value"]),
            "rank": rank,
            "similarity": float(row.get("similarity", 0.0)),
        }
        if score_column in row:
            entry["score"] = float(row[score_column])
        results.append(entry)
    return results


def rank_go_terms(
    phenotype: str,
    enrich_tbl: pd.DataFrame,
    k: int = 10,
    strategy: GoSemanticStrategy = GoSemanticStrategy.IMPROVED,
    causal_gene: Optional[str] = None,
    embedding_model: Optional[str] = None,
    hybrid_alpha: float = DEFAULT_HYBRID_ALPHA,
) -> list[dict]:
    if enrich_tbl is None or len(enrich_tbl) == 0:
        logger.warning("Empty enrichment table provided for GO semantic search")
        return []

    model = embedding_model or os.getenv("GO_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    data = enrich_tbl.copy()
    texts = [build_document_text(row, strategy) for _, row in data.iterrows()]
    if not texts:
        return []

    query = build_go_search_query(phenotype, causal_gene=causal_gene, strategy=strategy)
    doc_embeddings = _embed_texts(texts, model)
    query_embedding = _embed_texts([query], model)[0]

    data["similarity"] = [
        1.0 - scipy.spatial.distance.cosine(emb, query_embedding)
        for emb in doc_embeddings
    ]

    if strategy == GoSemanticStrategy.IMPROVED:
        data["score"] = hybrid_score(
            data["similarity"], data["Adjusted P-value"], alpha=hybrid_alpha
        )
        score_column = "score"
    else:
        data["score"] = data["similarity"]
        score_column = "similarity"

    ranked = data.sort_values(score_column, ascending=False).head(k)
    return _rows_to_results(ranked, score_column)


def parse_fixture_go_terms(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("Total GO terms:"):
                continue
            match = _FIXTURE_LINE_RE.match(line)
            if not match:
                continue
            rows.append(
                {
                    "ID": match.group("go_id"),
                    "Term": match.group("term").strip(),
                    "Desc": match.group("term").strip(),
                    "Adjusted P-value": float(match.group("adj_p")),
                    "Genes": match.group("genes").strip(),
                }
            )
    if not rows:
        raise ValueError(f"No GO terms parsed from fixture: {path}")
    return pd.DataFrame(rows)


def load_enrich_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        table = pd.read_csv(path)
        required = {"ID", "Term", "Desc", "Adjusted P-value", "Genes"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        return table
    return parse_fixture_go_terms(path)


def load_snapshot_metadata(path: str) -> dict[str, Any]:
    csv_path = Path(path)
    if csv_path.suffix == ".json":
        meta_path = csv_path
    elif csv_path.name.endswith("_enrich_tbl.csv"):
        meta_path = csv_path.with_name(
            csv_path.name[: -len("_enrich_tbl.csv")] + "_metadata.json"
        )
    else:
        meta_path = csv_path.with_suffix(".json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def parse_expected_go_ids(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        content = handle.read()
    seen: set[str] = set()
    ordered: list[str] = []
    for go_id in _GO_ID_RE.findall(content):
        if go_id not in seen:
            seen.add(go_id)
            ordered.append(go_id)
    return ordered


def overlap_at_k(result_ids: list[str], expected_ids: list[str], k: int = 10) -> int:
    return len(set(result_ids[:k]) & set(expected_ids))


def snapshot_enabled() -> bool:
    return os.getenv("GO_ENRICH_SNAPSHOT", "").strip().lower() in {"1", "true", "yes"}


def snapshot_dir() -> Path:
    return Path(os.getenv("GO_ENRICH_SNAPSHOT_DIR", "tests/fixtures/snapshots"))


def _safe_snapshot_part(value: str | None, fallback: str = "unknown") -> str:
    if not value or not str(value).strip():
        return fallback
    return re.sub(r"[^\w.-]", "_", str(value).strip()).strip("_") or fallback


def snapshot_stem(variant: str, causal_gene: str, tissue: str | None) -> str:
    tissue_part = _safe_snapshot_part(tissue, "non_tissue")
    return f"{_safe_snapshot_part(variant)}_{_safe_snapshot_part(causal_gene)}_{tissue_part}"


def maybe_save_enrich_snapshot(
    enrich_tbl: pd.DataFrame,
    *,
    phenotype: str,
    variant: str,
    causal_gene: str,
    tissue: str | None,
    project_id: str,
    hypothesis_id: str,
    meta: dict[str, Any] | None = None,
    enrich_tbl_all: pd.DataFrame | None = None,
) -> str | None:
    """Save enrich_tbl to CSV when GO_ENRICH_SNAPSHOT is enabled."""
    if not snapshot_enabled():
        return None

    out_dir = snapshot_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = snapshot_stem(variant, causal_gene, tissue)
    csv_path = out_dir / f"{stem}_enrich_tbl.csv"
    meta_path = out_dir / f"{stem}_metadata.json"

    enrich_tbl.to_csv(csv_path, index=False)
    all_csv_path = None
    if enrich_tbl_all is not None and len(enrich_tbl_all) > 0:
        all_csv_path = out_dir / f"{stem}_enrich_tbl_all.csv"
        enrich_tbl_all.to_csv(all_csv_path, index=False)

    metadata = {
        "phenotype": phenotype,
        "variant": variant,
        "causal_gene": causal_gene,
        "tissue": tissue,
        "project_id": project_id,
        "hypothesis_id": hypothesis_id,
        "row_count": len(enrich_tbl),
        "row_count_all": len(enrich_tbl_all) if enrich_tbl_all is not None else None,
        "snapshot_stem": stem,
        "enrich_tbl_file": csv_path.name,
        "enrich_tbl_all_file": all_csv_path.name if all_csv_path else None,
        "note": (
            "enrich_tbl = adj p < 0.05 (input to semantic search); "
            "enrich_tbl_all = full enrichr output before p filter"
        ),
        **(meta or {}),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(
        f"[GO snapshot] Saved {len(enrich_tbl)} significant rows to {csv_path}"
        + (
            f" and {len(enrich_tbl_all)} full enrichr rows to {all_csv_path}"
            if all_csv_path
            else ""
        )
    )
    return str(csv_path)
