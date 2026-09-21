"""Semantic ranking of enriched GO terms against a GWAS phenotype.

Primary strategy: **hybrid** (GO_SEMANTIC_STRATEGY=llm or hybrid).
    1. Score the full enrichment table (embedding similarity + p-value hybrid score).
    2. Build a shortlist via `prefilter_go_candidates_union`: guaranteed top-N by
       adjusted p-value, topped up with the highest-scoring remaining terms.
    3. Send the shortlist to an LLM (local gemma4 or OpenAI) for a single rerank call.
Switch model backend with GO_LLM_BACKEND=local|openai; no code change required.

`baseline` / `improved` / `embedding` and the legacy `llm-local` / `llm-openai`
(full-table batched map-reduce) strategies are kept for benchmarking via
scripts/replay_go_semantic_search.py but are not the production default.
"""

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
DEFAULT_LLM_MAX_CANDIDATES = 250  # 0 in env = auto-cap; see _resolve_llm_candidate_cap
DEFAULT_LLM_PREFILTER_K = 250
DEFAULT_LLM_PREFILTER_PVALUE_K = 150
DEFAULT_LLM_MODEL = "gemma4"
DEFAULT_OPENAI_LLM_MODEL = "gpt-4o"

ALL_AB_STRATEGIES = (
    "baseline",
    "improved",
    "embedding",
    "hybrid-local",
    "hybrid-openai",
    "llm-local",
    "llm-openai",
)

_FIXTURE_LINE_RE = re.compile(
    r"^(?P<term>.+?)\s+\((?P<go_id>GO:\d+)\)\s+\|\s+adj_p=(?P<adj_p>[\d.eE+-]+)\s+\|\s+genes:\s*(?P<genes>.+)$"
)
_GO_ID_RE = re.compile(r"GO:\d+")


class GoSemanticStrategy(str, Enum):
    BASELINE = "baseline"
    IMPROVED = "improved"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"  # alias: hybrid + GO_LLM_BACKEND
    HYBRID_LOCAL = "hybrid-local"
    HYBRID_OPENAI = "hybrid-openai"
    LLM = "llm"  # alias for hybrid-local (production default)
    LLM_LOCAL = "llm-local"
    LLM_OPENAI = "llm-openai"


def parse_go_semantic_strategy(value: str | None) -> GoSemanticStrategy:
    raw = (value or os.getenv("GO_SEMANTIC_STRATEGY", "llm")).strip().lower()
    if raw == "llm":
        return GoSemanticStrategy.HYBRID_LOCAL
    if raw == "hybrid":
        backend = os.getenv("GO_LLM_BACKEND", "local").strip().lower()
        if backend == "openai":
            return GoSemanticStrategy.HYBRID_OPENAI
        return GoSemanticStrategy.HYBRID_LOCAL
    try:
        return GoSemanticStrategy(raw)
    except ValueError:
        logger.warning(f"Unknown GO_SEMANTIC_STRATEGY={raw!r}; defaulting to hybrid-local")
        return GoSemanticStrategy.HYBRID_LOCAL


def llm_backend_for_strategy(strategy: GoSemanticStrategy) -> str:
    if strategy in (GoSemanticStrategy.LLM_OPENAI, GoSemanticStrategy.HYBRID_OPENAI):
        return "openai"
    return "local"


def is_hybrid_strategy(strategy: GoSemanticStrategy) -> bool:
    return strategy in (
        GoSemanticStrategy.HYBRID,
        GoSemanticStrategy.HYBRID_LOCAL,
        GoSemanticStrategy.HYBRID_OPENAI,
    )


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

    data = _score_enrichment_table(
        phenotype,
        enrich_tbl,
        causal_gene=causal_gene,
        embedding_model=embedding_model,
        hybrid_alpha=hybrid_alpha,
        strategy=strategy,
    )
    if len(data) == 0:
        return []

    score_column = "score" if strategy == GoSemanticStrategy.IMPROVED else "similarity"
    if score_column == "similarity":
        data["score"] = data["similarity"]
    ranked = data.sort_values(score_column, ascending=False).head(k)
    return _rows_to_results(ranked, score_column)


def _score_enrichment_table(
    phenotype: str,
    enrich_tbl: pd.DataFrame,
    causal_gene: Optional[str] = None,
    embedding_model: Optional[str] = None,
    hybrid_alpha: float = DEFAULT_HYBRID_ALPHA,
    strategy: GoSemanticStrategy = GoSemanticStrategy.IMPROVED,
) -> pd.DataFrame:
    """Attach similarity and hybrid score columns to an enrichment table."""
    model = embedding_model or os.getenv("GO_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    data = enrich_tbl.copy()
    texts = [build_document_text(row, strategy) for _, row in data.iterrows()]
    if not texts:
        return data.iloc[0:0]

    query = build_go_search_query(phenotype, causal_gene=causal_gene, strategy=strategy)
    doc_embeddings = _embed_texts(texts, model)
    query_embedding = _embed_texts([query], model)[0]

    data["similarity"] = [
        1.0 - scipy.spatial.distance.cosine(emb, query_embedding)
        for emb in doc_embeddings
    ]
    data["score"] = hybrid_score(
        data["similarity"], data["Adjusted P-value"], alpha=hybrid_alpha
    )
    return data


def prefilter_go_candidates_union(
    phenotype: str,
    enrich_tbl: pd.DataFrame,
    top_n: int,
    causal_gene: Optional[str] = None,
    embedding_model: Optional[str] = None,
    hybrid_alpha: float = DEFAULT_HYBRID_ALPHA,
    pvalue_k: Optional[int] = None,
) -> pd.DataFrame:
    """Union shortlist: guaranteed top enrichment terms plus semantic top-ups.

    Two arms are merged so the LLM sees both worlds:
      - `pvalue_k` terms with the strongest raw enrichment signal (adj p-value),
        guaranteed regardless of how well they embed against the phenotype text.
      - The remaining slots (up to `top_n`) filled by highest hybrid score
        (embedding similarity blended with p-value) among terms not already picked.
    """
    if enrich_tbl is None or len(enrich_tbl) == 0:
        return enrich_tbl

    top_n = min(top_n, len(enrich_tbl))
    if pvalue_k is None:
        default_pvalue_k = max(top_n // 2, top_n - 100)
        pvalue_k = int(
            os.getenv("GO_LLM_PREFILTER_PVALUE_K", str(default_pvalue_k))
        )
    pvalue_k = min(max(pvalue_k, 0), top_n)

    data = _score_enrichment_table(
        phenotype,
        enrich_tbl,
        causal_gene=causal_gene,
        embedding_model=embedding_model,
        hybrid_alpha=hybrid_alpha,
    )
    if len(data) == 0:
        return data

    by_pvalue = data.sort_values("Adjusted P-value", ascending=True).head(pvalue_k)
    selected_ids = {str(row["ID"]).strip() for _, row in by_pvalue.iterrows()}
    remaining = top_n - len(by_pvalue)

    extras = []
    if remaining > 0:
        for _, row in data.sort_values("score", ascending=False).iterrows():
            go_id = str(row["ID"]).strip()
            if go_id in selected_ids:
                continue
            extras.append(row)
            selected_ids.add(go_id)
            if len(extras) >= remaining:
                break

    if extras:
        shortlisted = pd.concat([by_pvalue, pd.DataFrame(extras)], ignore_index=True)
    else:
        shortlisted = by_pvalue.reset_index(drop=True)

    logger.info(
        f"Hybrid prefilter union: {len(by_pvalue)} by adj p-value + "
        f"{len(shortlisted) - len(by_pvalue)} by semantic score = "
        f"{len(shortlisted)} total (cap {top_n})"
    )
    return shortlisted


def rank_go_terms_hybrid_llm(
    phenotype: str,
    enrich_tbl: pd.DataFrame,
    k: int = 10,
    causal_gene: Optional[str] = None,
    prefilter_k: Optional[int] = None,
    backend: str = "local",
    model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    hybrid_alpha: float = DEFAULT_HYBRID_ALPHA,
) -> list[dict]:
    """Embedding+p-value union prefilter over the full pool, then one LLM rerank.

    This is the production strategy behind GO_SEMANTIC_STRATEGY=llm/hybrid.
    """
    if enrich_tbl is None or len(enrich_tbl) == 0:
        logger.warning("Empty enrichment table provided for hybrid LLM GO ranking")
        return []

    from src.services.llm import LLM

    if prefilter_k is None:
        prefilter_k = int(
            os.getenv("GO_LLM_PREFILTER_K", str(DEFAULT_LLM_PREFILTER_K))
        )
    prefilter_k = max(prefilter_k, k)
    pvalue_k = int(
        os.getenv("GO_LLM_PREFILTER_PVALUE_K", str(DEFAULT_LLM_PREFILTER_PVALUE_K))
    )

    pool_size = len(enrich_tbl)
    shortlisted = prefilter_go_candidates_union(
        phenotype,
        enrich_tbl,
        top_n=min(prefilter_k, pool_size),
        causal_gene=causal_gene,
        embedding_model=embedding_model,
        hybrid_alpha=hybrid_alpha,
        pvalue_k=pvalue_k,
    )
    logger.info(
        f"Hybrid LLM GO ranking: union prefilter kept {len(shortlisted)} of "
        f"{pool_size} terms for LLM rerank"
    )

    if model is None:
        if backend == "openai":
            model = os.getenv("GO_OPENAI_LLM_MODEL", DEFAULT_OPENAI_LLM_MODEL)
        else:
            model = os.getenv("GO_LLM_MODEL", DEFAULT_LLM_MODEL)

    llm = LLM()
    results = llm.rank_relevant_go_by_llm(
        phenotype=phenotype,
        enrich_tbl=shortlisted,
        k=k,
        causal_gene=causal_gene,
        max_candidates=len(shortlisted),
        model=model,
        backend=backend,
        prefiltered=True,
    )
    for entry in results:
        entry["prefilter"] = "pvalue_embedding_union"
        entry["prefilter_pool"] = pool_size
    return results


def rank_go_terms_llm(
    phenotype: str,
    enrich_tbl: pd.DataFrame,
    k: int = 10,
    causal_gene: Optional[str] = None,
    max_candidates: Optional[int] = None,
    backend: str = "local",
    model: Optional[str] = None,
) -> list[dict]:
    """Rank GO terms via an LLM reading the enrichment candidate pool."""
    if enrich_tbl is None or len(enrich_tbl) == 0:
        logger.warning("Empty enrichment table provided for LLM GO ranking")
        return []

    from src.services.llm import LLM

    if max_candidates is None:
        cap = int(os.getenv("GO_LLM_MAX_CANDIDATES", str(DEFAULT_LLM_MAX_CANDIDATES)))
    else:
        cap = max_candidates
    if model is None:
        if backend == "openai":
            model = os.getenv("GO_OPENAI_LLM_MODEL", DEFAULT_OPENAI_LLM_MODEL)
        else:
            model = os.getenv("GO_LLM_MODEL", DEFAULT_LLM_MODEL)
    llm = LLM()
    return llm.rank_relevant_go_by_llm(
        phenotype=phenotype,
        enrich_tbl=enrich_tbl,
        k=k,
        causal_gene=causal_gene,
        max_candidates=cap,
        model=model,
        backend=backend,
    )


def rank_go_terms_by_strategy(
    phenotype: str,
    enrich_tbl: pd.DataFrame,
    k: int = 10,
    strategy: GoSemanticStrategy | str | None = None,
    causal_gene: Optional[str] = None,
    embedding_model: Optional[str] = None,
    hybrid_alpha: float = DEFAULT_HYBRID_ALPHA,
    max_candidates: Optional[int] = None,
    llm=None,
) -> list[dict]:
    if isinstance(strategy, str) or strategy is None:
        strategy = parse_go_semantic_strategy(strategy)

    if is_hybrid_strategy(strategy):
        return rank_go_terms_hybrid_llm(
            phenotype,
            enrich_tbl,
            k=k,
            causal_gene=causal_gene,
            prefilter_k=max_candidates,
            backend=llm_backend_for_strategy(strategy),
            embedding_model=embedding_model,
            hybrid_alpha=hybrid_alpha,
        )
    if strategy in (
        GoSemanticStrategy.LLM,
        GoSemanticStrategy.LLM_LOCAL,
        GoSemanticStrategy.LLM_OPENAI,
    ):
        return rank_go_terms_llm(
            phenotype,
            enrich_tbl,
            k=k,
            causal_gene=causal_gene,
            max_candidates=max_candidates,
            backend=llm_backend_for_strategy(strategy),
        )
    if strategy == GoSemanticStrategy.EMBEDDING:
        if llm is None:
            from src.services.llm import LLM

            llm = LLM()
        return llm.get_relevant_go(phenotype, enrich_tbl, k=k)
    return rank_go_terms(
        phenotype,
        enrich_tbl,
        k=k,
        strategy=strategy,
        causal_gene=causal_gene,
        embedding_model=embedding_model,
        hybrid_alpha=hybrid_alpha,
    )


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
