from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.services import go_semantic_search as service
from src.services.go_semantic_search import GoSemanticStrategy


def _enrich_tbl(rows):
    """rows: list of (go_id, term, adj_p, genes)"""
    return pd.DataFrame(
        [
            {"ID": go_id, "Term": term, "Desc": term, "Adjusted P-value": adj_p, "Genes": genes}
            for go_id, term, adj_p, genes in rows
        ]
    )


# --- strategy parsing / dispatch -------------------------------------------------


def test_parse_go_semantic_strategy_llm_alias_maps_to_hybrid_local():
    assert service.parse_go_semantic_strategy("llm") == GoSemanticStrategy.HYBRID_LOCAL


def test_parse_go_semantic_strategy_hybrid_respects_backend_env(monkeypatch):
    monkeypatch.setenv("GO_LLM_BACKEND", "openai")
    assert service.parse_go_semantic_strategy("hybrid") == GoSemanticStrategy.HYBRID_OPENAI
    monkeypatch.setenv("GO_LLM_BACKEND", "local")
    assert service.parse_go_semantic_strategy("hybrid") == GoSemanticStrategy.HYBRID_LOCAL


def test_parse_go_semantic_strategy_unknown_value_falls_back_to_hybrid_local():
    assert service.parse_go_semantic_strategy("not-a-real-strategy") == GoSemanticStrategy.HYBRID_LOCAL


def test_parse_go_semantic_strategy_explicit_enum_names_roundtrip():
    assert service.parse_go_semantic_strategy("baseline") == GoSemanticStrategy.BASELINE
    assert service.parse_go_semantic_strategy("embedding") == GoSemanticStrategy.EMBEDDING


@pytest.mark.parametrize(
    "strategy,expected_backend",
    [
        (GoSemanticStrategy.LLM_OPENAI, "openai"),
        (GoSemanticStrategy.HYBRID_OPENAI, "openai"),
        (GoSemanticStrategy.LLM_LOCAL, "local"),
        (GoSemanticStrategy.HYBRID_LOCAL, "local"),
        (GoSemanticStrategy.BASELINE, "local"),
    ],
)
def test_llm_backend_for_strategy(strategy, expected_backend):
    assert service.llm_backend_for_strategy(strategy) == expected_backend


def test_is_hybrid_strategy():
    assert service.is_hybrid_strategy(GoSemanticStrategy.HYBRID_LOCAL)
    assert service.is_hybrid_strategy(GoSemanticStrategy.HYBRID_OPENAI)
    assert not service.is_hybrid_strategy(GoSemanticStrategy.LLM_LOCAL)
    assert not service.is_hybrid_strategy(GoSemanticStrategy.BASELINE)


def test_rank_go_terms_by_strategy_dispatches_hybrid_to_hybrid_llm(monkeypatch):
    hybrid_mock = MagicMock(return_value=["hybrid-result"])
    monkeypatch.setattr(service, "rank_go_terms_hybrid_llm", hybrid_mock)
    tbl = _enrich_tbl([("GO:1", "term", 0.01, "GENE1")])

    result = service.rank_go_terms_by_strategy(
        "phenotype", tbl, k=5, strategy="llm", causal_gene="GENE1", max_candidates=100
    )

    assert result == ["hybrid-result"]
    hybrid_mock.assert_called_once_with(
        "phenotype", tbl, k=5, causal_gene="GENE1", prefilter_k=100,
        backend="local", embedding_model=None, hybrid_alpha=service.DEFAULT_HYBRID_ALPHA,
    )


def test_rank_go_terms_by_strategy_dispatches_plain_llm_strategy(monkeypatch):
    llm_mock = MagicMock(return_value=["llm-result"])
    monkeypatch.setattr(service, "rank_go_terms_llm", llm_mock)
    tbl = _enrich_tbl([("GO:1", "term", 0.01, "GENE1")])

    result = service.rank_go_terms_by_strategy(
        "phenotype", tbl, k=5, strategy=GoSemanticStrategy.LLM_OPENAI, causal_gene="GENE1"
    )

    assert result == ["llm-result"]
    llm_mock.assert_called_once_with(
        "phenotype", tbl, k=5, causal_gene="GENE1", max_candidates=None, backend="openai"
    )


def test_rank_go_terms_by_strategy_embedding_uses_provided_llm(monkeypatch):
    llm = MagicMock()
    llm.get_relevant_go.return_value = ["embedding-result"]
    tbl = _enrich_tbl([("GO:1", "term", 0.01, "GENE1")])

    result = service.rank_go_terms_by_strategy(
        "phenotype", tbl, k=3, strategy=GoSemanticStrategy.EMBEDDING, llm=llm
    )

    assert result == ["embedding-result"]
    llm.get_relevant_go.assert_called_once_with("phenotype", tbl, k=3)


def test_rank_go_terms_by_strategy_falls_back_to_plain_rank_go_terms(monkeypatch):
    rank_mock = MagicMock(return_value=["baseline-result"])
    monkeypatch.setattr(service, "rank_go_terms", rank_mock)
    tbl = _enrich_tbl([("GO:1", "term", 0.01, "GENE1")])

    result = service.rank_go_terms_by_strategy(
        "phenotype", tbl, k=5, strategy=GoSemanticStrategy.BASELINE
    )

    assert result == ["baseline-result"]
    rank_mock.assert_called_once()


# --- pure scoring math ------------------------------------------------------------


def test_normalize_neg_log_p_maps_smallest_p_to_one():
    p = pd.Series([1e-10, 1e-2, 0.5])
    normalized = service.normalize_neg_log_p(p)
    assert normalized.iloc[0] == pytest.approx(1.0)
    assert normalized.iloc[-1] == pytest.approx(0.0)
    assert (normalized >= 0).all() and (normalized <= 1).all()


def test_normalize_neg_log_p_constant_column_returns_all_ones():
    p = pd.Series([0.01, 0.01, 0.01])
    normalized = service.normalize_neg_log_p(p)
    assert (normalized == 1.0).all()


def test_hybrid_score_blends_similarity_and_pvalue_by_alpha():
    similarity = pd.Series([1.0, 0.0])
    pvalues = pd.Series([0.5, 0.5])  # normalize_neg_log_p -> [1.0, 0.0] or [0.0, 1.0]? constant->1.0
    score = service.hybrid_score(similarity, pvalues, alpha=1.0)
    # alpha=1.0 means pure similarity
    assert list(score) == [1.0, 0.0]

    score_alpha_0 = service.hybrid_score(similarity, pvalues, alpha=0.0)
    # alpha=0.0 means pure p-value signal; constant p-values normalize to 1.0 for both
    assert list(score_alpha_0) == [1.0, 1.0]


# --- prefilter union: the core "surface the most significant terms" fix ----------


def _patch_score_enrichment_table(monkeypatch, similarity_by_id):
    """Avoid real embedding calls: stub _score_enrichment_table to attach a
    caller-controlled similarity/score column instead of calling OpenAI."""

    def fake_score(phenotype, enrich_tbl, causal_gene=None, embedding_model=None,
                    hybrid_alpha=service.DEFAULT_HYBRID_ALPHA, strategy=None):
        data = enrich_tbl.copy()
        data["similarity"] = [similarity_by_id.get(row["ID"], 0.0) for _, row in data.iterrows()]
        data["score"] = service.hybrid_score(data["similarity"], data["Adjusted P-value"], alpha=hybrid_alpha)
        return data

    monkeypatch.setattr(service, "_score_enrichment_table", fake_score)


def test_prefilter_union_guarantees_top_pvalue_terms_even_with_low_similarity(monkeypatch):
    # GO:1 has the strongest p-value but the worst semantic similarity to the
    # phenotype text -- exactly the case the old embedding-only ranking would
    # drop. The union prefilter must still keep it.
    _patch_score_enrichment_table(monkeypatch, {"GO:1": 0.0, "GO:2": 0.9, "GO:3": 0.8})
    tbl = _enrich_tbl(
        [
            ("GO:1", "strong signal, semantically distant", 1e-8, "GENE1"),
            ("GO:2", "weak signal, semantically close", 0.2, "GENE2"),
            ("GO:3", "mid signal, semantically close", 0.1, "GENE3"),
        ]
    )

    shortlist = service.prefilter_go_candidates_union(
        "phenotype", tbl, top_n=2, pvalue_k=1
    )

    assert "GO:1" in set(shortlist["ID"])
    assert len(shortlist) == 2


def test_prefilter_union_deduplicates_between_pvalue_and_semantic_arms(monkeypatch):
    # GO:1 is both the strongest p-value AND the most similar term -- it must
    # not be double-counted, leaving room for a genuine second pick.
    _patch_score_enrichment_table(monkeypatch, {"GO:1": 0.9, "GO:2": 0.1, "GO:3": 0.5})
    tbl = _enrich_tbl(
        [
            ("GO:1", "best of both", 1e-8, "GENE1"),
            ("GO:2", "weak everything", 0.5, "GENE2"),
            ("GO:3", "second best semantic", 0.2, "GENE3"),
        ]
    )

    shortlist = service.prefilter_go_candidates_union(
        "phenotype", tbl, top_n=2, pvalue_k=1
    )

    assert list(shortlist["ID"]) == ["GO:1", "GO:3"]


def test_prefilter_union_caps_at_available_rows():
    result = service.prefilter_go_candidates_union(
        "phenotype", _enrich_tbl([]), top_n=10
    )
    assert result.empty


def test_prefilter_union_returns_input_unchanged_when_empty():
    empty = _enrich_tbl([])
    assert service.prefilter_go_candidates_union("phenotype", empty, top_n=5) is empty


# --- fixture / replay parsing utilities -------------------------------------------


def test_parse_expected_go_ids_dedupes_and_preserves_order(tmp_path):
    path = tmp_path / "gold.md"
    path.write_text("GO:0006954 inflammatory response\nGO:0002250 immune\nGO:0006954 repeat")

    ids = service.parse_expected_go_ids(str(path))

    assert ids == ["GO:0006954", "GO:0002250"]


def test_overlap_at_k_counts_intersection_within_top_k():
    result_ids = ["GO:1", "GO:2", "GO:3", "GO:4"]
    expected_ids = ["GO:2", "GO:4", "GO:9"]
    assert service.overlap_at_k(result_ids, expected_ids, k=2) == 1
    assert service.overlap_at_k(result_ids, expected_ids, k=4) == 2


def test_load_enrich_table_from_csv_requires_expected_columns(tmp_path):
    path = tmp_path / "table.csv"
    pd.DataFrame({"ID": ["GO:1"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        service.load_enrich_table(str(path))
