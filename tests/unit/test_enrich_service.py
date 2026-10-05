import json
from copy import deepcopy
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.services import enrich as service


def _enrich():
    instance = object.__new__(service.Enrich)
    instance.ensembl_hgnc_map = {"ENSG1": "GENE1", "ENSG2": "GENE2"}
    instance.hgnc_ensembl_map = {"GENE1": "ENSG1"}
    instance.go_map = {"GO:1": {"desc": "description"}}
    instance.config = MagicMock(data_dir="/unused")
    return instance


def test_gene_identifier_normalization_and_mapping():
    enrich = _enrich()
    assert enrich.is_ensembl_id(" ENSG123 ") is True
    assert enrich.to_symbol("'ENSG1'") == "GENE1"
    assert enrich.to_symbol("gene1") == "GENE1"
    assert enrich.to_ensembl_id('"GENE1"') == "ensg1"
    assert enrich.to_ensembl_id("unknown") is None


def test_annotate_graph_gene_names_returns_copy(sample_graph):
    enrich = _enrich()
    original = deepcopy(sample_graph)
    annotated = enrich.annotate_graph_gene_names(sample_graph)
    assert annotated["nodes"][0]["name"] == "ENSG00000140968"
    assert sample_graph == original


def test_run_uses_tissue_gene_list_and_caps_background(monkeypatch):
    enrich = _enrich()
    enrich.get_coexpression_net = MagicMock(
        return_value=(["ENSG1", "ENSG2"], ["ENSG1"] * 6000)
    )
    enrich._process_enrichment_results = MagicMock(return_value="processed")
    api = MagicMock()
    api.return_value.results = pd.DataFrame()
    monkeypatch.setattr(service.gp, "enrichr", api)

    result = enrich.run("GENE1", tissue_name="Liver", coexpression_data="matrix")

    assert result == "processed"
    assert len(api.call_args.kwargs["background"]) == 5000
    assert api.call_args.kwargs["gene_list"] == ["GENE1", "GENE2"]
    assert api.call_args.kwargs["outdir"] is None


def test_run_returns_empty_frame_without_calling_api():
    enrich = _enrich()
    enrich.get_coexpression_net = MagicMock(return_value=[])
    enrich._load_fallback_background_data = MagicMock(return_value=["BG"])
    original = service.gp.enrichr
    service.gp.enrichr = MagicMock()
    try:
        result = enrich.run("GENE1")
        assert list(result.columns) == ["ID", "Term", "Desc", "Adjusted P-value", "Genes", "Term Size"]
        service.gp.enrichr.assert_not_called()
    finally:
        service.gp.enrichr = original


def test_run_with_tables_returns_filtered_and_full_tables(monkeypatch):
    enrich = _enrich()
    enrich.get_coexpression_net = MagicMock(return_value=["GENE1"])
    enrich._load_fallback_background_data = MagicMock(return_value=["BG"])
    raw = pd.DataFrame(
        {
            "Gene_set": ["s", "s"],
            "Term": ["significant (GO:1)", "not significant (GO:2)"],
            "Adjusted P-value": [0.01, 0.5],
            "Genes": ["GENE1;GENE2", "GENE1"],
        }
    )
    monkeypatch.setattr(service.gp, "enrichr", MagicMock(return_value=MagicMock(results=raw)))

    filtered, all_terms = enrich.run_with_tables("GENE1")

    assert list(filtered["ID"]) == ["GO:1"]
    assert list(all_terms["ID"]) == ["GO:1", "GO:2"]
    # go_map only knows GO:1; GO:2 must fall back to "NA", not raise.
    assert dict(zip(all_terms["ID"], all_terms["Desc"])) == {
        "GO:1": "description",
        "GO:2": "NA",
    }


def test_run_with_tables_returns_empty_frames_when_no_coexpressed_genes():
    enrich = _enrich()
    enrich.get_coexpression_net = MagicMock(return_value=[])
    enrich._load_fallback_background_data = MagicMock(return_value=["BG"])
    original = service.gp.enrichr
    service.gp.enrichr = MagicMock()
    try:
        filtered, all_terms = enrich.run_with_tables("GENE1")
        assert list(filtered.columns) == ["ID", "Term", "Desc", "Adjusted P-value", "Genes", "Term Size"]
        assert filtered.empty and all_terms.empty
        service.gp.enrichr.assert_not_called()
    finally:
        service.gp.enrichr = original


def test_retry_api_contract_is_available():
    assert hasattr(service, "EnrichrAPIUnavailableError")
    assert hasattr(service.Enrich, "_run_enrichr_with_retry")


def test_enrichr_retry_shrinks_background_until_success(monkeypatch):
    enrich = _enrich()
    api = MagicMock()
    success = MagicMock(results="results")
    api.side_effect = [RuntimeError("first"), RuntimeError("second"), success]
    monkeypatch.setattr(service.gp, "enrichr", api)

    result = enrich._run_enrichr_with_retry(
        gene_list=["GENE1"],
        gene_sets="GO_Biological_Process_2023",
        background=[f"GENE-{index}" for index in range(6000)],
        organism="human",
        outdir=None,
    )

    assert result == "results"
    assert [len(entry.kwargs["background"]) for entry in api.call_args_list] == [
        5000, 2500, 1000
    ]


def test_enrichr_retry_raises_typed_error_after_exhaustion(monkeypatch):
    enrich = _enrich()
    api = MagicMock(side_effect=RuntimeError("offline"))
    monkeypatch.setattr(service.gp, "enrichr", api)

    with pytest.raises(
        service.EnrichrAPIUnavailableError,
        match=r"background sizes \[5000, 2500, 1000\]",
    ):
        enrich._run_enrichr_with_retry(
            gene_list=["GENE1"],
            gene_sets="GO_Biological_Process_2023",
            background=[f"GENE-{index}" for index in range(6000)],
            organism="human",
        )


@pytest.fixture(autouse=True)
def _reset_go_term_size_cache():
    """Keep the module-level size memo from leaking between tests."""
    service._GO_TERM_SIZES = None
    yield
    service._GO_TERM_SIZES = None


def test_go_term_sizes_parses_library_and_caches_to_disk(monkeypatch, tmp_path):
    cache = tmp_path / "sizes.json"
    monkeypatch.setenv("GO_TERM_SIZES_CACHE", str(cache))
    library = MagicMock(return_value={
        "Regulation Of DNA-templated Transcription (GO:0006355)": ["A", "B", "C"],
        "Regulation Of miRNA-mediated Gene Silencing (GO:0060964)": ["A"],
        "Term without an id": ["A", "B"],
    })
    monkeypatch.setattr(service.gp, "get_library", library)

    sizes = service.go_term_sizes()

    assert sizes == {"GO:0006355": 3, "GO:0060964": 1}
    assert json.loads(cache.read_text()) == {"GO:0006355": 3, "GO:0060964": 1}


def test_go_term_sizes_reads_cache_without_hitting_the_network(monkeypatch, tmp_path):
    cache = tmp_path / "sizes.json"
    cache.write_text(json.dumps({"GO:0006355": 1922}))
    monkeypatch.setenv("GO_TERM_SIZES_CACHE", str(cache))
    library = MagicMock(side_effect=AssertionError("network should not be used"))
    monkeypatch.setattr(service.gp, "get_library", library)

    assert service.go_term_sizes() == {"GO:0006355": 1922}
    library.assert_not_called()


def test_go_term_sizes_degrades_to_empty_when_library_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("GO_TERM_SIZES_CACHE", str(tmp_path / "missing.json"))
    monkeypatch.setattr(
        service.gp, "get_library", MagicMock(side_effect=RuntimeError("offline"))
    )

    assert service.go_term_sizes() == {}


def test_process_results_falls_back_to_library_sizes_without_overlap(monkeypatch, tmp_path):
    """The background-corrected Enrichr endpoint omits Overlap entirely."""
    monkeypatch.setenv("GO_TERM_SIZES_CACHE", str(tmp_path / "sizes.json"))
    monkeypatch.setattr(service.gp, "get_library", MagicMock(return_value={
        "Regulation Of DNA-templated Transcription (GO:0006355)": ["G"] * 1922,
        "Regulation Of miRNA-mediated Gene Silencing (GO:0060964)": ["G"] * 15,
    }))
    enrich = _enrich()
    enrich.go_map = {
        "GO:0006355": {"desc": "broad"}, "GO:0060964": {"desc": "narrow"}
    }
    raw = pd.DataFrame([
        {
            "Gene_set": "GO_Biological_Process_2023",
            "Term": "Regulation Of DNA-templated Transcription (GO:0006355)",
            "Adjusted P-value": 1e-9,
            "Genes": "A;B",
        },
        {
            "Gene_set": "GO_Biological_Process_2023",
            "Term": "Regulation Of miRNA-mediated Gene Silencing (GO:0060964)",
            "Adjusted P-value": 1e-4,
            "Genes": "C",
        },
    ])

    out = enrich._process_enrichment_results(raw, p_threshold=None)

    assert out["Term Size"].tolist() == [1922, 15]


def test_process_results_filters_at_the_shared_significance_floor(monkeypatch):
    monkeypatch.setenv("GO_SIGNIFICANCE_MAX_P", "0.01")
    enrich = _enrich()
    enrich.go_map = {"GO:1": {"desc": "a"}, "GO:2": {"desc": "b"}}
    raw = pd.DataFrame([
        {"Gene_set": "GO", "Term": "kept (GO:1)", "Adjusted P-value": 0.005,
         "Genes": "A", "Overlap": "1/10"},
        {"Gene_set": "GO", "Term": "dropped (GO:2)", "Adjusted P-value": 0.03,
         "Genes": "B", "Overlap": "1/20"},
    ])

    out = enrich._process_enrichment_results(raw)

    assert list(out["ID"]) == ["GO:1"]
