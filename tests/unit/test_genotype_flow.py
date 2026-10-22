"""analysis_pipeline_flow in input_type="genotype" mode, with every task swapped
for an ImmediateTask stand-in (the genotype stage writes a small contract)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from conftest import ImmediateResult
from src.flows import analysis as flow_module
from src.tasks import analysis as analysis_module
from test_analysis_genotype_params import dgrp_sumstats


def _setup(tmp_path, monkeypatch, T):
    output_dir = tmp_path / "analysis"
    (output_dir / "preprocessed_data").mkdir(parents=True)
    pheno_dir = tmp_path / "uploads" / "user-1"
    pheno_dir.mkdir(parents=True)
    phenotype = pheno_dir / "lifespan.tsv"
    phenotype.write_text("DGRP_21\t50\n")
    panel_dir = tmp_path / "panels" / "dgrp2"
    panel_dir.mkdir(parents=True)
    ld_bfile = str(output_dir / "plink_binary" / "ld_ref")

    def fake_genotype(user_id, project_id, phenotype_path, species, out_dir, **kw):
        path = output_dir / "preprocessed_data" / "gwas_sumstats.tsv.gz"
        dgrp_sumstats().to_csv(path, sep="\t", index=False, compression="gzip")
        return {"sumstats_path": str(path), "ld_bfile": ld_bfile, "n_samples": 197,
                "n_variants": 8, "ld_n_variants": 8,
                "chromosome_codes": [1, 2, 3, 4, 5, 23],
                "chromosome_labels": ["2L", "2R", "3L", "3R", "4", "X"]}

    cojo = pd.DataFrame(
        {"Chr": [1, 4], "bp": [1632386, 23246255], "CHR": [1, 4]},
        index=pd.Index(["2L_1632386_SNP;rs202927577", "3R_19071977_SNP"], name="ID"))

    tasks = {
        "get_project_analysis_path_task": T(lambda *_a, **_k: str(output_dir)),
        "prepare_gwas_file_task": T(lambda *_a, **_k: None),
        "harmonize_sumstats_with_nextflow": T(lambda *_a, **_k: "never"),
        "run_combined_ldsc_tissue_analysis": T(lambda *_a, **_k: {"analysis_run_id": "x"}),
        "run_genotype_gwas_task": T(fake_genotype),
        "filter_significant_variants": T(analysis_module.filter_significant_variants.fn),
        "run_cojo_per_chromosome": T(lambda *_a, **_k: (cojo, str(output_dir / "combined_cojo_results.txt"))),
        "save_analysis_state_task": T(lambda *_a, **_k: None),
        "create_analysis_result_task": T(lambda *_a, **_k: str(output_dir / "results.json")),
        "get_results_from_opentargets_task": T(lambda *_a, **_k: None),
        "cleanup_sumstats_file": T(lambda *_a, **_k: None),
    }
    for name, task in tasks.items():
        monkeypatch.setattr(flow_module, name, task)

    saved = []

    def fake_save(df, temp_dir=None):
        saved.append(df.copy())
        return str(output_dir / "worker_sumstats.tsv")

    monkeypatch.setattr(flow_module, "save_sumstats_for_workers", fake_save)
    mapped = []

    def fake_map(batch_data_list):
        mapped.extend(batch_data_list)
        res = pd.DataFrame({"PIP": [0.95], "credible_set": [1]}, index=["3R_19071977_SNP"])
        return [ImmediateResult([res]) for _ in batch_data_list]

    monkeypatch.setattr(flow_module, "finemap_region_batch_worker", SimpleNamespace(map=fake_map))
    monkeypatch.setattr(flow_module, "_prefect_flow_run", SimpleNamespace(id="run-1"))
    monkeypatch.setattr(flow_module, "send_pipeline_notification", MagicMock())
    ot = MagicMock()
    monkeypatch.setattr(flow_module, "CredibleSetsHandler", ot)
    rmtree, removed = [], []
    monkeypatch.setattr(flow_module._shutil, "rmtree", lambda p, **k: rmtree.append(p))
    monkeypatch.setattr(flow_module.os, "remove", lambda p: removed.append(p))
    return SimpleNamespace(tasks=tasks, saved=saved, mapped=mapped, ot=ot, rmtree=rmtree,
                           removed=removed, phenotype=phenotype, panel_dir=panel_dir,
                           ld_bfile=ld_bfile, output_dir=output_dir)


def test_genotype_flow_wiring(tmp_path, monkeypatch, immediate_task_factory):
    h = _setup(tmp_path, monkeypatch, immediate_task_factory)
    result = flow_module.analysis_pipeline_flow.fn(
        "user-1", "project-1", input_type="genotype", species="drosophila",
        genotype_panel_id="dgrp2", phenotype_file_path=str(h.phenotype),
        opentargets_study_id="GCST1",  # must be ignored in genotype mode
    )
    assert result["results_file"].endswith("results.json")

    # Skipped stages
    for name in ("harmonize_sumstats_with_nextflow", "run_combined_ldsc_tissue_analysis",
                 "get_results_from_opentargets_task", "prepare_gwas_file_task"):
        assert h.tasks[name].calls == [], name
    h.ot.assert_not_called()

    # Cleanup never touches the phenotype or panel
    touched = [str(p) for p in h.rmtree + h.removed]
    assert not any(str(h.phenotype) in p or str(h.phenotype.parent) == p for p in touched)
    assert not any(str(h.panel_dir) in p for p in touched)
    assert h.phenotype.exists()

    # Genotype stage gets the panel, not gwas_file_path
    (args, kwargs), = h.tasks["run_genotype_gwas_task"].calls
    assert args[2] == str(h.phenotype) and args[3] == "drosophila"
    assert kwargs["panel_id"] == "dgrp2" and kwargs["genotype_bfile"] is None

    # Filter / COJO use profile values
    (_, fkw), = h.tasks["filter_significant_variants"].calls
    assert fkw == {"p_threshold": 1e-5, "min_maf": 0.05, "min_n": 100}
    (_, ckw), = h.tasks["run_cojo_per_chromosome"].calls
    assert ckw["chromosomes"] == [1, 2, 3, 4, 5, 23]
    assert ckw["ld_bfile"] == h.ld_bfile
    assert ckw["cojo_p"] == 1e-5 and ckw["maf_threshold"] == 0.05
    assert ckw["population"] == "dgrp2" and ckw["ref_genome"] == "dm6"

    # Workers get the window subset (all SNPs within ±100 kb of a lead), not only significant SNPs
    worker_df, = h.saved
    assert list(worker_df.index) == [
        "2L_1632386_SNP;rs202927577", "2L_1640000_DEL", "2L_1650000_SNP",
        "3R_19071977_SNP", "3R_19071990_INS"]
    regions = [r for b in h.mapped for r in b[0]]
    assert [(r["chr"], r["position"]) for r in regions] == [(1, 1632386), (4, 23246255)]
    fp = h.mapped[0][3]["finemap_params"]
    assert fp["ld_bfile"] == h.ld_bfile
    assert fp["window"] == 100 and fp["L"] == 10 and fp["maf_threshold"] == 0.01
    assert fp["coverage"] == 0.95 and fp["min_abs_corr"] == 0.5

    states = [a[2] for a, _ in h.tasks["save_analysis_state_task"].calls]
    assert states[0]["stage"] == "Genotype_QC_GWAS"
    assert not any(s.get("stage") in ("Harmonization", "LDSC_Tissue_Analysis") for s in states)
    completed = states[-1]
    assert completed["status"] == "Completed"
    assert completed["ldsc_status"] == "skipped"
    assert completed["input_type"] == "genotype" and completed["species"] == "drosophila"
    assert completed["n_samples"] == 197


def test_genotype_flow_param_overrides(tmp_path, monkeypatch, immediate_task_factory):
    h = _setup(tmp_path, monkeypatch, immediate_task_factory)
    flow_module.analysis_pipeline_flow.fn(
        "user-1", "project-1", input_type="genotype", species="drosophila",
        genotype_bfile="/data/my_panel", phenotype_file_path=str(h.phenotype),
        param_overrides={"L": -1, "sumstats_scope": "significant"},
    )
    (_, kwargs), = h.tasks["run_genotype_gwas_task"].calls
    assert kwargs["genotype_bfile"] == "/data/my_panel" and kwargs["panel_id"] is None
    assert h.mapped[0][3]["finemap_params"]["L"] == -1
    assert list(h.saved[0].index) == [
        "2L_1632386_SNP;rs202927577", "3R_19071977_SNP", "4_60836_SNP", "X_19511_DEL"]


@pytest.mark.parametrize("kwargs, match", [
    ({"input_type": "vcf"}, "Unknown input_type"),
    ({"species": "drosophila"}, "only supported for species='human'"),
    ({"input_type": "genotype", "species": "human", "phenotype_file_path": "/p"}, "does not support"),
    ({"input_type": "genotype", "species": "drosophila"}, "requires phenotype_file_path"),
])
def test_flow_rejects_invalid_mode(tmp_path, monkeypatch, immediate_task_factory, kwargs, match):
    h = _setup(tmp_path, monkeypatch, immediate_task_factory)
    with pytest.raises(ValueError, match=match):
        flow_module.analysis_pipeline_flow.fn("user-1", "project-1", **kwargs)
    assert h.tasks["harmonize_sumstats_with_nextflow"].calls == []
    assert h.tasks["run_genotype_gwas_task"].calls == []
