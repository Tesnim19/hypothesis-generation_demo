"""Golden tests for the human (sumstats) analysis path.

These pin the exact GCTA COJO and PLINK LD commands, and the kwargs the
analysis flow hands to each task, as produced by the code before genotype
input mode was added. Any change to the human path shows up as a diff
against the JSON files in tests/unit/golden/.

To regenerate deliberately: UPDATE_GOLDEN=1 pytest tests/unit/test_human_regression_commands.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from conftest import ImmediateResult
from src.tasks import analysis as analysis_module
from src.flows import analysis as flow_module


GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"


def _check_golden(name, actual):
    path = GOLDEN_DIR / name
    text = json.dumps(actual, indent=2, sort_keys=True, default=str) + "\n"
    if UPDATE:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return
    assert path.exists(), f"Golden file {path} missing; run with UPDATE_GOLDEN=1 on unmodified code"
    assert json.loads(text) == json.loads(path.read_text())


def _human_sumstats(n_per_chrom=2):
    rows = []
    for chrom in (1, 7, 22):
        for i in range(n_per_chrom):
            pos = 1_000_000 + i * 5_000
            rows.append({
                "variant_id": f"{chrom}:{pos}:A:G",
                "chromosome": chrom,
                "base_pair_location": pos,
                "p_value": 1e-10 * (i + 1),
                "effect_allele": "G",
                "other_allele": "A",
                "beta": 0.1 + i / 100,
                "standard_error": 0.01,
                "effect_allele_frequency": 0.3,
                "N": 10000,
            })
    return pd.DataFrame(rows).set_index("variant_id")


# --- Task-level: COJO ---------------------------------------------------

def test_human_cojo_commands_golden(tmp_path, monkeypatch):
    plink_dir = tmp_path / "plink"
    (plink_dir / "EUR").mkdir(parents=True)
    for chrom in range(1, 23):
        (plink_dir / "EUR" / f"EUR.chr{chrom}.1KG.GRCh38.bed").write_text("")

    cojo_tmp = tmp_path / "cojo_tmp"
    cojo_tmp.mkdir()
    monkeypatch.setattr(analysis_module, "get_shared_temp_dir", lambda **_kw: str(cojo_tmp))

    calls = []
    cojo_inputs = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        out = cmd[cmd.index("--out") + 1]
        chrom = cmd[cmd.index("--chr") + 1]
        cojo_file = cmd[cmd.index("--cojo-file") + 1]
        cojo_inputs.append(Path(cojo_file).read_text())
        if chrom in ("1", "22"):
            Path(f"{out}.jma.cojo").write_text(
                "Chr SNP bp refA freq b se p n freq_geno bJ bJ_se pJ LD_r\n"
                f"{chrom} {chrom}:1000000:A:G 1000000 G 0.3 0.1 0.01 1e-10 10000 0.3 0.1 0.01 1e-10 0\n"
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(analysis_module.subprocess, "run", fake_run)

    out_dir = tmp_path / "out"
    cojo_df, cojo_path = analysis_module.run_cojo_per_chromosome.fn(
        _human_sumstats(), str(plink_dir), str(out_dir),
        maf_threshold=0.01, population="EUR", ref_genome="GRCh38",
    )

    def norm(s):
        return s.replace(str(cojo_tmp), "<COJO_TMP>").replace(str(tmp_path), "<TMP>")

    cmds = sorted(([norm(a) for a in c] for c in calls), key=lambda c: int(c[c.index("--chr") + 1]))
    assert len(set(cojo_inputs)) == 1
    _check_golden("human_cojo_cmds.json", {
        "commands": cmds,
        "cojo_input": cojo_inputs[0],
        "result_index": sorted(cojo_df.index.tolist()),
        "result_chr": sorted(cojo_df["CHR"].tolist()),
        "output_path": norm(cojo_path),
    })


# --- Task-level: LD -----------------------------------------------------

def test_human_ld_command_golden(tmp_path, monkeypatch):
    plink_dir = tmp_path / "plink"
    (plink_dir / "EUR").mkdir(parents=True)
    sumstats = _human_sumstats()
    ids = [v for v in sumstats.index if v.startswith("7:")]
    bim = plink_dir / "EUR" / "EUR.chr7.1KG.GRCh38.bim"
    bim.write_text("".join(f"7\t{v}\t0\t{v.split(':')[1]}\tG\tA\n" for v in ids))

    cmds = []
    extracted = []

    def fake_run_command(cmd):
        cmds.append(cmd)
        toks = cmd.split()
        extracted.append(Path(toks[toks.index("--extract") + 1]).read_text().split())
        out = toks[toks.index("--out") + 1]
        Path(f"{out}.ld").write_text("1 0.5\n0.5 1\n")
        Path(f"{out}.snplist").write_text("\n".join(ids) + "\n")
        return SimpleNamespace(returncode=0, stderr="")

    bim_paths = []
    real_check = analysis_module.check_ld_dimensions

    def spy_check(ld, snp_df, bim_file_path):
        bim_paths.append(bim_file_path)
        return real_check(ld, snp_df, bim_file_path)

    monkeypatch.setattr(analysis_module, "run_command", fake_run_command)
    monkeypatch.setattr(analysis_module, "check_ld_dimensions", spy_check)

    ld_df, sub = analysis_module.calculate_ld_matrix(
        chr_num=7, filtered_ids=ids, sumstats=sumstats, plink_dir=str(plink_dir),
        population="EUR", ref_genome="GRCh38", maf=0.01, config=analysis_module.Config(),
    )
    assert ld_df is not None and list(ld_df.index) == ids

    toks = cmds[0].split()
    toks[toks.index("--extract") + 1] = "<EXTRACT>"
    toks[toks.index("--out") + 1] = "<OUT>"
    _check_golden("human_ld_cmds.json", {
        "command": [t.replace(str(tmp_path), "<TMP>") for t in toks],
        "extracted_ids": extracted[0],
        "bim_paths": [p.replace(str(tmp_path), "<TMP>") for p in bim_paths],
    })


# --- Flow-level ---------------------------------------------------------

def _summarize(value, tmp_path):
    """JSON-safe, path-normalized view of a task argument."""
    if isinstance(value, pd.DataFrame):
        return {"<DataFrame>": sorted(map(str, value.index)), "columns": sorted(value.columns)}
    if isinstance(value, dict):
        return {k: _summarize(v, tmp_path) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize(v, tmp_path) for v in value]
    if isinstance(value, str):
        return value.replace(str(tmp_path), "<TMP>")
    return value


def _record(task, tmp_path):
    return [
        {"args": _summarize(list(a), tmp_path), "kwargs": _summarize(k, tmp_path)}
        for a, k in task.calls
    ]


def _flow_harness(tmp_path, monkeypatch, T, *, ot_has_results=None, ldsc_error=None):
    """Swap every task the analysis flow uses for an ImmediateTask stand-in."""
    monkeypatch.setenv("PLINK_DIR_38", "/golden/plink_format_b38")
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()
    upload_dir = tmp_path / "uploads" / "user-1"
    upload_dir.mkdir(parents=True)
    raw_gwas = upload_dir / "raw.tsv.gz"
    raw_gwas.write_text("raw")
    harmonized = output_dir / "harmonized.tsv.gz"

    significant = _human_sumstats()
    cojo = pd.DataFrame(
        {"CHR": [1, 22], "bp": [1000000, 1000000]},
        index=pd.Index(["1:1000000:A:G", "22:1000000:A:G"], name="ID"),
    )

    def ldsc(*_a, **_k):
        if ldsc_error is not None:
            raise ldsc_error
        return {"analysis_run_id": "ldsc-1"}

    ot_result = pd.DataFrame({"PIP": [0.8], "credible_set": [1]}, index=["1_1000000_A_G"])
    tasks = {
        "get_project_analysis_path_task": T(lambda *_a, **_k: str(output_dir)),
        "prepare_gwas_file_task": T(lambda *_a, **_k: str(output_dir / "prepared.tsv.gz")),
        "harmonize_sumstats_with_nextflow": T(lambda *_a, **_k: str(harmonized)),
        "run_combined_ldsc_tissue_analysis": T(ldsc),
        "filter_significant_variants": T(lambda *_a, **_k: (significant, str(output_dir / "significant_variants.tsv"))),
        "run_cojo_per_chromosome": T(lambda *_a, **_k: (cojo, str(output_dir / "combined_cojo_results.txt"))),
        "save_analysis_state_task": T(lambda *_a, **_k: None),
        "create_analysis_result_task": T(lambda *_a, **_k: str(output_dir / "results.json")),
        "get_results_from_opentargets_task": T(lambda *_a, **_k: ot_result),
        "cleanup_sumstats_file": T(lambda *_a, **_k: None),
    }
    for name, task in tasks.items():
        monkeypatch.setattr(flow_module, name, task)

    saved_sumstats = []

    def fake_save_sumstats(df, temp_dir=None):
        saved_sumstats.append((df, temp_dir))
        return str(output_dir / "sumstats_worker.tsv")

    monkeypatch.setattr(flow_module, "save_sumstats_for_workers", fake_save_sumstats)

    mapped = []

    def fake_map(batch_data_list):
        mapped.extend(batch_data_list)
        result = pd.DataFrame({"PIP": [0.9], "credible_set": [1]}, index=["1:1000000:A:G"])
        return [ImmediateResult([result]) for _ in batch_data_list]

    monkeypatch.setattr(flow_module, "finemap_region_batch_worker", SimpleNamespace(map=fake_map))
    monkeypatch.setattr(flow_module, "_prefect_flow_run", SimpleNamespace(id="flow-run-1"))
    notify = MagicMock()
    monkeypatch.setattr(flow_module, "send_pipeline_notification", notify)
    rmtree_calls = []
    monkeypatch.setattr(flow_module._shutil, "rmtree", lambda p, **k: rmtree_calls.append(p))
    removed = []
    real_remove = os.remove
    monkeypatch.setattr(flow_module.os, "remove", lambda p: (removed.append(p), real_remove(p) if os.path.exists(p) else None))
    ot_handler = MagicMock()
    ot_handler.study_has_credible_sets.return_value = bool(ot_has_results)
    monkeypatch.setattr(flow_module, "CredibleSetsHandler", MagicMock(return_value=ot_handler))

    def snapshot():
        states = []
        for a, _k in tasks["save_analysis_state_task"].calls:
            states.append({k: v for k, v in a[2].items() if k not in ("started_at",)})
        golden = {
            name: _record(task, tmp_path)
            for name, task in tasks.items()
            if name != "save_analysis_state_task"
        }
        golden["states"] = states
        golden["save_sumstats_for_workers"] = _summarize([list(s) for s in saved_sumstats], tmp_path)
        golden["finemap_batches"] = _summarize(mapped, tmp_path)
        golden["rmtree"] = _summarize(rmtree_calls, tmp_path)
        golden["notifications"] = _summarize([list(c.args) + [c.kwargs] for c in notify.call_args_list], tmp_path)
        return golden

    def snapshot_full():
        golden = snapshot()
        golden["os_remove"] = _summarize(removed, tmp_path)
        golden["ot_checks"] = _summarize([list(c.args) for c in ot_handler.study_has_credible_sets.call_args_list], tmp_path)
        return golden

    return SimpleNamespace(raw_gwas=raw_gwas, snapshot=snapshot, snapshot_full=snapshot_full)


def test_human_flow_kwargs_golden(tmp_path, monkeypatch, immediate_task_factory):
    h = _flow_harness(tmp_path, monkeypatch, immediate_task_factory)
    result = flow_module.analysis_pipeline_flow.fn(
        "user-1", "project-1", gwas_file_path=str(h.raw_gwas),
        user_email="u@example.org", project_name="Proj",
    )
    assert result["results_file"].endswith("results.json")
    _check_golden("human_flow_kwargs.json", h.snapshot())


@pytest.mark.parametrize("ot_has_results", [True, False])
def test_human_flow_opentargets_golden(tmp_path, monkeypatch, immediate_task_factory, ot_has_results):
    h = _flow_harness(tmp_path, monkeypatch, immediate_task_factory, ot_has_results=ot_has_results)
    flow_module.analysis_pipeline_flow.fn(
        "user-1", "project-1", gwas_file_path=str(h.raw_gwas),
        opentargets_study_id="GCST000001", coverage=0.99,
    )
    _check_golden(f"human_flow_ot_{str(ot_has_results).lower()}.json", h.snapshot_full())


def test_human_flow_file_needs_processing_golden(tmp_path, monkeypatch, immediate_task_factory):
    h = _flow_harness(tmp_path, monkeypatch, immediate_task_factory)
    flow_module.analysis_pipeline_flow.fn(
        "user-1", "project-1", gwas_file_path=None, file_needs_processing=True,
        file_metadata_id="meta-1", file_storage_key="key-1", file_id_new="file-1",
        file_source_minio_path="s3://x/y", file_gwas_library_id="lib-1",
        population="EAS", maf_threshold=0.05, window=500, L=5, sample_size=1234,
    )
    _check_golden("human_flow_file_processing.json", h.snapshot_full())


def test_human_flow_ldsc_failure_golden(tmp_path, monkeypatch, immediate_task_factory):
    h = _flow_harness(tmp_path, monkeypatch, immediate_task_factory, ldsc_error=ValueError("ldsc boom"))
    with pytest.raises(RuntimeError, match="LDSC tissue analysis failed"):
        flow_module.analysis_pipeline_flow.fn("user-1", "project-1", gwas_file_path=str(h.raw_gwas))
    golden = h.snapshot_full()
    assert any(s.get("stage") == "LDSC_Analysis" and s["status"] == "Failed" for s in golden["states"])
    _check_golden("human_flow_ldsc_failure.json", golden)


# --- Task-level: filter -------------------------------------------------

def test_human_filter_golden(tmp_path):
    df = _human_sumstats(n_per_chrom=3).reset_index()
    df.loc[df.index[-1], "p_value"] = 1e-3  # one non-significant row
    df["rsid"] = [f"rs{i}" for i in range(len(df))]
    df["extra_col"] = "x"  # not in `needed`, must not be loaded
    path = tmp_path / "harmonized.tsv.gz"
    df.to_csv(path, sep="\t", index=False, compression="gzip")

    sig, sig_path = analysis_module.filter_significant_variants.fn(str(path), str(tmp_path))
    _check_golden("human_filter.json", {
        "index": list(map(str, sig.index)),
        "columns": list(sig.columns),
        "values": json.loads(sig.to_json(orient="split"))["data"],
        "file": Path(sig_path).read_text(),
        "path": sig_path.replace(str(tmp_path), "<TMP>"),
    })


# --- Task-level: finemap_region plumbing --------------------------------

def test_human_finemap_region_kwargs_golden(monkeypatch):
    sumstats = _human_sumstats(n_per_chrom=3)
    seen = {}

    def fake_ld(**kwargs):
        seen["ld"] = {k: v for k, v in kwargs.items() if k not in ("sumstats", "config")}
        ids = kwargs["filtered_ids"]
        return pd.DataFrame(1.0, index=ids, columns=ids), kwargs["sumstats"].loc[ids]

    def fake_susie(**kwargs):
        seen["susie"] = {k: v for k, v in kwargs.items() if k not in ("ld_df", "sub_region_sumstats_ld")}
        seen["susie_ids"] = list(kwargs["sub_region_sumstats_ld"].index)
        return pd.DataFrame({"PIP": [1.0]})

    monkeypatch.setattr(analysis_module, "calculate_ld_matrix", fake_ld)
    monkeypatch.setattr(analysis_module, "run_susie_finemapping", fake_susie)
    analysis_module.finemap_region(
        seed=42, sumstats=sumstats, chr_num=7, lead_variant_position=1005000,
        window=2000, population="EUR", L=-1, coverage=0.95, min_abs_corr=0.5,
        maf=0.01, ref_genome="GRCh38", plink_dir="/golden/plink",
    )
    _check_golden("human_finemap_region.json", seen)
