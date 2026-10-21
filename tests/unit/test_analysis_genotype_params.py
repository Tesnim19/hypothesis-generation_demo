"""Genotype-mode kwargs on the shared analysis tasks (COJO, LD, filter, batching).

Variant IDs follow the real DGRP format: <arm>_<dm3 pos>_<TYPE>[;rsID]. The
position in the ID is an old-build coordinate and differs from the dm6
position column, so nothing may parse it.
"""
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.tasks import analysis as analysis_module

# (variant_id, chromosome code, arm label, dm6 position, p, freq, N)
DGRP_ROWS = [
    ("2L_1632386_SNP;rs202927577", 1, "2L", 1632386, 1.069e-07, 0.30, 197),
    ("2L_1640000_DEL", 1, "2L", 1640000, 2e-06, 0.03, 197),        # MAF < 0.05
    ("2L_1650000_SNP", 1, "2L", 1650000, 3e-06, 0.40, 90),          # N < 100
    ("2L_1900000_SNP", 1, "2L", 1900000, 0.5, 0.40, 197),           # outside window
    ("3R_19071977_SNP", 4, "3R", 23246255, 1.081e-06, 0.20, 192),
    ("3R_19071990_INS", 4, "3R", 23246300, 0.2, 0.25, 190),
    ("4_60836_SNP", 5, "4", 40210, 1e-6, 0.10, 197),
    ("X_19511_DEL", 23, "X", 9388593, 4e-06, 0.15, 188),
]


def dgrp_sumstats():
    df = pd.DataFrame(DGRP_ROWS, columns=[
        "variant_id", "chromosome", "chromosome_label", "base_pair_location",
        "p_value", "effect_allele_frequency", "N"])
    df["effect_allele"] = "A"
    df["other_allele"] = "G"
    df["beta"] = 0.5
    df["standard_error"] = 0.1
    df["source"] = "plink2_glm"
    return df


def write_contract(path):
    dgrp_sumstats().to_csv(path, sep="\t", index=False, compression="gzip")
    return str(path)


def test_filter_min_maf_min_n_on_dgrp_ids(tmp_path):
    path = write_contract(tmp_path / "gwas_sumstats.tsv.gz")
    sig, _ = analysis_module.filter_significant_variants.fn(
        path, str(tmp_path), p_threshold=1e-5, min_maf=0.05, min_n=100)
    assert list(sig.index) == [
        "2L_1632386_SNP;rs202927577", "3R_19071977_SNP", "4_60836_SNP", "X_19511_DEL"]
    assert sig["chromosome"].dtype.kind == "i"
    assert "chromosome_label" in sig.columns
    assert "source" not in sig.columns

    # Without the new kwargs the old behaviour holds (p only).
    sig_all, _ = analysis_module.filter_significant_variants.fn(path, str(tmp_path), p_threshold=1e-5)
    assert len(sig_all) == 6


def test_genotype_cojo_command(tmp_path, monkeypatch):
    ld_ref = tmp_path / "plink_binary" / "ld_ref"
    ld_ref.parent.mkdir()
    Path(f"{ld_ref}.bed").write_text("")
    cojo_tmp = tmp_path / "cojo_tmp"
    cojo_tmp.mkdir()
    monkeypatch.setattr(analysis_module, "get_shared_temp_dir", lambda **_kw: str(cojo_tmp))

    calls, inputs = [], []

    def fake_run(cmd, *a, **k):
        calls.append(list(cmd))
        inputs.append(Path(cmd[cmd.index("--cojo-file") + 1]).read_text())
        chrom = cmd[cmd.index("--chr") + 1]
        out = cmd[cmd.index("--out") + 1]
        if chrom == "4":
            Path(f"{out}.jma.cojo").write_text(
                "Chr SNP bp refA freq b se p n freq_geno bJ bJ_se pJ LD_r\n"
                "4 3R_19071977_SNP 23246255 A 0.2 0.5 0.1 1e-6 192 0.2 0.5 0.1 1e-6 0\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(analysis_module.subprocess, "run", fake_run)
    sig = dgrp_sumstats().set_index("variant_id").iloc[[0, 4, 6, 7]]
    cojo_df, _ = analysis_module.run_cojo_per_chromosome.fn(
        sig, "/unused", str(tmp_path / "out"), maf_threshold=0.05, population="dgrp2",
        chromosomes=[1, 2, 3, 4, 5, 23], ld_bfile=str(ld_ref), cojo_p=1e-5)

    calls.sort(key=lambda c: int(c[c.index("--chr") + 1]))
    assert [c[c.index("--chr") + 1] for c in calls] == ["1", "2", "3", "4", "5", "23"]
    for c in calls:
        assert c[c.index("--bfile") + 1] == str(ld_ref)
        assert c[c.index("--maf") + 1] == "0.05"
        assert c[-2:] == ["--cojo-p", "1e-05"]
    # IDs reach GCTA verbatim, including ';'
    assert "2L_1632386_SNP;rs202927577 A G" in inputs[0]
    assert list(cojo_df.index) == ["3R_19071977_SNP"]
    assert int(cojo_df["CHR"].iloc[0]) == 4


def test_genome_wide_cojo_is_one_call_without_chr(tmp_path, monkeypatch):
    ld_ref = tmp_path / "plink_binary" / "ld_ref"
    ld_ref.parent.mkdir()
    Path(f"{ld_ref}.bed").write_text("")
    cojo_tmp = tmp_path / "cojo_tmp"
    cojo_tmp.mkdir()
    monkeypatch.setattr(analysis_module, "get_shared_temp_dir", lambda **_kw: str(cojo_tmp))

    calls = []

    def fake_run(cmd, *a, **k):
        calls.append(list(cmd))
        out = cmd[cmd.index("--out") + 1]
        Path(f"{out}.jma.cojo").write_text(
            "Chr SNP bp refA freq b se p n freq_geno bJ bJ_se pJ LD_r\n"
            "1 2L_1632386_SNP;rs202927577 1632386 A 0.3 0.5 0.1 1e-7 197 0.3 0.5 0.1 1e-7 0\n"
            "23 X_19511_DEL 9388593 A 0.15 0.5 0.1 4e-6 188 0.15 0.5 0.1 4e-6 0\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(analysis_module.subprocess, "run", fake_run)
    sig = dgrp_sumstats().set_index("variant_id").iloc[[0, 4, 6, 7]]
    cojo_df, _ = analysis_module.run_cojo_per_chromosome.fn(
        sig, "/unused", str(tmp_path / "out"), maf_threshold=0.05, population="dgrp2",
        chromosomes=[1, 2, 3, 4, 5, 23], ld_bfile=str(ld_ref), cojo_p=1e-5, genome_wide=True)

    assert len(calls) == 1
    assert "--chr" not in calls[0]
    assert calls[0][-2:] == ["--cojo-p", "1e-05"]
    # Each signal keeps its own chromosome code from GCTA's Chr column
    assert cojo_df["CHR"].to_dict() == {"2L_1632386_SNP;rs202927577": 1, "X_19511_DEL": 23}


def test_genome_wide_cojo_requires_ld_bfile():
    with pytest.raises(ValueError, match="needs ld_bfile"):
        analysis_module.run_cojo_per_chromosome.fn(
            dgrp_sumstats().set_index("variant_id"), "/unused", "/unused", genome_wide=True)


def test_ld_uses_ld_bfile(tmp_path, monkeypatch):
    ld_ref = tmp_path / "ld_ref"
    sumstats = dgrp_sumstats().set_index("variant_id")
    ids = ["3R_19071977_SNP", "3R_19071990_INS"]
    Path(f"{ld_ref}.bim").write_text(
        "4\t3R_19071977_SNP\t0\t23246255\tA\tG\n4\t3R_19071990_INS\t0\t23246300\tA\tG\n")
    cmds, extracted = [], []

    def fake_run_command(cmd):
        cmds.append(cmd)
        toks = cmd.split()
        extracted.append(Path(toks[toks.index("--extract") + 1]).read_text().split("\n"))
        out = toks[toks.index("--out") + 1]
        Path(f"{out}.ld").write_text("1 0.2\n0.2 1\n")
        Path(f"{out}.snplist").write_text("\n".join(ids) + "\n")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(analysis_module, "run_command", fake_run_command)
    ld_df, sub = analysis_module.calculate_ld_matrix(
        chr_num=4, filtered_ids=ids, sumstats=sumstats, plink_dir="/unused",
        population="dgrp2", ref_genome="dm6", maf=0.01, config=analysis_module.Config(),
        ld_bfile=str(ld_ref))
    toks = cmds[0].split()
    assert toks[toks.index("--bfile") + 1] == str(ld_ref)
    assert extracted[0][:2] == ids
    assert list(ld_df.index) == ids and list(sub.index) == ids


# Real DGRP LD has scattered pairwise NaNs (no fully-NaN rows): on 3R:23246255,
# 1,018 of 3,856 variants are in at least one NaN pair and 2,838 remain.
@pytest.mark.parametrize("drop, expected_ids", [
    (False, ["2L_1632386_SNP;rs202927577", "2L_1640000_DEL", "2L_1650000_SNP"]),  # zero-filled (today)
    (True, ["2L_1632386_SNP;rs202927577"]),                                       # both NaN-pair variants dropped
])
def test_ld_nonfinite_handling(tmp_path, monkeypatch, drop, expected_ids):
    ld_ref = tmp_path / "ld_ref"
    ids = ["2L_1632386_SNP;rs202927577", "2L_1640000_DEL", "2L_1650000_SNP"]

    def fake_run_command(cmd):
        toks = cmd.split()
        out = toks[toks.index("--out") + 1]
        Path(f"{out}.ld").write_text("1 0.2 0.3\n0.2 1 nan\n0.3 nan 1\n")
        Path(f"{out}.snplist").write_text("\n".join(ids) + "\n")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(analysis_module, "run_command", fake_run_command)
    kwargs = {"drop_nonfinite_ld": True} if drop else {}
    ld_df, sub = analysis_module.calculate_ld_matrix(
        chr_num=1, filtered_ids=ids, sumstats=dgrp_sumstats().set_index("variant_id"), plink_dir="/unused",
        population="dgrp2", ref_genome="dm6", maf=0.01, config=analysis_module.Config(),
        ld_bfile=str(ld_ref), **kwargs)
    assert list(ld_df.index) == expected_ids and list(sub.index) == expected_ids
    assert ld_df.notna().all().all()


def test_finemap_region_passes_ld_bfile(monkeypatch):
    seen = {}

    def fake_ld(**kw):
        seen.update(kw)
        return None, None

    monkeypatch.setattr(analysis_module, "calculate_ld_matrix", fake_ld)
    analysis_module.finemap_region(
        seed=1, sumstats=dgrp_sumstats().set_index("variant_id"), chr_num=4,
        lead_variant_position=23246255, window=100, ld_bfile="/x/ld_ref")
    assert seen["ld_bfile"] == "/x/ld_ref"
    assert seen["filtered_ids"] == ["3R_19071977_SNP", "3R_19071990_INS"]


@pytest.mark.parametrize("options, expected", [
    ({}, {}),
    ({"estimate_residual_variance": True, "sample_size_stat": "median"},
     {"estimate_residual_variance": True, "sample_size_stat": "median"}),
])
def test_finemap_region_passes_susie_options_only_when_set(monkeypatch, options, expected):
    seen = {}
    monkeypatch.setattr(analysis_module, "calculate_ld_matrix",
                        lambda **kw: (pd.DataFrame([[1.0]]), dgrp_sumstats().iloc[[4]]))

    def fake_susie(**kw):
        seen.update(kw)
        return None

    monkeypatch.setattr(analysis_module, "run_susie_finemapping", fake_susie)
    analysis_module.finemap_region(
        seed=1, sumstats=dgrp_sumstats().set_index("variant_id"), chr_num=4,
        lead_variant_position=23246255, window=100, ld_bfile="/x/ld_ref", **options)
    extras = {k: seen[k] for k in ("estimate_residual_variance", "sample_size_stat") if k in seen}
    assert extras == expected


def test_create_region_batches_from_columns():
    cojo = pd.DataFrame(
        {"CHR": [1, 4, 5, 23], "bp": [1632386, 23246255, 40210, 9388593], "refA": ["A"] * 4},
        index=pd.Index(["2L_1632386_SNP;rs202927577", "3R_19071977_SNP", "4_60836_SNP", "X_19511_DEL"], name="ID"))
    batches = analysis_module.create_region_batches(cojo, batch_size=3, coords_from_columns=True)
    regions = [r for b in batches for r in b]
    assert [(r["variant_id"], r["chr"], r["position"]) for r in regions] == [
        ("2L_1632386_SNP;rs202927577", 1, 1632386),
        ("3R_19071977_SNP", 4, 23246255),    # not 19071977 from the ID
        ("4_60836_SNP", 5, 40210),           # arm 4 is code 5, not 3R's 4
        ("X_19511_DEL", 23, 9388593),
    ]
    assert len(batches) == 2
    # Default path still parses colon IDs only, and drops these.
    assert analysis_module.create_region_batches(cojo, batch_size=3) == []


def test_subset_sumstats_to_windows(tmp_path):
    path = write_contract(tmp_path / "gwas_sumstats.tsv.gz")
    cojo = pd.DataFrame({"CHR": [1, 4], "bp": [1632386, 23246255]},
                        index=["2L_1632386_SNP;rs202927577", "3R_19071977_SNP"])
    subset = analysis_module.subset_sumstats_to_windows(path, cojo, window_kb=100)
    assert list(subset.index) == [
        "2L_1632386_SNP;rs202927577", "2L_1640000_DEL", "2L_1650000_SNP",
        "3R_19071977_SNP", "3R_19071990_INS"]
    sig, _ = analysis_module.filter_significant_variants.fn(path, str(tmp_path), p_threshold=1e-5)
    assert sorted(subset.columns) == sorted(sig.columns)

    # Round trip through the worker save/load keeps IDs and dtypes.
    worker_file = analysis_module.save_sumstats_for_workers.fn(subset, str(tmp_path))
    loaded = analysis_module.load_sumstats_from_file(worker_file)
    assert list(loaded.index) == list(subset.index)
    assert loaded["chromosome"].dtype.kind == "i"
