"""Genotype input stage (src/tasks/genotype.py).

Pure-Python tests always run. The end-to-end tests build a tiny synthetic
DGRP-like panel with `plink2 --dummy` and skip when plink/plink2 (or gcta64)
are not on PATH. The real DGRP panel is never used here.
"""
import gzip
import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.species import load_species_profile
from src.tasks import analysis as analysis_module
from src.tasks import genotype as gt

HAS_PLINK = shutil.which("plink") is not None and shutil.which("plink2") is not None
HAS_GCTA = shutil.which("gcta64") is not None
needs_plink = pytest.mark.skipif(not HAS_PLINK, reason="plink and plink2 not on PATH")
needs_gcta = pytest.mark.skipif(not (HAS_PLINK and HAS_GCTA), reason="plink/plink2/gcta64 not on PATH")

# Source labels as in the reference DGRP merged_qc.bim: X is written as 23.
ARMS = [("2L", "2L"), ("2R", "2R"), ("3L", "3L"), ("3R", "3R"), ("4", "4"), ("X", "23")]
PER_ARM = 100
CAUSAL_ARMS = ("3R", "X")  # arms that get a planted effect


@pytest.fixture
def fly(monkeypatch):
    return load_species_profile("drosophila", overrides={"genotype": {"min_lines": 3}})


def write_fam(prefix, iids, fid="line"):
    Path(f"{prefix}.fam").write_text("".join(f"{fid}\t{i}\t0\t0\t0\t-9\n" for i in iids))


# --- phenotype -----------------------------------------------------------

def test_phenotype_two_column_headerless(tmp_path, fly):
    write_fam(tmp_path / "panel", ["21", "26", "28", "31"])
    pheno = tmp_path / "p.tsv"
    pheno.write_text("DGRP_021\t50.5\nDGRP_26\t47\n28\t40\nDGRP_999\t10\n")
    out = gt.prepare_phenotype_table(str(pheno), str(tmp_path / "panel"), fly, tmp_path / "w")
    assert out["n_lines"] == 3
    table = pd.read_csv(out["pheno_path"], sep="\t", dtype=str)
    assert list(table.columns) == ["FID", "IID", "trait"]
    assert table["IID"].tolist() == ["21", "26", "28"]   # FID/IID taken from the .fam
    assert set(table["FID"]) == {"line"}
    keep = Path(out["keep_path"]).read_text().splitlines()
    assert keep == ["line\t21", "line\t26", "line\t28"]


def test_phenotype_with_header_and_csv(tmp_path, fly):
    write_fam(tmp_path / "panel", ["1", "2", "3"])
    pheno = tmp_path / "p.csv"
    pheno.write_text("line,lifespan\n1,3.5\n2,NA\n3,4\n")
    with pytest.raises(ValueError, match="Only 2"):
        gt.prepare_phenotype_table(str(pheno), str(tmp_path / "panel"), fly, tmp_path / "w")
    out = gt.prepare_phenotype_table(str(pheno), str(tmp_path / "panel"), fly, tmp_path / "w", min_lines=2)
    assert out["n_lines"] == 2  # NA dropped


def test_phenotype_dgrpool_columns_and_sex(tmp_path, fly):
    write_fam(tmp_path / "panel", ["21", "26", "28"])
    pheno = tmp_path / "summary.tsv.gz"
    with gzip.open(pheno, "wt") as fh:
        fh.write("DGRP\tsex\tmn_Lifespan\tsd_Lifespan\n"
                 "DGRP_021\tF\t50\t1\nDGRP_021\tM\t40\t1\n"
                 "DGRP_026\tF\t45\t1\nDGRP_026\tM\t41\t1\n"
                 "DGRP_028\tF\t44\t1\n")
    with pytest.raises(ValueError, match="Duplicate"):
        gt.prepare_phenotype_table(str(pheno), str(tmp_path / "panel"), fly, tmp_path / "w",
                                   id_column="DGRP", phenotype_column="mn_Lifespan")
    out = gt.prepare_phenotype_table(str(pheno), str(tmp_path / "panel"), fly, tmp_path / "w",
                                     id_column="DGRP", phenotype_column="mn_Lifespan", sex="F")
    table = pd.read_csv(out["pheno_path"], sep="\t")
    assert table["trait"].tolist() == [50, 45, 44]
    with pytest.raises(ValueError, match="lacks columns"):
        gt.prepare_phenotype_table(str(pheno), str(tmp_path / "panel"), fly, tmp_path / "w",
                                   id_column="DGRP", phenotype_column="nope")


def test_phenotype_rejects_bad_input(tmp_path, fly):
    write_fam(tmp_path / "panel", ["1", "2", "3"])
    wide = tmp_path / "wide.tsv"
    wide.write_text("1\t2\t3\n")
    with pytest.raises(ValueError, match="expected 2"):
        gt.prepare_phenotype_table(str(wide), str(tmp_path / "panel"), fly, tmp_path / "w")
    bad = tmp_path / "bad.tsv"
    bad.write_text("1\t2\nfoo-bar\t3\n")
    with pytest.raises(ValueError, match="Unrecognised"):
        gt.prepare_phenotype_table(str(bad), str(tmp_path / "panel"), fly, tmp_path / "w")
    dup = tmp_path / "dup.tsv"
    dup.write_text("DGRP_01\t2\n1\t3\n2\t4\n")
    with pytest.raises(ValueError, match="Duplicate"):
        gt.prepare_phenotype_table(str(dup), str(tmp_path / "panel"), fly, tmp_path / "w")


# --- LD bfile remap ----------------------------------------------------------

def test_build_ld_bfile_remaps_and_keeps_ids(tmp_path, fly):
    src = tmp_path / "merged_qc"
    rows = [("2L", "2L_5670_SNP;rs205447007", 9000), ("3R", "3R_19071977_SNP", 23246255),
            ("4", "4_60836_SNP", 40210), ("23", "X_19511_DEL", 9388593), ("X", "X_20000_INS", 9400000)]
    Path(f"{src}.bim").write_text("".join(f"{c}\t{i}\t0\t{p}\tA\tG\n" for c, i, p in rows))
    Path(f"{src}.bed").write_bytes(b"\x6c\x1b\x01")
    write_fam(src, ["1"])
    before = Path(f"{src}.bim").read_text()

    out = gt.build_ld_bfile(str(src), str(tmp_path / "plink_binary" / "ld_ref"), fly)
    bim = pd.read_csv(f"{out['ld_bfile']}.bim", sep="\t", header=None, dtype=str)
    assert bim[0].tolist() == ["1", "4", "5", "23", "23"]
    assert bim[1].tolist() == [r[1] for r in rows]
    assert bim[3].tolist() == [str(r[2]) for r in rows]
    assert out["chromosome_codes"] == [1, 4, 5, 23]
    assert Path(f"{out['ld_bfile']}.bed").is_symlink()
    assert Path(f"{src}.bim").read_text() == before  # source untouched

    Path(f"{src}.bim").write_text("Y\tY_1_SNP\t0\t1\tA\tG\n")
    with pytest.raises(ValueError, match="Unknown chromosome"):
        gt.build_ld_bfile(str(src), str(tmp_path / "ld2"), fly)


# --- GLM -> contract ---------------------------------------------------------

GLM_HEADER = "#CHROM\tPOS\tID\tREF\tALT\tA1\tOMITTED\tA1_FREQ\tTEST\tOBS_CT\tBETA\tSE\tT_STAT\tP\tERRCODE\n"


def write_glm(path, rows):
    Path(path).write_text(GLM_HEADER + "".join("\t".join(map(str, r)) + "\n" for r in rows))


def test_merge_glm_to_contract_round_trips_through_filter(tmp_path, fly):
    write_glm(tmp_path / "2L.glm", [
        ("2L", 1632386, "2L_1632386_SNP;rs202927577", "A", "G", "G", "A", 0.3, "ADD", 197, 1.5, 0.2, 7.5, 1.069e-07, "."),
        ("2L", 1700000, "2L_1700000_DEL", "AT", "A", "A", ".", 0.2, "ADD", 190, 0.1, 0.2, 0.5, 0.6, "."),
        ("2L", 1700000, "2L_1700000_DEL", "AT", "A", "A", ".", 0.2, "PC1", 190, 0.1, 0.2, 0.5, 0.6, "."),
        ("2L", 1800000, "2L_1800000_SNP", "C", "T", "T", "C", 0.2, "ADD", 190, "NA", "NA", "NA", "NA", "CONST_ALLELE"),
        ("2L", 1900000, "2L_1900000_SNP", "C", "T", "T", "C", 0.2, "ADD", 190, 0.1, 0, 0.5, 0.6, "."),
    ])
    write_glm(tmp_path / "X.glm", [
        ("X", 9388593, "X_19511_DEL", "C", "T", "T", "C", 0.15, "ADD", 188, 0.9, 0.2, 4.5, 4e-06, "."),
    ])
    out = tmp_path / "preprocessed_data" / "gwas_sumstats.tsv.gz"
    gwas = gt.merge_glm_to_contract({"2L": tmp_path / "2L.glm", "X": tmp_path / "X.glm"}, fly, str(out))
    assert list(gwas.columns) == gt.CONTRACT_COLUMNS
    assert gwas["variant_id"].tolist() == ["2L_1632386_SNP;rs202927577", "2L_1700000_DEL", "X_19511_DEL"]
    assert gwas["chromosome"].tolist() == [1, 1, 23]
    assert gwas["chromosome_label"].tolist() == ["2L", "2L", "X"]
    assert gwas.loc[1, "other_allele"] == "AT"   # OMITTED '.' -> REF/ALT fallback
    assert gwas["N"].tolist() == [197, 190, 188]

    with gzip.open(out, "rt") as fh:
        assert fh.readline().rstrip("\n").split("\t") == gt.CONTRACT_COLUMNS
    sig, _ = analysis_module.filter_significant_variants.fn(
        str(out), str(tmp_path), p_threshold=1e-5, min_maf=0.05, min_n=100)
    assert list(sig.index) == ["2L_1632386_SNP;rs202927577", "X_19511_DEL"]
    assert sig["chromosome"].dtype.kind == "i"
    assert sig.loc["X_19511_DEL", "N"] == 188


def test_merge_glm_rejects_wrong_chromosome(tmp_path, fly):
    write_glm(tmp_path / "a.glm", [
        ("3R", 1, "3R_1_SNP", "A", "G", "G", "A", 0.3, "ADD", 10, 1, 1, 1, 0.1, "."),
    ])
    with pytest.raises(ValueError, match="Unexpected chromosomes"):
        gt.merge_glm_to_contract({"2L": tmp_path / "a.glm"}, fly, str(tmp_path / "o.tsv.gz"))


def test_task_requires_exactly_one_genotype_source(tmp_path):
    pheno = tmp_path / "p.tsv"
    pheno.write_text("1\t2\n")
    for kwargs in ({}, {"panel_id": "dgrp2", "genotype_bfile": "/x"}):
        with pytest.raises(ValueError, match="exactly one"):
            gt.run_genotype_gwas_task.fn(None, None, str(pheno), "drosophila", str(tmp_path), **kwargs)


# --- end to end with plink2 --dummy ------------------------------------------

def _sha(prefix):
    return {ext: hashlib.sha256(Path(f"{prefix}{ext}").read_bytes()).hexdigest() for ext in (".bed", ".bim", ".fam")}


def make_dummy_panel(root):
    """60 lines x 600 variants, 100 per arm, DGRP-style IDs whose embedded
    position deliberately differs from the dm6 POS column. Returns
    (prefix, phenotype path, {arm: (causal variant id, dm6 position)})."""
    root.mkdir(parents=True, exist_ok=True)
    prefix = root / "dgrp_like"
    subprocess.run(["plink2", "--dummy", "60", str(PER_ARM * len(ARMS)), "acgt", "--seed", "7",
                    "--make-bed", "--out", str(prefix)], check=True, capture_output=True)
    bim = pd.read_csv(f"{prefix}.bim", sep="\t", header=None, dtype=str)
    labels, ids, pos = [], [], []
    for a, (arm, src_label) in enumerate(ARMS):
        for i in range(PER_ARM):
            vtype = "DEL" if i % 7 == 3 else "SNP"
            vid = f"{arm}_{11 + i * 37}_{vtype}" + (f";rs{1000 + a * PER_ARM + i}" if i % 5 == 0 else "")
            labels.append(src_label)
            ids.append(vid)
            pos.append(str(100_000 + i * 2_000))
    bim[0], bim[1], bim[3] = labels, ids, pos
    bim.to_csv(f"{prefix}.bim", sep="\t", header=False, index=False)
    write_fam(prefix, [str(i + 1) for i in range(60)])

    # Causal variant per arm: the most common one, so QC never drops it.
    subprocess.run(["plink2", "--bfile", str(prefix), "--freq", "--allow-extra-chr",
                    "--out", str(prefix)], check=True, capture_output=True)
    freq = pd.read_csv(f"{prefix}.afreq", sep="\t", dtype={"ID": str})
    freq["maf"] = np.minimum(freq["ALT_FREQS"], 1 - freq["ALT_FREQS"])
    freq["arm"] = [lab for lab in labels]
    freq["pos"] = [int(p) for p in pos]
    causal = {}
    for arm in CAUSAL_ARMS:
        best = freq[freq["arm"] == dict(ARMS)[arm]].sort_values("maf", ascending=False).iloc[0]
        causal[arm] = (best["ID"], int(best["pos"]))

    # Planted effects: additive dosage at the causal 3R and X variants.
    dosages = []
    for vid, _pos in causal.values():
        out = root / "dose"
        subprocess.run(["plink2", "--bfile", str(prefix), "--snp", vid, "--allow-extra-chr",
                        "--export", "A", "--out", str(out)], check=True, capture_output=True)
        raw = pd.read_csv(f"{out}.raw", sep=r"\s+")
        dosages.append(raw.iloc[:, 6].astype(float).fillna(1.0).to_numpy())
    rng = np.random.default_rng(1)
    y = 3 * dosages[0] + 3 * dosages[1] + rng.normal(0, 1, 60)
    pheno = root / "phenotype.tsv"
    lines = [f"DGRP_{i + 1:03d}\t{v:.4f}" for i, v in enumerate(y)] + ["DGRP_999\t1.0"]
    pheno.write_text("line\tvalue\n" + "\n".join(lines) + "\n")
    return str(prefix), str(pheno), causal


def _assert_outputs(result, causal):
    assert result["n_samples"] == 60
    assert result["chromosome_labels"] == ["2L", "2R", "3L", "3R", "4", "X"]
    assert result["chromosome_codes"] == [1, 2, 3, 4, 5, 23]
    gwas = pd.read_csv(result["sumstats_path"], sep="\t", compression="gzip", dtype={"variant_id": str})
    assert list(gwas.columns) == gt.CONTRACT_COLUMNS
    assert len(gwas) == result["n_variants"] > 400
    assert gwas["chromosome"].dtype.kind == "i" and gwas["N"].dtype.kind == "i"
    assert set(gwas["chromosome"]) == {1, 2, 3, 4, 5, 23}
    assert gwas["N"].eq(60).all()
    assert gwas["variant_id"].str.contains(";rs").any()

    ld_bim = pd.read_csv(f"{result['ld_bfile']}.bim", sep="\t", header=None, dtype=str)
    assert set(ld_bim[0].astype(int)) == {1, 2, 3, 4, 5, 23}
    assert set(gwas["variant_id"]) <= set(ld_bim[1])        # IDs match .bim column 2
    # dm6 position comes from POS, not the ID
    vid, pos = causal["3R"]
    row = gwas.set_index("variant_id").loc[vid]
    assert row["base_pair_location"] == pos
    assert row["chromosome"] == 4 and row["chromosome_label"] == "3R"
    assert row["p_value"] < 1e-5


@needs_plink
def test_end_to_end_direct_bfile(tmp_path, monkeypatch):
    prefix, pheno, causal = make_dummy_panel(tmp_path / "src")
    before = _sha(prefix)
    src_listing = sorted(p.name for p in (tmp_path / "src").iterdir())
    monkeypatch.setattr(gt, "_require_binaries", lambda *a: None)
    out_dir = tmp_path / "analysis"
    result = gt.run_genotype_gwas_task.fn(
        None, None, pheno, "drosophila", str(out_dir), genotype_bfile=prefix,
        overrides={"genotype": {"min_lines": 50}},
    )
    _assert_outputs(result, causal)
    assert _sha(prefix) == before
    assert sorted(p.name for p in (tmp_path / "src").iterdir()) == src_listing


@needs_plink
def test_end_to_end_hosted_panel(tmp_path, monkeypatch):
    prefix, pheno, causal = make_dummy_panel(tmp_path / "src")
    panel_dir = tmp_path / "panels" / "dgrp2"
    panel_dir.mkdir(parents=True)
    for arm, src_label in ARMS:
        out = panel_dir / f"DGRP.{arm}"
        subprocess.run(["plink2", "--bfile", prefix, "--chr", src_label, "--allow-extra-chr",
                        "--make-bed", "--out", str(out)], check=True, capture_output=True)
    before = {arm: _sha(panel_dir / f"DGRP.{arm}") for arm, _ in ARMS}
    monkeypatch.setenv("GENOTYPE_PANEL_DIR", str(tmp_path / "panels"))
    out_dir = tmp_path / "analysis"
    result = gt.run_genotype_gwas_task.fn(
        None, None, pheno, "drosophila", str(out_dir), panel_id="dgrp2",
        overrides={"genotype": {"min_lines": 50}},
    )
    _assert_outputs(result, causal)
    assert {arm: _sha(panel_dir / f"DGRP.{arm}") for arm, _ in ARMS} == before


@needs_gcta
def test_genotype_outputs_drive_real_cojo(tmp_path, monkeypatch):
    """The contract + LD bfile run through the real GCTA COJO task, including --chr 23."""
    prefix, pheno, causal = make_dummy_panel(tmp_path / "src")
    out_dir = tmp_path / "analysis"
    result = gt.run_genotype_gwas_task.fn(
        None, None, pheno, "drosophila", str(out_dir), genotype_bfile=prefix,
        overrides={"genotype": {"min_lines": 50}},
    )
    sig, _ = analysis_module.filter_significant_variants.fn(
        result["sumstats_path"], str(out_dir), p_threshold=1e-4, min_maf=0.05, min_n=50)
    cojo_tmp = tmp_path / "cojo_tmp"
    cojo_tmp.mkdir()
    monkeypatch.setattr(analysis_module, "get_shared_temp_dir", lambda **_kw: str(cojo_tmp))
    cojo_df, _ = analysis_module.run_cojo_per_chromosome.fn(
        sig, "/unused", str(out_dir), maf_threshold=0.05, population="test",
        chromosomes=result["chromosome_codes"], ld_bfile=result["ld_bfile"], cojo_p=1e-4)
    # With 60 random lines a neighbour in chance LD may win the window, so
    # check the arms and that GCTA's chr/bp agree with the contract columns.
    assert set(cojo_df["CHR"].astype(int)) == {4, 23}   # 3R and X; X (23) not dropped
    gwas = pd.read_csv(result["sumstats_path"], sep="\t", compression="gzip",
                       dtype={"variant_id": str}).set_index("variant_id")
    for vid, row in cojo_df.iterrows():
        assert int(row["bp"]) == gwas.loc[vid, "base_pair_location"]
        assert int(row["CHR"]) == gwas.loc[vid, "chromosome"]

    batches = analysis_module.create_region_batches(cojo_df, coords_from_columns=True)
    regions = {r["variant_id"]: r for b in batches for r in b}
    vid_x = cojo_df.index[cojo_df["CHR"].astype(int) == 23][0]
    assert (regions[vid_x]["chr"], regions[vid_x]["position"]) == (23, gwas.loc[vid_x, "base_pair_location"])


@needs_plink
def test_real_plink_ld_on_genotype_ld_reference(tmp_path, monkeypatch):
    """calculate_ld_matrix with real plink 1.9 on the per-run LD reference:
    ';rs' IDs survive --extract, and chrX (code 23, .fam sex 0) is not dropped."""
    prefix, pheno, causal = make_dummy_panel(tmp_path / "src")
    result = gt.run_genotype_gwas_task.fn(
        None, None, pheno, "drosophila", str(tmp_path / "analysis"), genotype_bfile=prefix,
        overrides={"genotype": {"min_lines": 50}},
    )
    # Same body as the run_command task, without starting Prefect.
    monkeypatch.setattr(analysis_module, "run_command",
                        lambda cmd: subprocess.run(cmd, shell=True, capture_output=True, text=True))
    sumstats = pd.read_csv(result["sumstats_path"], sep="\t", compression="gzip",
                           dtype={"variant_id": str}).set_index("variant_id")
    ld_bim = pd.read_csv(f"{result['ld_bfile']}.bim", sep="\t", header=None, dtype=str).set_index(1)

    for arm in ("3R", "X"):
        code = 4 if arm == "3R" else 23
        region = sumstats[sumstats["chromosome"] == code].iloc[:30]
        ld_df, sub = analysis_module.calculate_ld_matrix(
            chr_num=code, filtered_ids=region.index.tolist(), sumstats=sumstats, plink_dir=None,
            population="test", ref_genome="dm6", maf=0.01, config=analysis_module.Config(),
            ld_bfile=result["ld_bfile"])
        assert ld_df is not None and len(ld_df) >= 2, arm
        assert any(";rs" in v for v in ld_df.index), arm
        off_diag = ld_df.values[~np.eye(len(ld_df), dtype=bool)]
        assert np.abs(off_diag).max() > 0, arm
        assert list(sub.index) == list(ld_df.index)

    # Allele orientation: the z-score allele (plink2 A1) vs the LD .bim A1 (plink 1.9).
    common = sumstats.index.intersection(ld_bim.index)
    mismatched = int((sumstats.loc[common, "effect_allele"] != ld_bim.loc[common, 4]).sum())
    assert mismatched == 0, f"{mismatched} variants with effect_allele != LD .bim A1"
