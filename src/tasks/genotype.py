"""Genotype input mode: genotypes + phenotype -> GWAS summary statistics.

Steps: keep the phenotyped lines,
per-chromosome QC, merge, LD prune, PCA, then ``plink2 --glm`` per chromosome
with the leading PCs as covariates. The stage writes

* ``<output_dir>/preprocessed_data/gwas_sumstats.tsv.gz``: the harmonized
  sumstats column contract read by filter_significant_variants, and
* ``<output_dir>/plink_binary/ld_ref.{bed,bim,fam}``: the QC'd merged set with
  numeric chromosome codes, used as the LD reference for COJO and SuSiE.

Genotypes come either from a server-hosted panel (already split per
chromosome, resolved through the species profile) or from a single PLINK
prefix (split per chromosome here). Source files are only ever read.

Variant IDs are kept verbatim from the .bim (DGRP: ``3R_19071977_SNP``,
``2L_5670_SNP;rs205447007``). Chromosome and position always come from
explicit columns; the IDs carry old-build coordinates and are never parsed.

External tools run through ``subprocess.run([...], check=True)`` directly
rather than the run_command task, so no task is nested in another.
"""
import gzip
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from prefect import task

from src.species import load_species_profile


PHENO_NAME = "trait"
# Pinned so OMITTED and A1_FREQ are present whatever the plink2 default is.
GLM_COLS = "chrom,pos,ref,alt,omitted,a1freq,test,nobs,beta,se,tz,p,err"
CONTRACT_COLUMNS = [
    "variant_id", "chromosome", "base_pair_location", "effect_allele", "other_allele",
    "beta", "standard_error", "p_value", "effect_allele_frequency", "N",
    "chromosome_label", "source",
]
_GLM_REQUIRED = ["#CHROM", "POS", "ID", "REF", "ALT", "A1", "OMITTED", "A1_FREQ",
                 "TEST", "OBS_CT", "BETA", "SE", "P"]
_BFILE_EXTS = (".bed", ".bim", ".fam")


# === HELPERS ===

def _run(cmd):
    logger.info(f"[GENOTYPE] Running: {shlex.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"[GENOTYPE] Command failed ({e.returncode}): {shlex.join(cmd)}\n{e.stderr or e.stdout}")
        raise


def _trio_exists(prefix):
    return all(Path(f"{prefix}{ext}").is_file() for ext in _BFILE_EXTS)


def _require_trio(prefix, what):
    missing = [f"{prefix}{ext}" for ext in _BFILE_EXTS if not Path(f"{prefix}{ext}").is_file()]
    if missing:
        raise FileNotFoundError(f"[GENOTYPE] Missing {what} files: {missing}")


def _require_binaries(*names):
    missing = [n for n in names if shutil.which(n) is None]
    if missing:
        raise FileNotFoundError(f"[GENOTYPE] Required binaries not on PATH: {missing}")


def _read_fam(prefix):
    return pd.read_csv(f"{prefix}.fam", sep=r"\s+", header=None, dtype=str,
                       usecols=[0, 1, 4], names=["FID", "IID", "SEX"])


def _extra_chr(profile):
    return ["--allow-extra-chr"] if profile.genotype["qc"].get("allow_extra_chr") else []


def normalize_line_id(value, iid_regex):
    """Normalize a line ID with the profile regex (group 1). Pure digits lose
    leading zeros, so DGRP_021, 021 and 21 all become "21". None if no match."""
    m = re.match(iid_regex, str(value).strip())
    if not m:
        return None
    ident = m.group(1)
    return str(int(ident)) if ident.isdigit() else ident


# === PHENOTYPE ===

def _read_table(path, has_header):
    with open(path, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    opener = gzip.open if gz else open
    with opener(path, "rt") as fh:
        first = fh.readline()
    sep = "\t" if "\t" in first else ("," if "," in first else r"\s+")
    return pd.read_csv(path, sep=sep, header=0 if has_header else None, dtype=str,
                       engine="python" if sep == r"\s+" else "c", compression="infer")


def prepare_phenotype_table(phenotype_path, fam_prefix, profile, work_dir, *,
                            id_column=None, phenotype_column=None, sex=None,
                            sex_column="sex", min_lines=None):
    """Write PLINK ``.pheno`` (FID IID trait) and keep files for the matched lines.

    Default input is a 2-column table (line, value), header optional. A wider
    table (e.g. DGRPool's DGRP / sex / mn_Lifespan) needs *id_column* and
    *phenotype_column*; *sex* keeps only rows whose *sex_column* equals it.
    Line IDs are normalised on both sides with the profile ``iid_regex``.
    """
    iid_regex = profile.genotype["iid_regex"]
    min_lines = profile.genotype["min_lines"] if min_lines is None else min_lines
    named = id_column is not None or phenotype_column is not None or sex is not None

    if named:
        table = _read_table(phenotype_path, has_header=True)
        id_column = id_column or table.columns[0]
        if phenotype_column is None:
            raise ValueError("[GENOTYPE] phenotype_column is required when id_column or sex is given")
        needed = [id_column, phenotype_column] + ([sex_column] if sex is not None else [])
        missing = [c for c in needed if c not in table.columns]
        if missing:
            raise ValueError(f"[GENOTYPE] Phenotype table lacks columns {missing}; have {list(table.columns)}")
        if sex is not None:
            table = table[table[sex_column].astype(str).str.strip().str.upper() == str(sex).strip().upper()]
            logger.info(f"[GENOTYPE] Kept {len(table)} phenotype rows with {sex_column}={sex}")
        pheno = pd.DataFrame({"line": table[id_column], "value": table[phenotype_column]})
    else:
        table = _read_table(phenotype_path, has_header=False)
        if table.shape[1] != 2:
            raise ValueError(
                f"[GENOTYPE] Phenotype table has {table.shape[1]} columns; expected 2 (line, value) "
                f"or pass id_column/phenotype_column"
            )
        table.columns = ["line", "value"]
        if pd.isna(pd.to_numeric(table["value"].iloc[0], errors="coerce")):
            table = table.iloc[1:]  # header row
        pheno = table

    pheno = pheno.copy()
    pheno["IID_norm"] = [normalize_line_id(v, iid_regex) for v in pheno["line"]]
    bad = pheno.loc[pheno["IID_norm"].isna(), "line"].tolist()
    if bad:
        raise ValueError(f"[GENOTYPE] Unrecognised line IDs (regex {iid_regex}): {bad[:5]}")
    pheno["value"] = pd.to_numeric(pheno["value"], errors="coerce")
    n_missing = int(pheno["value"].isna().sum())
    if n_missing:
        logger.warning(f"[GENOTYPE] Dropping {n_missing} lines with missing/non-numeric phenotype")
        pheno = pheno.dropna(subset=["value"])
    dups = pheno.loc[pheno["IID_norm"].duplicated(keep=False), "line"].tolist()
    if dups:
        raise ValueError(f"[GENOTYPE] Duplicate line IDs in phenotype table: {dups[:6]}")

    fam = _read_fam(fam_prefix)
    fam["IID_norm"] = [normalize_line_id(v, iid_regex) for v in fam["IID"]]
    merged = pheno.merge(fam[["FID", "IID", "IID_norm"]], on="IID_norm", how="inner")
    n_unmatched = len(pheno) - len(merged)
    if n_unmatched:
        logger.warning(f"[GENOTYPE] {n_unmatched} phenotyped lines have no genotypes and are dropped")
    if len(merged) < min_lines:
        raise ValueError(
            f"[GENOTYPE] Only {len(merged)} phenotyped lines have genotypes; at least {min_lines} required"
        )

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    pheno_path = work_dir / f"{PHENO_NAME}.pheno"
    keep_path = work_dir / "analysis_lines.keep"
    merged[["FID", "IID", "value"]].rename(columns={"value": PHENO_NAME}).to_csv(
        pheno_path, sep="\t", index=False)
    merged[["FID", "IID"]].to_csv(keep_path, sep="\t", index=False, header=False)
    logger.info(f"[GENOTYPE] Prepared phenotype for {len(merged)} lines: {pheno_path}")
    return {"pheno_path": str(pheno_path), "keep_path": str(keep_path), "n_lines": len(merged)}


# === GENOTYPE SOURCES ===

def resolve_panel(profile, panel_id):
    """Per-chromosome prefixes of a server-hosted panel, keyed by profile label."""
    panels = profile.genotype.get("panels") or {}
    if panel_id not in panels:
        raise ValueError(f"[GENOTYPE] Unknown genotype panel {panel_id!r} for {profile.name}; "
                         f"available: {sorted(panels)}")
    panel = panels[panel_id]
    per_chrom = {c: os.path.join(panel["dir"], panel["per_chrom_prefix"].format(chrom=c))
                 for c in profile.chromosomes}
    for chrom, prefix in per_chrom.items():
        _require_trio(prefix, f"panel {panel_id} chromosome {chrom}")
    return per_chrom


def _rewrite_bim_chrom(bim_path, label_for):
    """Rewrite column 1 of a .bim in place via label_for(source_label)."""
    tmp = f"{bim_path}.tmp"
    try:
        with open(bim_path) as src, open(tmp, "w") as dst:
            for n, line in enumerate(src, 1):
                fields = line.split()
                if len(fields) != 6:
                    raise ValueError(f"[GENOTYPE] Malformed .bim row {n} in {bim_path}")
                fields[0] = str(label_for(fields[0]))
                dst.write("\t".join(fields) + "\n")
        os.replace(tmp, bim_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def split_bfile_per_chrom(bfile, profile, out_dir):
    """Split one all-chromosome PLINK prefix into per-chromosome sets in *out_dir*.

    Source labels are matched through the profile (labels plus aliases, so X
    written as "23" works); unknown labels (scaffolds, Y, mito) are skipped.
    The output .bim carries the profile label.
    """
    _require_trio(bfile, "genotype bfile")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    observed = pd.read_csv(f"{bfile}.bim", sep=r"\s+", header=None, usecols=[0], dtype=str)[0].unique()
    sources = {c: [] for c in profile.chromosomes}
    skipped = []
    for label in observed:
        try:
            sources[profile.canonical_label(label)].append(label)
        except ValueError:
            skipped.append(label)
    if skipped:
        logger.warning(f"[GENOTYPE] Ignoring chromosomes not in the {profile.name} profile: {skipped}")

    per_chrom = {}
    for chrom in profile.chromosomes:
        if not sources[chrom]:
            logger.warning(f"[GENOTYPE] No variants on chromosome {chrom} in {bfile}")
            continue
        prefix = str(out_dir / f"split.{chrom}")
        _run(["plink2", "--bfile", str(bfile), "--chr", ",".join(sources[chrom]),
              "--allow-extra-chr", "--make-bed", "--out", prefix])
        _rewrite_bim_chrom(f"{prefix}.bim", lambda _src, c=chrom: c)
        per_chrom[chrom] = prefix
    if not per_chrom:
        raise ValueError(f"[GENOTYPE] No {profile.name} chromosomes found in {bfile}")
    return per_chrom


# === QC / PCA / GLM ===

def qc_merge_pca(per_chrom, keep_path, profile, work_dir):
    """Per-chromosome QC, merge, LD prune and PCA."""
    qc = profile.genotype["qc"]
    extra = _extra_chr(profile)
    qc_dir = Path(work_dir) / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    qc_prefixes = {}
    for chrom, prefix in per_chrom.items():
        out = str(qc_dir / chrom)
        _run(["plink2", "--bfile", prefix, "--keep", str(keep_path),
              "--maf", str(qc["maf"]), "--geno", str(qc["geno"]), *extra,
              "--make-bed", "--out", out])
        _require_trio(out, f"QC output for {chrom}")
        qc_prefixes[chrom] = out

    merged = str(Path(work_dir) / "merged_qc")
    chroms = list(qc_prefixes)
    if len(chroms) > 1:
        merge_list = Path(work_dir) / "merge_list.txt"
        merge_list.write_text("".join(
            f"{qc_prefixes[c]}.bed {qc_prefixes[c]}.bim {qc_prefixes[c]}.fam\n" for c in chroms[1:]))
        _run(["plink", "--bfile", qc_prefixes[chroms[0]], "--merge-list", str(merge_list),
              *extra, "--make-bed", "--out", merged])
    else:
        _run(["plink", "--bfile", qc_prefixes[chroms[0]], *extra, "--make-bed", "--out", merged])
    _require_trio(merged, "merged QC set")

    prune = str(Path(work_dir) / "pca_prune")
    _run(["plink2", "--bfile", merged, *extra, "--indep-pairwise", *map(str, qc["indep_pairwise"]),
          "--out", prune])
    pca = str(Path(work_dir) / "pca")
    # plink2 rejects arm labels like 2L without --allow-extra-chr.
    _run(["plink2", "--bfile", merged, *extra, "--extract", f"{prune}.prune.in",
          "--pca", str(qc["pca_compute"]), "--out", pca])
    eigenvec = f"{pca}.eigenvec"
    if not os.path.isfile(eigenvec):
        raise RuntimeError(f"[GENOTYPE] PCA produced no eigenvectors: {eigenvec}")
    return {"qc_prefixes": qc_prefixes, "merged_prefix": merged, "eigenvec": eigenvec}


def _check_sample_alignment(prefix, keep_path, chrom):
    fam = _read_fam(prefix)
    keep = pd.read_csv(keep_path, sep="\t", header=None, dtype=str, names=["FID", "IID"])
    if set(zip(fam.FID, fam.IID)) != set(zip(keep.FID, keep.IID)):
        raise ValueError(f"[GENOTYPE] Samples in {prefix}.fam do not match the analysis lines ({chrom})")


def run_glm_per_chrom(qc_prefixes, pheno_path, keep_path, eigenvec, profile, work_dir):
    """plink2 --glm per chromosome with PC1..PCk covariates."""
    n_pcs = profile.genotype["qc"]["n_pcs_covar"]
    glm_dir = Path(work_dir) / "glm"
    glm_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for chrom, prefix in qc_prefixes.items():
        _check_sample_alignment(prefix, keep_path, chrom)
        out = str(glm_dir / chrom)
        _run(["plink2", "--bfile", prefix, "--pheno", str(pheno_path), "--pheno-name", PHENO_NAME,
              "--covar", eigenvec, "--covar-name", f"PC1-PC{n_pcs}",
              "--glm", "hide-covar", f"cols={GLM_COLS}",
              "--out", out, "--no-psam-pheno", "--allow-extra-chr"])
        result = f"{out}.{PHENO_NAME}.glm.linear"
        if not os.path.isfile(result):
            raise RuntimeError(f"[GENOTYPE] plink2 produced no GWAS output for {chrom}: {result}")
        outputs[chrom] = result
    return outputs


def merge_glm_to_contract(glm_files, profile, out_path):
    """Merge per-chromosome plink2 results into the sumstats contract (gzip TSV).

    ``chromosome`` is the numeric code, ``chromosome_label`` the profile label
    (taken from the per-chromosome loop, cross-checked against #CHROM).
    """
    frames = []
    for chrom, path in glm_files.items():
        df = pd.read_csv(path, sep="\t", dtype={"ID": str, "#CHROM": str}, low_memory=False)
        missing = [c for c in _GLM_REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"[GENOTYPE] {path} lacks columns {missing}")
        n_non_add = int((df["TEST"] != "ADD").sum())
        if n_non_add:
            logger.info(f"[GENOTYPE] Excluded {n_non_add} non-additive tests from {path}")
        df = df[df["TEST"] == "ADD"]
        observed = {profile.canonical_label(c) for c in df["#CHROM"].unique()}
        if observed - {chrom}:
            raise ValueError(f"[GENOTYPE] Unexpected chromosomes {sorted(observed)} in {path} (expected {chrom})")

        other = df["OMITTED"].where(~df["OMITTED"].isin([".", ""]) & df["OMITTED"].notna(),
                                    np.where(df["A1"] == df["REF"], df["ALT"], df["REF"]))
        frames.append(pd.DataFrame({
            "variant_id": df["ID"],
            "chromosome": profile.chrom_code(chrom),
            "base_pair_location": pd.to_numeric(df["POS"], errors="coerce"),
            "effect_allele": df["A1"],
            "other_allele": other,
            "beta": pd.to_numeric(df["BETA"], errors="coerce"),
            "standard_error": pd.to_numeric(df["SE"], errors="coerce"),
            "p_value": pd.to_numeric(df["P"], errors="coerce"),
            "effect_allele_frequency": pd.to_numeric(df["A1_FREQ"], errors="coerce"),
            "N": pd.to_numeric(df["OBS_CT"], errors="coerce"),
            "chromosome_label": chrom,
            "source": "plink2_glm",
        }))
    gwas = pd.concat(frames, ignore_index=True)

    # Validity filters: finite values, distinct alleles, valid frequency, p, se and N
    numeric = ["base_pair_location", "beta", "standard_error", "p_value", "effect_allele_frequency", "N"]
    valid = (
        np.isfinite(gwas[numeric].to_numpy(dtype=float)).all(axis=1)
        & gwas[["variant_id", "effect_allele", "other_allele"]].notna().all(axis=1).to_numpy()
        & (gwas["effect_allele"] != gwas["other_allele"]).to_numpy()
        & gwas["effect_allele_frequency"].between(0, 1).to_numpy()
        & gwas["p_value"].between(0, 1).to_numpy()
        & (gwas["base_pair_location"] > 0).to_numpy()
        & (gwas["standard_error"] > 0).to_numpy()
        & (gwas["N"] > 0).to_numpy()
    )
    n_invalid = int((~valid).sum())
    if n_invalid:
        logger.info(f"[GENOTYPE] Excluded {n_invalid:,} invalid GWAS rows")
    gwas = gwas[valid].copy()
    if gwas.empty:
        raise ValueError("[GENOTYPE] No valid GWAS results remain after quality checks")
    dups = gwas.loc[gwas["variant_id"].duplicated(), "variant_id"].tolist()
    if dups:
        raise ValueError(f"[GENOTYPE] Duplicate variant IDs in GWAS results: {dups[:5]}")
    gwas["base_pair_location"] = gwas["base_pair_location"].astype(np.int64)
    gwas["N"] = gwas["N"].astype(np.int64)
    gwas["chromosome"] = gwas["chromosome"].astype(np.int64)
    gwas = gwas[CONTRACT_COLUMNS]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{out_path}.tmp"
    gwas.to_csv(tmp, sep="\t", index=False, compression="gzip")
    os.replace(tmp, out_path)
    logger.info(f"[GENOTYPE] Wrote {len(gwas):,} variants to {out_path}")
    return gwas


def build_ld_bfile(merged_prefix, out_prefix, profile):
    """LD reference for COJO/SuSiE: symlink .bed/.fam, rewrite .bim chromosomes
    to numeric codes. Unknown chromosomes raise."""
    _require_trio(merged_prefix, "merged QC set")
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    for ext in (".bed", ".fam"):
        src = Path(f"{merged_prefix}{ext}").resolve()
        dst = Path(f"{out_prefix}{ext}")
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    shutil.copyfile(f"{merged_prefix}.bim", f"{out_prefix}.bim")
    codes = set()

    def to_code(label):
        code = profile.chrom_code(label)
        codes.add(code)
        return code

    _rewrite_bim_chrom(f"{out_prefix}.bim", to_code)
    with open(f"{out_prefix}.bim") as fh:
        n_variants = sum(1 for _ in fh)
    logger.info(f"[GENOTYPE] LD reference {out_prefix}: {n_variants:,} variants, chromosome codes {sorted(codes)}")
    return {"ld_bfile": str(out_prefix), "n_variants": n_variants, "chromosome_codes": sorted(codes)}


# === PREFECT TASK ===

@task()
def run_genotype_gwas_task(user_id, project_id, phenotype_path, species, output_dir,
                           panel_id=None, genotype_bfile=None, overrides=None,
                           phenotype_options=None):
    """Stage 0b of genotype input mode: genotypes + phenotype -> sumstats + LD reference.

    Exactly one of *panel_id* (hosted per-chromosome panel from the species
    profile) or *genotype_bfile* (a single PLINK prefix) must be given.
    *phenotype_options* may hold id_column, phenotype_column, sex, sex_column.
    """
    if (panel_id is None) == (genotype_bfile is None):
        raise ValueError("[GENOTYPE] Give exactly one of genotype_panel_id or genotype_bfile")
    if not phenotype_path or not os.path.isfile(phenotype_path):
        raise FileNotFoundError(f"[GENOTYPE] Phenotype file not found: {phenotype_path}")
    _require_binaries("plink", "plink2")

    profile = load_species_profile(species, overrides=overrides)
    if "genotype" not in profile.supported_input_types:
        raise ValueError(f"[GENOTYPE] Species profile {profile.name} does not support genotype input")

    work_dir = Path(output_dir) / "genotype_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[GENOTYPE] Species {profile.name}, source "
                f"{'panel ' + panel_id if panel_id else 'bfile ' + str(genotype_bfile)}, work dir {work_dir}")

    if panel_id is not None:
        per_chrom = resolve_panel(profile, panel_id)
        fam_prefix = next(iter(per_chrom.values()))
    else:
        _require_trio(genotype_bfile, "genotype bfile")
        fam_prefix = genotype_bfile
        per_chrom = split_bfile_per_chrom(genotype_bfile, profile, work_dir / "split")

    pheno = prepare_phenotype_table(phenotype_path, fam_prefix, profile, work_dir,
                                    **(phenotype_options or {}))
    qc = qc_merge_pca(per_chrom, pheno["keep_path"], profile, work_dir)
    glm_files = run_glm_per_chrom(qc["qc_prefixes"], pheno["pheno_path"], pheno["keep_path"],
                                  qc["eigenvec"], profile, work_dir)

    sumstats_path = os.path.join(output_dir, "preprocessed_data", "gwas_sumstats.tsv.gz")
    gwas = merge_glm_to_contract(glm_files, profile, sumstats_path)
    ld = build_ld_bfile(qc["merged_prefix"], os.path.join(output_dir, "plink_binary", "ld_ref"), profile)

    if user_id and project_id:
        try:
            from src.utils import get_deps
            storage = get_deps()["storage"]
            object_key = f"generated_sumstats/{user_id}/{project_id}/{os.path.basename(sumstats_path)}"
            if storage and storage.upload_file(sumstats_path, object_key):
                logger.info(f"[GENOTYPE] Uploaded to MinIO: s3://hypothesis/{object_key}")
            else:
                logger.warning("[GENOTYPE] Failed to upload sumstats to MinIO, file remains local only")
        except Exception as e:
            logger.warning(f"[GENOTYPE] MinIO upload skipped: {e}")

    return {
        "sumstats_path": sumstats_path,
        "ld_bfile": ld["ld_bfile"],
        "n_samples": pheno["n_lines"],
        "n_variants": len(gwas),
        "ld_n_variants": ld["n_variants"],
        "chromosome_codes": [profile.chrom_code(c) for c in qc["qc_prefixes"]],
        "chromosome_labels": list(qc["qc_prefixes"]),
    }
