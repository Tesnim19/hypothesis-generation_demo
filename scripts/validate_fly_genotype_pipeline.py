#!/usr/bin/env python3
"""Local validation of genotype input mode against a reference DGRP analysis.

Not part of the service and not run in CI. Calls the pipeline's task
functions directly (``.fn``) with no Mongo/MinIO: no user or project id is
passed, so nothing is saved or uploaded.

Mode A (``--mode a``): the reference merged sumstats + merged_qc set
  -> our sumstats contract -> LD reference -> filter -> COJO -> SuSiE.
  Isolates our downstream wiring from any GWAS differences.
Mode B (``--mode b``): DGRP panel + phenotype -> our genotype stage
  -> the same downstream, plus a sumstats diff against the reference.

Checks (INFO rows are reported but not judged):
  1. LD .bim has 2,258,460 variants; 949,142 with MAF >= 0.05 (plink --freq).
  2. N = 197 lines.
  3. (Mode B) Same SNP set as the reference sumstats; |dbeta|, |dse| <= 1e-6 relative;
     p equal to 4 significant figures.
  4. Exactly 15 SNPs go into COJO.
  5. COJO signals mapped back to arms are exactly
     {2L:1632386, 2R:19928970, 3R:23246255, X:9388593}.
  6. SuSiE L=10: the 3R:23246255 locus has a credible set containing that SNP
     with top PIP >= 0.9 (reference: 0.952); 1-3 credible sets in total.
  7. (INFO) The L=-1 (Optuna) run.

SuSiE needs rpy2 + susieR (the dask-worker image has them; a bare host
usually does not). Without them checks 6-7 are reported as SKIP.

Point --data-dir (or $DGRP_REFERENCE_DATA_DIR) at the reference data/ folder,
then run from the repo root, e.g. inside the dask-worker container:
  python scripts/validate_fly_genotype_pipeline.py --mode a
  python scripts/validate_fly_genotype_pipeline.py --mode b --source panel
  python scripts/validate_fly_genotype_pipeline.py --mode a --check-inputs
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

DATA_DIR_ENV = "DGRP_REFERENCE_DATA_DIR"
FLY_ARMS = ("2L", "2R", "3L", "3R", "4", "X")

EXPECTED_LD_VARIANTS = 2_258_460
EXPECTED_LD_MAF05 = 949_142
EXPECTED_N = 197
EXPECTED_COJO_INPUT = 15
EXPECTED_SIGNALS = {("2L", 1632386), ("2R", 19928970), ("3R", 23246255), ("X", 9388593)}
SUSIE_LOCUS = ("3R", 23246255)
SUSIE_MIN_TOP_PIP = 0.9
SUSIE_CS_RANGE = (1, 3)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["a", "b"], required=True,
                   help="a: the reference sumstats + merged_qc; b: panel + phenotype -> full run")
    p.add_argument("--data-dir", type=Path, default=os.environ.get(DATA_DIR_ENV),
                   help=f"Root of the reference data/ layout (default: ${DATA_DIR_ENV})")
    p.add_argument("--reference-sumstats", type=Path,
                   help="Reference merged sumstats (default <data>/gwas/tmp/lifespan_female_gwas_sumstats.tsv)")
    p.add_argument("--merged-qc", type=Path,
                   help="Reference merged QC PLINK prefix, Mode A LD source (default <data>/gwas/tmp/merged_qc)")
    p.add_argument("--source", choices=["panel", "bfile"], default="panel",
                   help="Mode B genotypes: per-arm panel <data>/reference/DGRP.{arm} or the whole-genome "
                        "<data>/reference/dgrp2_dm6_dbSNP.vcf prefix")
    p.add_argument("--phenotype", type=Path,
                   help="Mode B phenotype (default <data>/gwas/lifespan_female_raw.tsv.gz, DGRPool format)")
    p.add_argument("--phenotype-format", choices=["dgrpool", "pheno", "two-column"], default="dgrpool",
                   help="dgrpool: DGRP/sex/mn_Lifespan, females; pheno: FID IID S18_1537_F; "
                        "two-column: line, value")
    p.add_argument("--out-dir", type=Path, default=Path(tempfile.gettempdir()) / "hypgen_fly_validation",
                   help="Where all outputs go (never inside --data-dir)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-susie", action="store_true", help="Stop after COJO (checks 1-5)")
    p.add_argument("--skip-optuna", action="store_true", help="Skip the informational L=-1 run")
    p.add_argument("--check-inputs", action="store_true", help="Only check that inputs exist, then exit")
    p.add_argument("--min-lines", type=int, default=None,
                   help="Mode B: override the profile's genotype.min_lines (dry runs on small data)")
    p.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                   help="Profile parameter override (e.g. min_n=50), for dry runs on other data; "
                        "the expected values above assume none")
    args = p.parse_args(argv)
    if args.data_dir is None:
        p.error(f"--data-dir is required (or set ${DATA_DIR_ENV})")
    d = Path(args.data_dir)
    args.data_dir = d
    args.reference_sumstats = args.reference_sumstats or d / "gwas" / "tmp" / "lifespan_female_gwas_sumstats.tsv"
    args.merged_qc = args.merged_qc or d / "gwas" / "tmp" / "merged_qc"
    args.phenotype = args.phenotype or (
        d / "gwas" / "lifespan_female.pheno" if args.phenotype_format == "pheno"
        else d / "gwas" / "lifespan_female_raw.tsv.gz")
    return args


def required_inputs(args):
    trio = lambda prefix: [Path(f"{prefix}{ext}") for ext in (".bed", ".bim", ".fam")]
    files = [args.reference_sumstats]
    if args.mode == "a":
        files += trio(args.merged_qc)
    else:
        files.append(args.phenotype)
        ref = args.data_dir / "reference"
        if args.source == "panel":
            for arm in FLY_ARMS:
                files += trio(ref / f"DGRP.{arm}")
        else:
            files += trio(ref / "dgrp2_dm6_dbSNP.vcf")
    return files


class Report:
    def __init__(self):
        self.rows = []

    def add(self, check, status, detail):
        self.rows.append((check, status, detail))
        print(f"[VALIDATE] {status:4s} {check}: {detail}", flush=True)

    def expect(self, check, ok, detail):
        self.add(check, "PASS" if ok else "FAIL", detail)

    def summary(self):
        print("\n==== Summary ====")
        for check, status, detail in self.rows:
            print(f"{status:4s}  {check:38s} {detail}")
        return not any(s == "FAIL" for _, s, _ in self.rows)


# === conversion (Mode A) ===

def reference_sumstats_to_contract(reference_path, profile, out_path):
    """Reference CHR POS SNP A1 A2 freq b se p N table -> our gzip sumstats contract."""
    import pandas as pd
    ref = pd.read_csv(reference_path, sep="\t", dtype={"SNP": str, "CHR": str}, low_memory=False)
    contract = pd.DataFrame({
        "variant_id": ref["SNP"],
        "chromosome": [profile.chrom_code(c) for c in ref["CHR"]],
        "base_pair_location": ref["POS"].astype("int64"),
        "effect_allele": ref["A1"],
        "other_allele": ref["A2"],
        "beta": ref["b"],
        "standard_error": ref["se"],
        "p_value": ref["p"],
        "effect_allele_frequency": ref["freq"],
        "N": ref["N"].round().astype("int64"),
        "chromosome_label": [profile.canonical_label(c) for c in ref["CHR"]],
        "source": "reference",
    })
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    contract.to_csv(out_path, sep="\t", index=False, compression="gzip")
    return contract


# === checks ===

def check_ld_reference(report, ld_bfile, work):
    import subprocess
    import pandas as pd
    with open(f"{ld_bfile}.bim") as fh:
        n_bim = sum(1 for _ in fh)
    out = work / "ld_freq"
    subprocess.run(["plink", "--bfile", ld_bfile, "--freq", "--out", str(out)],
                   check=True, capture_output=True)
    frq = pd.read_csv(f"{out}.frq", sep=r"\s+")
    n_maf = int((frq["MAF"] >= 0.05).sum())
    report.expect("1. LD reference size", n_bim == EXPECTED_LD_VARIANTS and n_maf == EXPECTED_LD_MAF05,
                  f"{n_bim:,} variants (want {EXPECTED_LD_VARIANTS:,}), {n_maf:,} MAF>=0.05 "
                  f"(want {EXPECTED_LD_MAF05:,})")
    with open(f"{ld_bfile}.fam") as fh:
        return sum(1 for _ in fh)


def allele_orientation(report, contract_path, ld_bfile):
    """SuSiE z-scores use the sumstats effect allele; the LD matrix uses the LD
    .bim A1. The pipeline never aligns them, so count disagreements."""
    import pandas as pd
    ours = pd.read_csv(contract_path, sep="\t", compression="gzip", dtype={"variant_id": str},
                       usecols=["variant_id", "effect_allele"]).set_index("variant_id")
    bim = pd.read_csv(f"{ld_bfile}.bim", sep=r"\s+", header=None, dtype=str, usecols=[1, 4]).set_index(1)
    common = ours.index.intersection(bim.index)
    flipped = int((ours.loc[common, "effect_allele"] != bim.loc[common, 4]).sum())
    report.add("   effect allele vs LD .bim A1", "INFO",
               f"{flipped:,} of {len(common):,} shared variants disagree; "
               f"{len(ours.index.difference(bim.index)):,} sumstats IDs absent from the LD .bim")


def compare_to_reference(report, ours_path, reference_path):
    import numpy as np
    import pandas as pd
    ours = pd.read_csv(ours_path, sep="\t", compression="gzip", dtype={"variant_id": str}).set_index("variant_id")
    ref = pd.read_csv(reference_path, sep="\t", dtype={"SNP": str}).set_index("SNP")
    only_ours, only_reference = ours.index.difference(ref.index), ref.index.difference(ours.index)
    common = ours.index.intersection(ref.index)
    o, h = ours.loc[common], ref.loc[common]
    allele_mismatch = int((o["effect_allele"] != h["A1"]).sum())

    def rel(a, b):
        return (np.abs(a - b) / np.maximum(np.abs(b), 1e-300)).max()

    d_beta, d_se = rel(o["beta"], h["b"]), rel(o["standard_error"], h["se"])
    p_mismatch = int((o["p_value"].map(lambda x: f"{x:.3e}") != h["p"].map(lambda x: f"{x:.3e}")).sum())
    ok = (len(only_ours) == 0 and len(only_reference) == 0 and allele_mismatch == 0
          and d_beta <= 1e-6 and d_se <= 1e-6 and p_mismatch == 0)
    report.expect("3. sumstats match the reference", ok,
                  f"{len(common):,} common, {len(only_ours)} only ours, {len(only_reference)} only reference, "
                  f"A1 mismatches {allele_mismatch}, max rel dbeta {d_beta:.2e}, dse {d_se:.2e}, "
                  f"p mismatches (4 s.f.) {p_mismatch}")
    n_med = ours["N"].median()
    report.add("   N per SNP (ours)", "INFO", f"median {n_med}, range {ours['N'].min()}-{ours['N'].max()}")


def run_susie(report, label, contract_path, cojo_df, ld_bfile, profile, params, L, seed, work, judged):
    from src.tasks import analysis as am
    if not am.HAS_SUSIE:
        report.add(label, "SKIP", "rpy2/susieR not available on this host (run in the dask-worker container)")
        return
    subset = am.subset_sumstats_to_windows(contract_path, cojo_df, params["window_kb"])
    # Round-trip through the worker file format, as the real pipeline does
    worker_file = am.save_sumstats_for_workers.fn(subset, str(work))
    sumstats = am.load_sumstats_from_file(worker_file)
    regions = [r for b in am.create_region_batches(cojo_df, batch_size=100, coords_from_columns=True) for r in b]
    n_sets, locus_ok, locus_detail = 0, False, "locus not fine-mapped"
    lead_code, lead_pos = profile.chrom_code(SUSIE_LOCUS[0]), SUSIE_LOCUS[1]
    lead_ids = sumstats.index[(sumstats["chromosome"] == lead_code) & (sumstats["base_pair_location"] == lead_pos)]
    for region in regions:
        res = am.finemap_region(
            seed=seed, sumstats=sumstats, chr_num=region["chr"], lead_variant_position=region["position"],
            window=params["window_kb"], population="dgrp2", L=L,
            coverage=params["coverage"], min_abs_corr=params["min_abs_corr"],
            maf=params["ld_maf"], ref_genome=profile.genome_build, ld_bfile=ld_bfile,
            estimate_residual_variance=params["estimate_residual_variance"],
            sample_size_stat=params["sample_size_stat"],
            drop_nonfinite_ld=params["drop_nonfinite_ld"],
        )
        arm = profile.code_to_label[region["chr"]]
        if res is None or len(res) == 0:
            report.add(f"   {label} {arm}:{region['position']}", "INFO", "no credible set")
            continue
        sets = sorted(res["cs"].unique())
        n_sets += len(sets)
        report.add(f"   {label} {arm}:{region['position']}", "INFO",
                   "; ".join(f"cs{c}: {int((res['cs'] == c).sum())} variants, top PIP "
                             f"{res.loc[res['cs'] == c, 'PIP'].max():.3f}" for c in sets))
        if (region["chr"], region["position"]) == (lead_code, lead_pos):
            hit = res[res.index.isin(lead_ids)]
            if len(hit):
                cs = hit["cs"].iloc[0]
                top = res.loc[res["cs"] == cs, "PIP"].max()
                locus_ok = top >= SUSIE_MIN_TOP_PIP
                locus_detail = f"{list(lead_ids)} in cs{cs}, top PIP {top:.3f}"
            else:
                locus_detail = f"{list(lead_ids)} not in any credible set"
    if judged:
        in_range = SUSIE_CS_RANGE[0] <= n_sets <= SUSIE_CS_RANGE[1]
        report.expect(f"6. {label} {SUSIE_LOCUS[0]}:{SUSIE_LOCUS[1]}", locus_ok, locus_detail)
        report.expect(f"6. {label} credible sets", in_range, f"{n_sets} total (want {SUSIE_CS_RANGE})")
    else:
        report.add(f"7. {label}", "INFO", f"{n_sets} credible sets; {locus_detail}")


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.out_dir.resolve()
    if str(out_dir).startswith(str(args.data_dir.resolve())):
        sys.exit("--out-dir must not be inside --data-dir (inputs are read-only)")

    missing = [str(f) for f in required_inputs(args) if not f.is_file()]
    if missing:
        print("[VALIDATE] Missing inputs:\n  " + "\n  ".join(missing))
        if args.check_inputs:
            return 1
        sys.exit(1)
    if args.check_inputs:
        print("[VALIDATE] All inputs present")
        return 0

    # Heavy imports only after argument parsing so --help stays fast.
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from src.species import load_species_profile, resolve_effective_params
    from src.tasks import analysis as am
    from src.tasks import genotype as gt

    work = out_dir / f"mode_{args.mode}"
    (work / "preprocessed_data").mkdir(parents=True, exist_ok=True)
    temp_root = work / "temp"
    temp_root.mkdir(exist_ok=True)
    # get_shared_temp_dir hard-codes /app/data/temp (container path); it is
    # imported by name into src.tasks.analysis, so patch it there.
    am.get_shared_temp_dir = lambda user_id=None, prefix="": tempfile.mkdtemp(prefix=f"{prefix}_", dir=temp_root)

    profile = load_species_profile("drosophila")
    overrides = {}
    for item in args.override:
        key, _, value = item.partition("=")
        overrides[key] = None if value.lower() == "none" else (
            value if key == "sumstats_scope" else float(value) if any(c in value for c in ".e") else int(value))
    params = resolve_effective_params(profile, "genotype", {}, overrides or None)
    report = Report()
    print(f"[VALIDATE] Mode {args.mode.upper()}, outputs in {work}")
    print(f"[VALIDATE] Effective parameters: {params}")

    if args.mode == "a":
        contract_path = str(work / "preprocessed_data" / "gwas_sumstats.tsv.gz")
        reference_sumstats_to_contract(args.reference_sumstats, profile, contract_path)
        ld_bfile = gt.build_ld_bfile(str(args.merged_qc), str(work / "plink_binary" / "ld_ref"), profile)["ld_bfile"]
        n_lines = check_ld_reference(report, ld_bfile, work)
        report.expect("2. N lines", n_lines == EXPECTED_N, f"{n_lines} in .fam (want {EXPECTED_N})")
    else:
        if args.phenotype_format == "dgrpool":
            pheno_opts = {"id_column": "DGRP", "phenotype_column": "mn_Lifespan", "sex": "F"}
        elif args.phenotype_format == "pheno":
            pheno_opts = {"id_column": "IID", "phenotype_column": "S18_1537_F"}
        else:
            pheno_opts = None
        ref = args.data_dir / "reference"
        geno_overrides = {"panels": {"dgrp2": {"dir": str(ref)}}}
        if args.min_lines is not None:
            geno_overrides["min_lines"] = args.min_lines
        source = ({"panel_id": "dgrp2"} if args.source == "panel"
                  else {"genotype_bfile": str(ref / "dgrp2_dm6_dbSNP.vcf")})
        result = gt.run_genotype_gwas_task.fn(
            None, None, str(args.phenotype), "drosophila", str(work),
            phenotype_options=pheno_opts, overrides={"genotype": geno_overrides}, **source)
        contract_path, ld_bfile = result["sumstats_path"], result["ld_bfile"]
        report.expect("2. N lines", result["n_samples"] == EXPECTED_N,
                      f"{result['n_samples']} matched lines (want {EXPECTED_N})")
        check_ld_reference(report, ld_bfile, work)
        compare_to_reference(report, contract_path, args.reference_sumstats)

    allele_orientation(report, contract_path, ld_bfile)

    sig, _ = am.filter_significant_variants.fn(
        contract_path, str(work), p_threshold=params["p_threshold"],
        min_maf=params["min_maf"], min_n=params["min_n"])
    report.expect("4. SNPs into COJO", len(sig) == EXPECTED_COJO_INPUT,
                  f"{len(sig)} (want {EXPECTED_COJO_INPUT})")
    if len(sig) == 0:
        report.add("5. COJO signals", "FAIL", "no SNPs passed the filter; downstream not run")
        return 0 if report.summary() else 2

    try:
        cojo_df, _ = am.run_cojo_per_chromosome.fn(
            sig, None, str(work), maf_threshold=params["cojo_maf"], population="dgrp2",
            ref_genome=profile.genome_build, chromosomes=profile.chromosome_codes,
            ld_bfile=ld_bfile, cojo_p=params["cojo_p"],
            genome_wide=params["cojo_scope"] == "genome_wide")
    except Exception as e:
        report.add("5. COJO signals", "FAIL", f"COJO failed: {e}")
        return 0 if report.summary() else 2
    signals = {(profile.code_to_label[int(r["CHR"])], int(r["bp"])) for _, r in cojo_df.iterrows()}
    report.expect("5. COJO signals", signals == EXPECTED_SIGNALS,
                  f"{sorted(signals)}; missing {sorted(EXPECTED_SIGNALS - signals)}, "
                  f"extra {sorted(signals - EXPECTED_SIGNALS)}")

    if not args.skip_susie:
        run_susie(report, f"SuSiE L={params['L']}", contract_path, cojo_df, ld_bfile, profile, params,
                  L=params["L"], seed=args.seed, work=work, judged=True)
        if not args.skip_optuna:
            run_susie(report, "SuSiE L=-1 (Optuna)", contract_path, cojo_df, ld_bfile, profile, params,
                      L=-1, seed=args.seed, work=work, judged=False)

    return 0 if report.summary() else 2


if __name__ == "__main__":
    sys.exit(main())
