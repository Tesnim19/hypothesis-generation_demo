import pytest

from src.species import (
    EFFECTIVE_PARAM_KEYS,
    load_species_profile,
    resolve_effective_params,
)

HUMAN_FLOW_DEFAULTS = {"maf_threshold": 0.01, "window": 2000, "L": -1,
                       "coverage": 0.95, "min_abs_corr": 0.5}


def test_human_profile_resolves_to_todays_values():
    profile = load_species_profile("human")
    assert profile.supported_input_types == ("sumstats",)
    assert profile.chromosome_codes == list(range(1, 23))
    assert profile.ld_reference == {"kind": "panel_per_chromosome"}
    assert profile.steps["harmonize"] and profile.steps["ldsc_tissue"] and profile.steps["opentargets"]

    params = resolve_effective_params(profile, "sumstats", HUMAN_FLOW_DEFAULTS)
    assert params["p_threshold"] == 5e-8
    assert params["cojo_p"] is None
    assert params["cojo_maf"] == params["ld_maf"] == 0.01
    assert params["min_maf"] is None and params["min_n"] is None
    assert params["window_kb"] == 2000
    assert params["L"] == -1
    assert params["coverage"] == 0.95 and params["min_abs_corr"] == 0.5
    assert params["sumstats_scope"] == "significant"
    assert params["cojo_scope"] == "per_chromosome"
    assert params["estimate_residual_variance"] is None and params["sample_size_stat"] is None
    assert params["chromosome_codes"] == list(range(1, 23))


def test_human_profile_matches_filter_default_p_threshold():
    import inspect
    from src.tasks.analysis import filter_significant_variants
    default = inspect.signature(filter_significant_variants.fn).parameters["p_threshold"].default
    assert load_species_profile("human").thresholds["p_threshold"] == default


def test_human_ld_pattern_unchanged():
    from src.config import Config
    assert Config().get_plink_file_pattern(population="EUR", chrom=7) == "EUR.chr7.1KG.GRCh38"


def test_human_rejects_genotype_mode():
    with pytest.raises(ValueError, match="does not support"):
        resolve_effective_params(load_species_profile("human"), "genotype", HUMAN_FLOW_DEFAULTS)


def test_fly_code_map_and_aliases():
    fly = load_species_profile("drosophila")
    assert fly.chromosomes == ("2L", "2R", "3L", "3R", "4", "X")
    assert fly.chromosome_codes == [1, 2, 3, 4, 5, 23]
    assert fly.chrom_code("X") == 23
    assert fly.chrom_code("23") == 23          # plink / DGRP merged_qc.bim write X as 23
    assert fly.chrom_code("4") == 5            # arm 4, not 3R
    assert fly.canonical_label("chr2L") == "2L"
    with pytest.raises(ValueError):
        fly.chrom_code("Y")
    with pytest.raises(ValueError):
        fly.chrom_code("1")  # numeric codes are not labels


def test_fly_lookup_by_species_name():
    assert load_species_profile("drosophila_melanogaster").name == "drosophila"


def test_fly_effective_params_profile_wins():
    fly = load_species_profile("drosophila")
    params = resolve_effective_params(fly, "genotype", HUMAN_FLOW_DEFAULTS)
    assert params["p_threshold"] == 1e-5
    assert params["cojo_p"] == 1e-5
    assert params["cojo_maf"] == 0.05
    assert params["cojo_scope"] == "genome_wide"
    assert params["ld_maf"] == 0.01
    assert params["min_maf"] == 0.05 and params["min_n"] == 100
    assert params["window_kb"] == 100
    assert params["sumstats_scope"] == "all_in_window"
    assert params["L"] == 10
    assert params["estimate_residual_variance"] is True and params["sample_size_stat"] == "median"
    assert params["steps"]["harmonize"] is False and params["steps"]["ldsc_tissue"] is False
    with pytest.raises(ValueError, match="does not support"):
        resolve_effective_params(fly, "sumstats", HUMAN_FLOW_DEFAULTS)


def test_fly_param_overrides():
    fly = load_species_profile("drosophila")
    params = resolve_effective_params(fly, "genotype", HUMAN_FLOW_DEFAULTS, {"L": -1, "p_threshold": 1e-6})
    assert params["L"] == -1 and params["p_threshold"] == 1e-6
    with pytest.raises(ValueError, match="Unknown param_overrides"):
        resolve_effective_params(fly, "genotype", HUMAN_FLOW_DEFAULTS, {"nonsense": 1})
    assert "window_kb" in EFFECTIVE_PARAM_KEYS


def test_profile_overrides_and_env_expansion(monkeypatch):
    monkeypatch.setenv("GENOTYPE_PANEL_DIR", "/panels")
    monkeypatch.setenv("DATA_DIR", "/data")
    fly = load_species_profile("drosophila", overrides={"genotype": {"min_lines": 5}})
    assert fly.genotype["min_lines"] == 5
    assert fly.genotype["qc"]["maf"] == 0.01  # nested merge keeps siblings
    assert fly.genotype["panels"]["dgrp2"]["dir"] == "/panels/dgrp2"
    assert fly.annotations["enhancer_dir"] == "/data/enhancers/dm"


@pytest.mark.parametrize("name", ["../etc/passwd", "Human", "", None])
def test_invalid_profile_names(name):
    with pytest.raises(ValueError):
        load_species_profile(name)


def test_validation_rejects_bad_code_map(tmp_path):
    (tmp_path / "bad.yaml").write_text(
        "species: bad\ngenome_build: x\nsupported_input_types: [sumstats]\n"
        "chromosomes: ['a', 'b']\nchrom_code_map: {a: 1, b: 1}\n"
        "ld_reference: {kind: per_run}\n"
        "thresholds: {p_threshold: 1, cojo_p: null, cojo_maf: null, ld_maf: null, min_maf: null, min_n: null}\n"
        "finemap: {sumstats_scope: significant}\nsusie: {}\nsteps: {}\n"
    )
    with pytest.raises(ValueError, match="unique"):
        load_species_profile("bad", profile_dir=str(tmp_path))
