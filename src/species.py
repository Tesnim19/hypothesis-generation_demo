"""Species profiles for the analysis pipeline.

A profile (config/species/<name>.yaml) holds everything organism-specific:
chromosome labels and their numeric PLINK/GCTA codes, thresholds, the
fine-mapping window, SuSiE settings and which pipeline steps run.

Only genotype input mode reads profiles today. The human sumstats path keeps
using the flow parameters directly, so its behaviour cannot drift from a YAML
edit; human.yaml documents those values and is pinned by tests.
"""
from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger


INPUT_TYPES = ("sumstats", "genotype")
SUMSTATS_SCOPES = ("significant", "all_in_window")
COJO_SCOPES = ("per_chromosome", "genome_wide")
_NAME_RE = re.compile(r"^[a-z0-9_]+$")
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_REQUIRED_KEYS = (
    "species", "genome_build", "supported_input_types", "chromosomes",
    "chrom_code_map", "ld_reference", "thresholds", "finemap", "susie", "steps",
)
_REQUIRED_THRESHOLDS = ("p_threshold", "cojo_p", "cojo_maf", "ld_maf", "min_maf", "min_n")


@dataclass(frozen=True)
class SpeciesProfile:
    name: str
    species: str
    genome_build: str
    supported_input_types: tuple
    chromosomes: tuple
    chrom_code_map: dict
    chrom_aliases: dict = field(default_factory=dict)
    plink_chr_set: object = None
    ld_reference: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    finemap: dict = field(default_factory=dict)
    susie: dict = field(default_factory=dict)
    steps: dict = field(default_factory=dict)
    annotations: dict = field(default_factory=dict)
    genotype: dict | None = None

    @property
    def chromosome_codes(self) -> list[int]:
        """Numeric codes of `chromosomes`, in profile order."""
        return [self.chrom_code_map[c] for c in self.chromosomes]

    @property
    def code_to_label(self) -> dict:
        return {code: label for label, code in self.chrom_code_map.items()}

    def canonical_label(self, label) -> str:
        """Map a label as written by plink or a source panel to the profile label.

        Accepts the profile label itself ("2L") or an alias ("23" -> "X").
        Numeric codes are deliberately not accepted: for the fly, "4" is the
        label of arm 4 (code 5) while code 4 is 3R. Raises ValueError otherwise.
        """
        label = str(label)
        if label in self.chrom_code_map:
            return label
        if label in self.chrom_aliases:
            return self.chrom_aliases[label]
        raise ValueError(f"Unknown chromosome {label!r} for species profile {self.name!r}")

    def chrom_code(self, label) -> int:
        return self.chrom_code_map[self.canonical_label(label)]


def _deep_merge(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _expand_vars(value, env: dict):
    if isinstance(value, dict):
        return {k: _expand_vars(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_vars(v, env) for v in value]
    if isinstance(value, str):
        def repl(m):
            if m.group(1) not in env:
                raise ValueError(f"Species profile references unset variable ${{{m.group(1)}}}")
            return env[m.group(1)]
        return _VAR_RE.sub(repl, value)
    return value


def _validate(name: str, raw: dict) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"Species profile {name!r} missing keys: {missing}")
    bad_types = set(raw["supported_input_types"]) - set(INPUT_TYPES)
    if bad_types or not raw["supported_input_types"]:
        raise ValueError(f"Species profile {name!r} has invalid supported_input_types: {raw['supported_input_types']}")

    chroms = [str(c) for c in raw["chromosomes"]]
    code_map = raw["chrom_code_map"]
    if len(set(chroms)) != len(chroms):
        raise ValueError(f"Species profile {name!r} lists a chromosome twice")
    if set(chroms) != set(map(str, code_map)):
        raise ValueError(f"Species profile {name!r}: chromosomes and chrom_code_map keys differ")
    codes = list(code_map.values())
    if any(not isinstance(c, int) or isinstance(c, bool) or c < 1 for c in codes):
        raise ValueError(f"Species profile {name!r}: chrom codes must be positive integers")
    if len(set(codes)) != len(codes):
        raise ValueError(f"Species profile {name!r}: chrom codes must be unique")
    for alias, target in (raw.get("chrom_aliases") or {}).items():
        if str(target) not in code_map:
            raise ValueError(f"Species profile {name!r}: alias {alias!r} points to unknown chromosome {target!r}")
        if str(alias) in code_map:
            raise ValueError(f"Species profile {name!r}: alias {alias!r} shadows a chromosome label")

    missing_thr = [k for k in _REQUIRED_THRESHOLDS if k not in raw["thresholds"]]
    if missing_thr:
        raise ValueError(f"Species profile {name!r} missing thresholds: {missing_thr}")
    if raw["finemap"].get("sumstats_scope") not in SUMSTATS_SCOPES:
        raise ValueError(f"Species profile {name!r}: finemap.sumstats_scope must be one of {SUMSTATS_SCOPES}")
    if raw["finemap"].get("ld_nonfinite", "zero_fill") not in ("zero_fill", "drop"):
        raise ValueError(f"Species profile {name!r}: finemap.ld_nonfinite must be 'zero_fill' or 'drop'")
    if raw["thresholds"].get("cojo_scope", "per_chromosome") not in COJO_SCOPES:
        raise ValueError(f"Species profile {name!r}: thresholds.cojo_scope must be one of {COJO_SCOPES}")
    if raw["ld_reference"].get("kind") not in ("panel_per_chromosome", "per_run"):
        raise ValueError(f"Species profile {name!r}: unknown ld_reference.kind")
    if "genotype" in raw["supported_input_types"]:
        geno = raw.get("genotype") or {}
        for key in ("qc", "min_lines", "iid_regex", "panels"):
            if key not in geno:
                raise ValueError(f"Species profile {name!r}: genotype.{key} is required for genotype input")
        re.compile(geno["iid_regex"])


def _profile_dir() -> str:
    from src.config import Config  # lazy: keeps this module light to import
    return Config.from_env().species_profile_dir


def _var_env() -> dict:
    from src.config import Config
    config = Config.from_env()
    env = dict(os.environ)
    env.setdefault("DATA_DIR", config.data_dir)
    env.setdefault("GENOTYPE_PANEL_DIR", config.genotype_panel_dir)
    return env


def load_species_profile(name: str, overrides: dict | None = None,
                         profile_dir: str | None = None) -> SpeciesProfile:
    """Load config/species/<name>.yaml, apply nested *overrides*, validate."""
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f"Invalid species profile name: {name!r}")
    profile_dir = Path(profile_dir or _profile_dir())
    path = profile_dir / f"{name}.yaml"
    if not path.is_file():
        # Also accept the `species` field value, e.g. "drosophila_melanogaster".
        for candidate in sorted(profile_dir.glob("*.yaml")):
            with open(candidate) as fh:
                if (yaml.safe_load(fh) or {}).get("species") == name:
                    path = candidate
                    break
        else:
            raise ValueError(f"No species profile {name!r} in {profile_dir}")

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    raw = _deep_merge(raw, overrides or {})
    raw = _expand_vars(raw, _var_env())
    _validate(name, raw)

    chromosomes = tuple(str(c) for c in raw["chromosomes"])
    profile = SpeciesProfile(
        name=path.stem,
        species=raw["species"],
        genome_build=str(raw["genome_build"]),
        supported_input_types=tuple(raw["supported_input_types"]),
        chromosomes=chromosomes,
        chrom_code_map={str(k): int(v) for k, v in raw["chrom_code_map"].items()},
        chrom_aliases={str(k): str(v) for k, v in (raw.get("chrom_aliases") or {}).items()},
        plink_chr_set=raw.get("plink_chr_set"),
        ld_reference=raw["ld_reference"],
        thresholds=raw["thresholds"],
        finemap=raw["finemap"],
        susie=raw["susie"],
        steps=raw["steps"],
        annotations=raw.get("annotations") or {},
        genotype=raw.get("genotype"),
    )
    logger.info(f"[SPECIES] Loaded profile {profile.name} ({profile.species}, {profile.genome_build}) from {path}")
    return profile


# Keys accepted in `param_overrides` (genotype mode only).
EFFECTIVE_PARAM_KEYS = (
    "p_threshold", "cojo_p", "cojo_maf", "cojo_scope", "ld_maf", "min_maf", "min_n",
    "window_kb", "sumstats_scope", "L", "coverage", "min_abs_corr",
)


def resolve_effective_params(profile: SpeciesProfile, input_type: str, flow_params: dict,
                             param_overrides: dict | None = None) -> dict:
    """Resolve the parameters the pipeline should use.

    sumstats mode: the flow parameters win, exactly as today; the profile only
    supplies chromosome codes. genotype mode: the profile wins, with `null`
    entries falling back to the flow parameters, then *param_overrides* on top.
    Flow defaults (window=2000, maf=0.01, L=-1) cannot be told apart from
    user-passed values, so they never override the profile.
    """
    if input_type not in INPUT_TYPES:
        raise ValueError(f"Unknown input_type {input_type!r}; expected one of {INPUT_TYPES}")
    if input_type not in profile.supported_input_types:
        raise ValueError(
            f"Species profile {profile.name!r} does not support input_type={input_type!r} "
            f"(supported: {list(profile.supported_input_types)})"
        )

    maf = flow_params.get("maf_threshold", 0.01)
    if input_type == "sumstats":
        if param_overrides:
            raise ValueError("param_overrides is only supported in genotype input mode")
        return {
            "p_threshold": 5e-8,
            "cojo_p": None,
            "cojo_maf": maf,
            "cojo_scope": "per_chromosome",
            "ld_maf": maf,
            "min_maf": None,
            "min_n": None,
            "window_kb": flow_params.get("window", 2000),
            "sumstats_scope": "significant",
            "L": flow_params.get("L", -1),
            "coverage": flow_params.get("coverage", 0.95),
            "min_abs_corr": flow_params.get("min_abs_corr", 0.5),
            "estimate_residual_variance": None,
            "sample_size_stat": None,
            "drop_nonfinite_ld": None,
            "chromosome_codes": profile.chromosome_codes,
            "steps": dict(profile.steps),
        }

    def pick(section, key, fallback):
        value = section.get(key)
        return fallback if value is None else value

    thr, fm, su = profile.thresholds, profile.finemap, profile.susie
    params = {
        "p_threshold": pick(thr, "p_threshold", 5e-8),
        "cojo_p": thr.get("cojo_p"),
        "cojo_maf": pick(thr, "cojo_maf", maf),
        "cojo_scope": pick(thr, "cojo_scope", "per_chromosome"),
        "ld_maf": pick(thr, "ld_maf", maf),
        "min_maf": thr.get("min_maf"),
        "min_n": thr.get("min_n"),
        "window_kb": pick(fm, "window_kb", flow_params.get("window", 2000)),
        "sumstats_scope": fm.get("sumstats_scope", "significant"),
        "L": pick(su, "L", flow_params.get("L", -1)),
        "coverage": pick(su, "coverage", flow_params.get("coverage", 0.95)),
        "min_abs_corr": pick(su, "min_abs_corr", flow_params.get("min_abs_corr", 0.5)),
        "estimate_residual_variance": su.get("estimate_residual_variance"),
        "sample_size_stat": su.get("sample_size"),
        "drop_nonfinite_ld": fm.get("ld_nonfinite") == "drop",
    }
    unknown = set(param_overrides or {}) - set(EFFECTIVE_PARAM_KEYS)
    if unknown:
        raise ValueError(f"Unknown param_overrides keys: {sorted(unknown)}")
    params.update(param_overrides or {})
    if params["sumstats_scope"] not in SUMSTATS_SCOPES:
        raise ValueError(f"sumstats_scope must be one of {SUMSTATS_SCOPES}")
    if params["cojo_scope"] not in COJO_SCOPES:
        raise ValueError(f"cojo_scope must be one of {COJO_SCOPES}")
    if params["sample_size_stat"] not in (None, "first", "median"):
        raise ValueError("susie.sample_size must be null, 'first' or 'median'")
    params["chromosome_codes"] = profile.chromosome_codes
    params["steps"] = dict(profile.steps)
    return params
