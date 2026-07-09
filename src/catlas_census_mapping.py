from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class ResolvedCensusCellFilter:
    """How to query Census ``obs`` for one LDSC cell-type name."""

    ldsc_name: str
    #: Try these `cell_type` string filters in order (already lowercase where needed).
    cell_type_labels: list[str] = field(default_factory=list)
    #: Try these CL ids in order after labels exhausted.
    cl_ids: list[str] = field(default_factory=list)
    source: str = "passthrough"
    skip_coexpression: bool = False
    skip_reason: Optional[str] = None


def _escape_soma_string_literal(value: str) -> str:
    """Escape single quotes for SOMA value_filter string literals."""
    return value.replace("'", "''")


def _expand_label_candidates(col2: str) -> list[str]:
    """Prefer full string; if comma-separated, also try each part."""
    s = col2.strip()
    if not s:
        return []
    candidates: list[str] = [s]
    if "," in s:
        for part in s.split(","):
            p = part.strip()
            if p and p not in candidates:
                candidates.append(p)
    return candidates


def _parse_cl_ids_from_cell_ontology_ids(raw: str) -> tuple[list[str], bool]:
    """
    Parse the third column of Cell_ontology.tsv.
    Returns (ordered CL ids, uberon_only) — uberon_only True if there is no CL token.
    """
    tokens = [t.strip() for t in raw.replace(" ", "").split(",") if t.strip()]
    cls: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        if t.upper().startswith("CL:"):
            if t not in seen:
                seen.add(t)
                cls.append(t)
    if not cls:
        return [], True
    return cls, False


class CatlasCensusMapper:
    """
    Load Catlas TSVs and resolve LDSC ``Name`` (underscore form) to Census filters.
    """

    def __init__(
        self,
        cell_ontology_path: str,
        catlas_aliases_path: str,
    ) -> None:
        self._cell_ontology_path = cell_ontology_path
        self._catlas_aliases_path = catlas_aliases_path

        # "Atrial Cardiomyocyte" -> (col2 label string, col3 ids raw)
        self._by_spaced_cell_type: dict[str, tuple[str, str]] = {}
        self._cre_key_to_cl: dict[str, str] = {}

        self._load()

    def _load(self) -> None:
        with open(self._cell_ontology_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                cell = (row.get("Cell type") or "").strip()
                label = (row.get("closest Cell Ontology term(s)") or "").strip()
                ids_raw = (row.get("Cell Ontology ID") or "").strip()
                if cell and label and ids_raw:
                    self._by_spaced_cell_type[cell] = (label, ids_raw)

        with open(self._catlas_aliases_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                cre = (row.get("cre_key") or "").strip()
                cl = (row.get("ontology_id") or "").strip()
                if cre and cl.upper().startswith("CL:"):
                    self._cre_key_to_cl[cre] = cl

        logger.info(
            f"[CatlasCensus] loaded {len(self._by_spaced_cell_type)} Cell Ontology rows, "
            f"{len(self._cre_key_to_cl)} cre_key aliases"
        )

    def _first_row_for_cl(self, cl: str) -> tuple[str, str] | None:
        """First ontology row whose id column contains this CL id."""
        for _cell, (label, ids_raw) in self._by_spaced_cell_type.items():
            cls, _ = _parse_cl_ids_from_cell_ontology_ids(ids_raw)
            if cl in cls:
                return label, ids_raw
        return None

    def resolve(self, ldsc_name: str) -> ResolvedCensusCellFilter:
        spaced = ldsc_name.replace("_", " ")

        def apply_ontology_row(col2: str, ids_raw: str, src: str) -> ResolvedCensusCellFilter:
            cls, uberon_only = _parse_cl_ids_from_cell_ontology_ids(ids_raw)
            if uberon_only or not cls:
                reason = "uberon_only" if uberon_only else "no_cl_in_row"
                logger.info(
                    f"[CatlasCensus] skip ldsc={ldsc_name!r} reason={reason} ({src})"
                )
                return ResolvedCensusCellFilter(
                    ldsc_name=ldsc_name,
                    source=src,
                    skip_coexpression=True,
                    skip_reason=reason,
                )
            lab_out: list[str] = []
            for cand in _expand_label_candidates(col2):
                low = cand.lower()
                if low not in lab_out:
                    lab_out.append(low)
            return ResolvedCensusCellFilter(
                ldsc_name=ldsc_name,
                cell_type_labels=lab_out,
                cl_ids=cls,
                source=src,
            )

        if ldsc_name in self._cre_key_to_cl:
            cl_primary = self._cre_key_to_cl[ldsc_name]
            row2 = self._by_spaced_cell_type.get(spaced)
            if row2:
                col2, ids_raw = row2
                return apply_ontology_row(col2, ids_raw, "alias_cell_ontology")
            hit = self._first_row_for_cl(cl_primary)
            if hit:
                col2, ids_raw = hit
                return apply_ontology_row(col2, ids_raw, "alias_lookup")
            return ResolvedCensusCellFilter(
                ldsc_name=ldsc_name,
                cl_ids=[cl_primary],
                source="alias_cl_only",
            )

        row = self._by_spaced_cell_type.get(spaced)
        if row:
            col2, ids_raw = row
            return apply_ontology_row(col2, ids_raw, "cell_ontology")

        return ResolvedCensusCellFilter(
            ldsc_name=ldsc_name,
            cell_type_labels=[spaced.lower()],
            cl_ids=[],
            source="passthrough",
        )


_mapper_instance: CatlasCensusMapper | None = None


def resolve_ldsc_for_census(
    ldsc_name: str,
    *,
    repo_root: str,
    cell_ontology_rel: str,
    catlas_aliases_rel: str,
) -> ResolvedCensusCellFilter:
    """Resolve using Catlas tables under ``repo_root``; passthrough if files are missing."""
    co = os.path.normpath(os.path.join(repo_root, cell_ontology_rel))
    al = os.path.normpath(os.path.join(repo_root, catlas_aliases_rel))
    if not os.path.isfile(co) or not os.path.isfile(al):
        logger.warning(
            f"[CatlasCensus] missing mapping files co={co} exists={os.path.isfile(co)} "
            f"al={al} exists={os.path.isfile(al)}; passthrough"
        )
        spaced = ldsc_name.replace("_", " ").lower()
        return ResolvedCensusCellFilter(
            ldsc_name=ldsc_name,
            cell_type_labels=[spaced],
            source="passthrough_missing_tables",
        )
    return get_catlas_census_mapper(co, al).resolve(ldsc_name)


def get_catlas_census_mapper(
    cell_ontology_path: str,
    catlas_aliases_path: str,
) -> CatlasCensusMapper:
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = CatlasCensusMapper(
            cell_ontology_path=cell_ontology_path,
            catlas_aliases_path=catlas_aliases_path,
        )
    return _mapper_instance


def reset_catlas_census_mapper_for_tests() -> None:
    global _mapper_instance
    _mapper_instance = None
