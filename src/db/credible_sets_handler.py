"""
Handler for querying OpenTargets pre-computed credible sets from PostgreSQL.

Used to check whether a GWAS study already has credible sets available so the
pipeline can skip local SuSiE fine-mapping and return OpenTargets results directly.
"""

import json
import math
import os
from typing import Optional

from loguru import logger

try:
    import psycopg2
    import psycopg2.extras
    _PSYCOPG2_OK = True
except ImportError:
    _PSYCOPG2_OK = False


def _parse_variant_id(variant_id: str) -> dict:
    """Parse 'chr_pos_ref_alt' into component fields."""
    parts = variant_id.split("_")
    if len(parts) >= 4:
        return {
            "chromosome": parts[0],
            "position": int(parts[1]),
            "ref_allele": parts[2],
            "minor_allele": parts[3],
        }
    return {"chromosome": None, "position": None, "ref_allele": None, "minor_allele": None}


def _log_pvalue(mantissa, exponent) -> Optional[float]:
    """Convert mantissa/exponent p-value to -log10(p)."""
    if mantissa is None or exponent is None:
        return None
    try:
        return -(math.log10(float(mantissa)) + int(exponent))
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def convert_ot_row_to_credible_set(ot_row: dict, coverage: float = 0.95) -> dict:
    """
    Convert one OpenTargets credible set row (from credible_sets table) into the
    format expected by AnalysisHandler.save_credible_set().
    """
    locus = ot_row.get("locus") or []
    if isinstance(locus, str):
        locus = json.loads(locus)

    variants, posterior_probs, betas, chromosomes = [], [], [], []
    log_pvalues, positions, ref_alleles, minor_alleles, rs_ids = [], [], [], [], []
    ref_allele_freqs = []

    lead_eaf = ot_row.get("effect_allele_frequency")
    lead_log_p = _log_pvalue(ot_row.get("p_value_mantissa"), ot_row.get("p_value_exponent"))

    for v in locus:
        vid = v.get("variantId", "")
        parsed = _parse_variant_id(vid)

        variants.append(vid)
        posterior_probs.append(float(v.get("posteriorProbability") or 0))
        betas.append(float(v.get("beta") or ot_row.get("beta") or 0))
        chromosomes.append(parsed["chromosome"] or ot_row.get("chromosome"))
        positions.append(parsed["position"] or ot_row.get("position"))
        ref_alleles.append(parsed["ref_allele"])
        minor_alleles.append(parsed["minor_allele"])
        rs_ids.append(None)
        ref_allele_freqs.append(float(lead_eaf) if lead_eaf is not None else None)

        lv = _log_pvalue(v.get("pValueMantissa"), v.get("pValueExponent"))
        log_pvalues.append(lv if lv is not None else lead_log_p)

    variants_data = {
        "variant": variants,
        "posterior_prob": posterior_probs,
        "beta": betas,
        "chromosome": chromosomes,
        "log_pvalue": log_pvalues,
        "position": positions,
        "ref_allele": ref_alleles,
        "minor_allele": minor_alleles,
        "ref_allele_freq": ref_allele_freqs,
        "rs_id": rs_ids,
    }

    return {
        "coverage": coverage,
        "variants": {"data": variants_data},
        "metadata": {
            "source": "opentargets",
            "study_locus_id": ot_row.get("study_locus_id"),
            "finemap_method": ot_row.get("finemap_method"),
            "confidence": ot_row.get("confidence"),
            "chr": ot_row.get("chromosome"),
            "position": ot_row.get("position"),
        },
    }


class CredibleSetsHandler:
    """Query OpenTargets credible sets from the dedicated PostgreSQL database."""

    def __init__(self):
        self._conn = None

    def _get_conn(self):
        if not _PSYCOPG2_OK:
            raise RuntimeError("psycopg2 is not installed")
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=os.environ.get("CREDIBLE_SETS_DB_HOST", "localhost"),
                port=int(os.environ.get("CREDIBLE_SETS_DB_PORT", "5411")),
                dbname=os.environ.get("CREDIBLE_SETS_DB_NAME", "credible_sets"),
                user=os.environ.get("CREDIBLE_SETS_DB_USER", "credsets"),
                password=os.environ.get("CREDIBLE_SETS_DB_PASSWORD", ""),
            )
        return self._conn

    def study_has_credible_sets(self, study_id: str) -> bool:
        """Fast check: does this GWAS study have credible sets in OpenTargets?"""
        if not study_id or not _PSYCOPG2_OK:
            return False
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM gwas_study_index WHERE study_id = %s LIMIT 1",
                    (study_id,),
                )
                return cur.fetchone() is not None
        except Exception as exc:
            logger.warning(f"[CredibleSets] study_has_credible_sets({study_id}): {exc}")
            return False

    def get_credible_sets_for_region(
        self,
        study_id: str,
        chromosome: str,
        position_start: int,
        position_end: int,
    ) -> list:
        """
        Fetch all credible sets for a study within a genomic window.
        Returns raw rows as dicts — pass each through convert_ot_row_to_credible_set()
        before saving.
        """
        if not study_id or not _PSYCOPG2_OK:
            return []
        try:
            conn = self._get_conn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT study_locus_id, study_id, variant_id,
                           chromosome, position,
                           beta, p_value_mantissa, p_value_exponent,
                           effect_allele_frequency, standard_error,
                           finemap_method, confidence, locus
                    FROM credible_sets
                    WHERE study_id    = %s
                      AND chromosome  = %s
                      AND position BETWEEN %s AND %s
                    ORDER BY position
                    """,
                    (study_id, str(chromosome), position_start, position_end),
                )
                return [dict(r) for r in cur.fetchall()]
        except Exception as exc:
            logger.warning(f"[CredibleSets] get_credible_sets_for_region({study_id}): {exc}")
            return []

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
