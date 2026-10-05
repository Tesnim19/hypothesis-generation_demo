import copy
import json
import os
import pickle
import re
import threading
from typing import List, Optional

import gseapy as gp
import pandas as pd
from loguru import logger

from src.config import Config
from src.catlas_census_mapping import CatlasMappingError
from src.tasks.gene_expression import get_coexpression_matrix_for_tissue

_ENSG_RE = re.compile(r"^ENSG\d+$", re.IGNORECASE)


_SIGNIFICANCE_DEFAULT = object()


def significance_floor() -> float:
    """Max adjusted p-value for a GO term to count as a real enrichment hit.

    Single source of truth: the enrichment filter and the ranking layer's
    floor must agree, or lowering GO_SIGNIFICANCE_MAX_P silently tightens one
    and not the other.
    """
    return float(os.getenv("GO_SIGNIFICANCE_MAX_P", "0.05"))


_GO_ID_IN_TERM_RE = re.compile(r"\((GO:\d+)\)")
_GO_TERM_SIZES: dict[str, int] | None = None
_GO_TERM_SIZES_LOCK = threading.Lock()


def _parse_term_size(overlap: pd.Series) -> pd.Series:
    """Extract the GO term's total gene count from Enrichr's "k/n" Overlap."""
    return pd.to_numeric(
        overlap.astype(str).str.split("/").str[-1], errors="coerce"
    )


def go_term_sizes(library: str = "GO_Biological_Process_2023") -> dict[str, int]:
    """Map GO id -> number of genes annotated to it, from the Enrichr library.

    Enrichr's background-corrected endpoint (the one used whenever a custom
    background is supplied, i.e. always here) omits the Overlap column, so the
    term's size is not recoverable from the results themselves. Downstream
    ranking needs it as a specificity signal, so it is read once from the same
    gene-set library the enrichment was scored against and cached on disk.
    Failure is non-fatal: callers fall back to size-agnostic ranking.
    """
    global _GO_TERM_SIZES
    if _GO_TERM_SIZES is not None:
        return _GO_TERM_SIZES

    # Dask runs tasks on threads; without this two workers can both miss the
    # memo and fetch the library concurrently on first use.
    with _GO_TERM_SIZES_LOCK:
        if _GO_TERM_SIZES is not None:
            return _GO_TERM_SIZES
        _GO_TERM_SIZES = _load_go_term_sizes(library)
    return _GO_TERM_SIZES


def _load_go_term_sizes(library: str) -> dict[str, int]:
    cache_path = os.getenv("GO_TERM_SIZES_CACHE", "data/go_term_sizes.json")
    try:
        with open(cache_path) as fh:
            return {k: int(v) for k, v in json.load(fh).items()}
    except (OSError, ValueError, TypeError):
        pass

    try:
        sizes: dict[str, int] = {}
        for term, genes in gp.get_library(name=library, organism="Human").items():
            match = _GO_ID_IN_TERM_RE.search(term)
            if match:
                sizes[match.group(1)] = len(genes)
        if not sizes:
            raise ValueError(f"no GO ids parsed from library {library}")
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w") as fh:
                json.dump(sizes, fh)
        except OSError as exc:
            logger.warning(f"Could not cache GO term sizes to {cache_path}: {exc}")
        logger.info(f"Loaded {len(sizes)} GO term sizes from {library}")
    except Exception as exc:
        logger.warning(
            f"Could not load GO term sizes from {library}: {exc}. "
            "Ranking will fall back to size-agnostic scoring."
        )
        sizes = {}

    return sizes


class EnrichrAPIUnavailableError(RuntimeError):
    """Raised when all attempts to call the Enrichr API fail."""

    def __init__(self, message: str, *, variant: str | None = None) -> None:
        super().__init__(message)
        self.variant = variant

    def as_detail(self) -> dict[str, str]:
        detail = {"error_type": "enrichr_service_unavailable", "message": str(self)}
        if self.variant:
            detail["variant"] = self.variant
        return detail


class Enrich:

    def __init__(self, ensembl_hgnc_map_path, hgnc_ensembl_map_path,
                 go_map_path):

        with open(ensembl_hgnc_map_path, "rb") as f:
            self.ensembl_hgnc_map = pickle.load(f)
        with open(hgnc_ensembl_map_path, "rb") as f:
            self.hgnc_ensembl_map = pickle.load(f)
        with open(go_map_path, "rb") as f:
            self.go_map = pickle.load(f)
        
        self.config = Config.from_env()

    def _load_fallback_coexpression_data(self) -> List[str]:
        """Load hardcoded brown preadipocytes coexpression data as fallback."""
        fallback_path = f"{self.config.data_dir}/brown_preadipocytes_irx3_corr_top_500_genes.pkl"
        with open(fallback_path, "rb") as f:
            return pickle.load(f)
    
    def _load_fallback_background_data(self) -> List[str]:
        """Load hardcoded brown preadipocytes background genes as fallback."""
        fallback_path = f"{self.config.data_dir}/brown_preadipocytes_irx3_corr_background_genes.pkl"
        with open(fallback_path, "rb") as f:
            return pickle.load(f)
    
    @staticmethod
    def is_ensembl_id(gene: str) -> bool:
        return bool(gene and _ENSG_RE.match(str(gene).strip()))

    @staticmethod
    def _normalize_gene_token(gene: str) -> str:
        return str(gene).strip().strip("'\"")

    def to_symbol(self, gene: str) -> str:
        """Resolve an Ensembl ID or gene name to an HGNC symbol (uppercase)."""
        if not gene:
            return gene
        token = self._normalize_gene_token(gene)
        if self.is_ensembl_id(token):
            symbol = self.ensembl_hgnc_map.get(token.upper())
            if symbol:
                return symbol.upper()
            return token.upper()
        return token.upper()

    def to_ensembl_id(self, gene: str) -> Optional[str]:
        """Resolve a gene symbol or Ensembl ID to a lowercase Ensembl ID."""
        if not gene:
            return None
        token = self._normalize_gene_token(gene)
        if self.is_ensembl_id(token):
            return token.lower()
        ensembl_id = self.hgnc_ensembl_map.get(token.upper())
        return ensembl_id.lower() if ensembl_id else None

    def annotate_graph_gene_names(self, graph: dict) -> dict:
        """Return a graph copy with gene node names set to HGNC symbols (ids unchanged)."""
        resolved = copy.deepcopy(graph)
        for node in resolved.get("nodes", []):
            if node.get("type") != "gene":
                continue
            node["name"] = self.to_symbol(node.get("name") or node.get("id", ""))
        return resolved

    def get_hgnc_syms(self, ensg_ids):
        hgnc_symbols = []
        for g in ensg_ids:
            sym = self.ensembl_hgnc_map.get(g.upper(), None)
            if sym is not None:
                hgnc_symbols.append(sym)

        return hgnc_symbols

    def get_ensembl_ids(self, hgnc_syms):
        ensembl_ids = []
        for g in hgnc_syms:
            ensembl_id = self.hgnc_ensembl_map.get(g.upper(), None)
            if ensembl_id is not None:
                ensembl_ids.append(ensembl_id.lower())
            else:
                logger.warning(f"Couldn't find ensembl id for {g.upper()}")

        return ensembl_ids

    def get_coexpression_net(self, relevant_gene, tissue_name=None, k=500, coexpression_data=None):
        """
        Return top correlated genes for a gene using CellxGene.
        """
        if coexpression_data is not None:
            top_positive_tuples, top_negative_tuples, all_genes = coexpression_data
        elif not tissue_name:
            return self._load_fallback_coexpression_data()
        else:
            try:
                logger.info(f"[Enrich] Inline coexpression query for '{relevant_gene}' in '{tissue_name}'")
                top_positive_tuples, top_negative_tuples, all_genes = get_coexpression_matrix_for_tissue.fn(
                    relevant_gene, tissue_name, k=k
                )
            except CatlasMappingError:
                raise
            except Exception as e:
                logger.error(f"Error running CellxGene coexpression analysis: {e}")
                return self._load_fallback_coexpression_data()
        
        # Extract gene symbols from tuples
        if top_positive_tuples and isinstance(top_positive_tuples[0], tuple):
            top_positive_genes = [gene_data[0] for gene_data in top_positive_tuples]
        else:
            top_positive_genes = top_positive_tuples
        
        # Return both top genes and all genes for background
        return top_positive_genes, all_genes


    def _process_enrichment_results(
        self, res: pd.DataFrame, p_threshold=_SIGNIFICANCE_DEFAULT
    ) -> pd.DataFrame:
        """
        Process and filter enrichment results from gseapy.
        p_threshold=None keeps all rows returned by enrichr.
        """
        res = res.copy()  # Avoid SettingWithCopyWarning
        res.drop("Gene_set", axis=1, inplace=True)
        res.insert(1, "ID", res["Term"].apply(
            lambda x: x.split("(")[1].split(")")[0]))
        res["Term"] = res["Term"].apply(lambda x: x.split("(")[0])
        if p_threshold is _SIGNIFICANCE_DEFAULT:
            p_threshold = significance_floor()
        if p_threshold is not None:
            res = res[res["Adjusted P-value"] < p_threshold].copy()
        desc = []
        for _, row in res.iterrows():
            go_id = row["ID"]
            go_name = row["Term"]
            try:
                go_desc = self.go_map[go_id]["desc"]
                desc.append(go_desc)
            except KeyError:
                logger.warning(f"Couldn't find term {go_id}, {go_name} in go_map")
                desc.append("NA")
        res["Desc"] = desc

        if "Overlap" in res.columns:
            res["Term Size"] = _parse_term_size(res["Overlap"])
        else:
            sizes = go_term_sizes()
            res["Term Size"] = (
                res["ID"].map(sizes).astype("Float64") if sizes else pd.NA
            )
            # Enrichr's server-side library is not guaranteed to match the
            # published one we read sizes from, so a term can be scored here
            # and absent there. Unmatched terms fall back to neutral median
            # specificity, which is safe but silent -- warn if it stops being
            # a handful of terms.
            if sizes is not None and len(res):
                matched = int(pd.to_numeric(res["Term Size"], errors="coerce").notna().sum())
                if matched < len(res) * 0.9:
                    logger.warning(
                        f"GO term sizes matched only {matched}/{len(res)} enriched "
                        f"terms; specificity weighting is degraded. The Enrichr "
                        f"library version may have diverged from the cached sizes."
                    )
        return res[
            ["ID", "Term", "Desc", "Adjusted P-value", "Genes", "Term Size"]
        ].copy()

    def _run_enrichr_with_retry(
        self,
        *,
        gene_list,
        gene_sets,
        background,
        organism,
        outdir=None,
    ) -> pd.DataFrame:
        """Call Enrichr with progressively smaller backgrounds."""
        background_sizes = []
        for limit in (5000, 2500, 1000):
            size = min(len(background), limit)
            if size not in background_sizes:
                background_sizes.append(size)

        last_error = None
        for background_size in background_sizes:
            try:
                logger.info(
                    f"Calling Enrichr with background size {background_size}"
                )
                return gp.enrichr(
                    gene_list=gene_list,
                    gene_sets=gene_sets,
                    background=background[:background_size],
                    organism=organism,
                    outdir=outdir,
                ).results
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"Enrichr call failed with background size {background_size}: {exc}"
                )

        raise EnrichrAPIUnavailableError(
            "Enrichr API remained unavailable after retrying with background sizes "
            f"{background_sizes}"
        ) from last_error

    def _run_enrichr(self, relevant_gene, tissue_name=None, coexpression_data=None):
        library = "GO_Biological_Process_2023"
        organism = "human"
        causal_gene_symbol = self.to_symbol(relevant_gene)
        ensembl_gene = self.to_ensembl_id(relevant_gene)
        if ensembl_gene is None:
            ensembl_gene = relevant_gene
            logger.warning(
                f"Could not map '{relevant_gene}' to Ensembl ID; "
                "coexpression queries may fail"
            )

        coexpression_result = self.get_coexpression_net(
            ensembl_gene, tissue_name, coexpression_data=coexpression_data
        )

        if isinstance(coexpression_result, tuple):
            gene_list_ensembl, all_tissue_genes = coexpression_result
            gene_list = self.get_hgnc_syms(gene_list_ensembl)
            max_background_size = 5000
            if len(all_tissue_genes) > max_background_size:
                logger.info(
                    f"Limiting background from {len(all_tissue_genes)} to "
                    f"{max_background_size} genes for better enrichment signal"
                )
                background_genes_ensembl = all_tissue_genes[:max_background_size]
            else:
                background_genes_ensembl = all_tissue_genes
            background_genes = self.get_hgnc_syms(background_genes_ensembl)
            logger.info(
                f"Running tissue-specific enrichment for {causal_gene_symbol} "
                f"in {tissue_name}"
            )
            logger.info(
                f"Using tissue-specific background: {len(background_genes)} genes "
                "from CellxGene analysis"
            )
            logger.info(
                f"Converted {len(gene_list_ensembl)} Ensembl IDs to "
                f"{len(gene_list)} HGNC symbols"
            )
        else:
            gene_list = coexpression_result
            background_genes = self._load_fallback_background_data()
            logger.info(f"Running standard enrichment for {causal_gene_symbol}")

        logger.info(f"Relevant Gene: {causal_gene_symbol}")
        logger.info(f"Gene list sample: {gene_list[:5] if gene_list else []}")
        logger.info(f"Total coexpressed genes: {len(gene_list) if gene_list else 0}")

        if not gene_list:
            logger.warning("No coexpressed genes found, returning empty results")
            return None

        return self._run_enrichr_with_retry(
            gene_list=gene_list,
            gene_sets=library,
            background=background_genes,
            organism=organism,
            outdir=None,
        )

    def run(self, relevant_gene, tissue_name=None, coexpression_data=None):
        """
        Given a gene, return the enriched GO terms based on its co-expression network.
        If coexpression_data is provided (from Dask task), use it instead of computing.
        """
        raw = self._run_enrichr(relevant_gene, tissue_name, coexpression_data)
        if raw is None:
            return pd.DataFrame(columns=["ID", "Term", "Desc", "Adjusted P-value", "Genes", "Term Size"])
        return self._process_enrichment_results(raw)

    def run_with_tables(self, relevant_gene, tissue_name=None, coexpression_data=None):
        """
        Like run(), but also returns the full parsed enrichr table (no p-value filter).
        """
        empty = pd.DataFrame(columns=["ID", "Term", "Desc", "Adjusted P-value", "Genes", "Term Size"])
        raw = self._run_enrichr(relevant_gene, tissue_name, coexpression_data)
        if raw is None:
            return empty, empty
        filtered = self._process_enrichment_results(raw)
        all_terms = self._process_enrichment_results(raw, p_threshold=None)
        logger.info(
            f"Enrichr returned {len(all_terms)} GO terms; "
            f"{len(filtered)} pass adj p < 0.05"
        )
        return filtered, all_terms
