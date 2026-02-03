from collections import namedtuple
from typing import NamedTuple, List
import pickle
import gseapy as gp
from config import Config
from config import create_dependencies
from gene_expression_tasks import get_coexpression_matrix_for_tissue
import pandas as pd
from loguru import logger


class Enrich:

    def __init__(self, ensembl_hgnc_map_path, hgnc_ensembl_map_path,
                 go_map_path):

        self.ensembl_hgnc_map = pickle.load(open(ensembl_hgnc_map_path, "rb"))
        self.hgnc_ensembl_map = pickle.load(open(hgnc_ensembl_map_path, "rb"))
        self.go_map = pickle.load(open(go_map_path, "rb"))

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

    def get_coexpression_net(self, relevant_gene, tissue_name=None, cell_type=None, k=500, user_id=None, project_id=None):
        """
        Given a gene, tissue and cell_type, return the top correlated genes using CellxGene API.
        :param relevant_gene: Gene symbol (HGNC)
        :param tissue_name: Tissue name from user selection (GTEx format)
        :param cell_type: Cell type for more specificity (optional)
        :param k: Number of top correlated genes to return
        :param user_id: User ID for project-specific work directory (required for tissue-specific analysis)
        :param project_id: Project ID for project-specific work directory (required for tissue-specific analysis)
        :return: List of gene symbols (extracted from tuples like notebook)
        """
        if not tissue_name:
            # Fallback to hardcoded data if no tissue specified
            config = Config.from_env()
            brown_preadipocytes_top_corr_genes = pickle.load(open(f"{config.data_dir}/brown_preadipocytes_irx3_corr_top_500_genes.pkl", "rb"))
            return brown_preadipocytes_top_corr_genes
        
        if not user_id or not project_id:
            logger.warning(f"user_id and project_id required for tissue-specific analysis, falling back to hardcoded data")
            config = Config.from_env()
            brown_preadipocytes_top_corr_genes = pickle.load(open(f"{config.data_dir}/brown_preadipocytes_irx3_corr_top_500_genes.pkl", "rb"))
            return brown_preadipocytes_top_corr_genes
        
        try:
            # Get UBERON ID from database
            
            config = Config.from_env()
            deps = create_dependencies(config)
            gene_expression = deps['gene_expression']
            
            # Retrieve tissue mapping from database
            tissue_mapping = gene_expression.get_tissue_mapping(user_id, project_id, tissue_name)
            
            if not tissue_mapping:
                logger.warning(f"No tissue mapping found in database for '{tissue_name}', falling back to hardcoded data")
                brown_preadipocytes_top_corr_genes = pickle.load(open(f"{config.data_dir}/brown_preadipocytes_irx3_corr_top_500_genes.pkl", "rb"))
                return brown_preadipocytes_top_corr_genes
            
            # Extract UBERON ID from database record
            tissue_uberon_id = tissue_mapping.get('cellxgene_descendant_uberon_id') or tissue_mapping.get('cellxgene_parent_uberon_id')
            
            if not tissue_uberon_id:
                logger.warning(f"No UBERON ID found in mapping for '{tissue_name}', falling back to hardcoded data")
                brown_preadipocytes_top_corr_genes = pickle.load(open(f"{config.data_dir}/brown_preadipocytes_irx3_corr_top_500_genes.pkl", "rb"))
                return brown_preadipocytes_top_corr_genes
            
            logger.info(f"Using tissue mapping from database: {tissue_name} -> {tissue_uberon_id}")
            
            # Run tissue-specific coexpression analysis using UBERON ID
            top_positive_tuples, top_negative_tuples, all_genes = get_coexpression_matrix_for_tissue(
                relevant_gene, tissue_uberon_id, cell_type, k=k
            )
            
            # Extract gene symbols from tuples like notebook: [(gene_symbol, correlation), ...]
            if top_positive_tuples and isinstance(top_positive_tuples[0], tuple):
                # Format: [(gene_symbol, correlation), ...] -> [gene_symbol, ...]
                top_positive_genes = [gene_data[0] for gene_data in top_positive_tuples]
            else:
                # Fallback if format is different
                top_positive_genes = top_positive_tuples
            
            # Return both top genes and all genes for background
            return top_positive_genes, all_genes
            
        except Exception as e:
            print(f"Error running CellxGene coexpression analysis: {e}")
            # Fallback to hardcoded data
            config = Config.from_env()
            brown_preadipocytes_top_corr_genes = pickle.load(open(f"{config.data_dir}/brown_preadipocytes_irx3_corr_top_500_genes.pkl", "rb"))
            return brown_preadipocytes_top_corr_genes


    def run(self, relevant_gene, tissue_name=None, user_id=None, project_id=None):
        """
        Given a gene, return the enriched GO terms based on its co-expression network.
        Now supports tissue-specific analysis when tissue_name is provided.
        :param relevant_gene: Gene symbol
        :param tissue_name: Tissue name from LDSC analysis (optional)
        :param user_id: User ID for project-specific work directory (required for tissue-specific analysis)
        :param project_id: Project ID for project-specific work directory (required for tissue-specific analysis)
        """
        library = "GO_Biological_Process_2023"
        organism = "Human"

        config = Config.from_env()
        
        # Get coexpressed genes (now tissue-specific if tissue_name provided)
        coexpression_result = self.get_coexpression_net(relevant_gene, tissue_name, user_id=user_id, project_id=project_id)
        
        # Handle different return types (tuple for tissue-specific, list for fallback)
        if isinstance(coexpression_result, tuple):
            gene_list_ensembl, all_tissue_genes = coexpression_result
            # Convert Ensembl IDs to HGNC symbols for both gene list and background
            gene_list = self.get_hgnc_syms(gene_list_ensembl)
            
            # Use a reasonable background size (5000 genes) instead of all tissue genes
            # Large backgrounds (>40k) dilute enrichment signals and result in no significant terms
            max_background_size = 5000
            if len(all_tissue_genes) > max_background_size:
                logger.info(f"Limiting background from {len(all_tissue_genes)} to {max_background_size} genes for better enrichment signal")
                background_genes_ensembl = all_tissue_genes[:max_background_size]
            else:
                background_genes_ensembl = all_tissue_genes
            
            background_genes = self.get_hgnc_syms(background_genes_ensembl)
            logger.info(f"Running tissue-specific enrichment for {relevant_gene} in {tissue_name}")
            logger.info(f"Using tissue-specific background: {len(background_genes)} genes from CellxGene analysis")
            logger.info(f"Converted {len(gene_list_ensembl)} Ensembl IDs to {len(gene_list)} HGNC symbols")
        else:
            # Fallback case - no tissue specified or fallback data used (already HGNC symbols)
            gene_list = coexpression_result
            background_genes = pickle.load(open(f"{config.data_dir}/brown_preadipocytes_irx3_corr_background_genes.pkl", "rb"))
            logger.info(f"Running standard enrichment for {relevant_gene}")
        
        logger.info(f"Relevant Gene: {relevant_gene}")
        logger.info(f"Gene list sample: {gene_list[:5] if gene_list else []}")
        logger.info(f"Total coexpressed genes: {len(gene_list) if gene_list else 0}")
        
        if not gene_list:
            logger.warning("No coexpressed genes found, returning empty results")
            return pd.DataFrame(columns=["ID", "Term", "Desc", "Adjusted P-value", "Genes"])
        
        res = gp.enrichr(gene_list=gene_list,
                         gene_sets=library,
                         background=background_genes,
                         organism=organism,
                         outdir=None).results
        
        logger.debug(f"After enrichr API call: {len(res)} results")
        if len(res) > 0:
            logger.debug(f"Sample Term format: {res['Term'].iloc[0]}")
            logger.debug(f"Sample columns: {res.columns.tolist()}")
        else:
            logger.warning(f"Enrichr returned 0 results for gene {relevant_gene}. Check if gene symbols are valid HGNC symbols.")
            return pd.DataFrame(columns=["ID", "Term", "Desc", "Adjusted P-value", "Genes"])
        
        res.drop("Gene_set", axis=1, inplace=True)
        res.insert(1, "ID", res["Term"].apply(
            lambda x: x.split("(")[1].split(")")[0]))
        res["Term"] = res["Term"].apply(lambda x: x.split("(")[0])
        
        logger.debug(f"After parsing Term column: {len(res)} results")
        
        res = res[res["Adjusted P-value"] < 0.05]
        
        logger.debug(f"After p-value filter (< 0.05): {len(res)} results")
        
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
        res.drop(res.columns.difference(["ID", "Term", "Desc", "Adjusted P-value", "Genes"]), inplace=True, axis=1)
        
        logger.debug(f"Final enrichment results: {len(res)} GO terms")
        if len(res) > 0:
            logger.debug(f"Sample result - ID: {res['ID'].iloc[0]}, Term: {res['Term'].iloc[0]}, Desc: {res['Desc'].iloc[0][:50]}...")
        
        return res