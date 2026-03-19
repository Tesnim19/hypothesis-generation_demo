#!/usr/bin/env python

import os
import argparse
import logging
from prefect import serve
from app.workers.workflows.flows import (
    enrichment_flow,
    analysis_pipeline_flow,
    child_enrichment_batch_flow,
    # hypothesis_flow,  # Commented out - handled synchronously in POST /hypothesis
)
from app.core.config import get_settings
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_deployments(config):
    """Create and configure Prefect deployments"""
    
    # Set environment variables from config for the deployment
    os.environ.update({
        "ENSEMBL_HGNC_MAP": config.ensembl_hgnc_map,
        "HGNC_ENSEMBL_MAP": config.hgnc_ensembl_map,
        "GO_MAP": config.go_map,
        "SWIPL_HOST": config.swipl_host,
        "SWIPL_PORT": str(config.swipl_port),
    })
    
    # 1. Enrichment Deployment
    enrich_deploy = enrichment_flow.to_deployment(
        name="enrichment-flow-deployment",
        work_pool_name="interactive-pool",
        tags=["background_enrichment", "production"],
        description="Background enrichment processing for gene hypothesis generation",
        version="1.0.0"
    )

    # 2. Analysis Pipeline Deployment
    analysis_deploy = analysis_pipeline_flow.to_deployment(
        name="analysis-pipeline-deployment",
        work_pool_name="analysis-pool",
        tags=["analysis", "production"],
        description="GWAS Analysis Pipeline (Harmonization -> Fine-mapping)",
        version="1.0.0"
    )
    
    # 3. Child Enrichment Batch Deployment
    child_batch_deploy = child_enrichment_batch_flow.to_deployment(
        name="child-batch-deployment",
        work_pool_name="interactive-pool",
        tags=["background_hypothesis", "production"],
        description="Background processing for child enrichment hypotheses",
        version="1.0.0"
    )

    # 4. Hypothesis Generation Deployment
    # COMMENTED OUT: Hypothesis is now handled synchronously in POST /hypothesis
    # hypothesis_deploy = hypothesis_flow.to_deployment(...)
    
    return [enrich_deploy, analysis_deploy, child_batch_deploy]

def main():
    """Main deployment service entry point"""
    load_dotenv()
    
    # Try to get config from arguments first, fallback to environment
    config = get_settings() 
    logger.info("✅ Configuration loaded")
    
    # Validate critical configuration
    if not all([config.ensembl_hgnc_map, config.hgnc_ensembl_map, config.go_map]):
        raise ValueError("Missing required configuration: ensembl_hgnc_map, hgnc_ensembl_map, go_map")
    
    print(f" Starting Prefect deployment service...")
    logger.info(f"- SWIPL Host: {config.swipl_host}:{config.swipl_port}")
    logger.info(f"- Data files: {config.ensembl_hgnc_map}")
    
    deployments = setup_deployments(config)
    
    # Start serving deployments
    logger.info(f"Serving {len(deployments)} deployments...")
    serve(*deployments)

if __name__ == "__main__":
    main()