import os
import sys
import json
from loguru import logger
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import create_dependencies, Config
from gwas_manifest_parser import GWASManifestParser
from dotenv import load_dotenv

def seed_gwas_library(gwas_handler, manifest_path):
    """Idempotent seeding of the GWAS manifest CSV"""
    if not manifest_path or not os.path.exists(manifest_path):
        logger.warning(f"GWAS Manifest not found at {manifest_path}. Skipping.")
        return

    # Check if empty
    if gwas_handler.get_entry_count() > 0:
        logger.info("GWAS library already seeded. Skipping.")
        return

    logger.info(f"Seeding GWAS library from {manifest_path}...")
    try:
        parser = GWASManifestParser(manifest_path)
        entries = parser.parse()
        valid_entries, _, _ = parser.validate_entries(entries)
        
        if valid_entries:
            result = gwas_handler.bulk_create_gwas_entries(valid_entries)
            logger.success(f"GWAS Seed: {result['inserted_count']} added, {result['skipped_count']} skipped.")
    except Exception as e:
        logger.error(f"GWAS Seed failed: {e}")

def seed_phenotypes(phenotype_handler, json_path):
    """Idempotent seeding of the Phenotype catalog JSON"""
    if not json_path or not os.path.exists(json_path):
        logger.warning(f"Phenotype catalog not found at {json_path}. Skipping.")
        return

    # Check if empty
    if phenotype_handler.count_phenotypes() > 0:
        logger.info("Phenotypes already seeded. Skipping.")
        return

    logger.info(f"Seeding Phenotypes from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
        
        # Map JSON keys ('name') to Handler keys ('phenotype_name')
        formatted_data = [
            {'id': item['id'], 'phenotype_name': item['name']} 
            for item in raw_data if 'id' in item and 'name' in item
        ]
        
        if formatted_data:
            result = phenotype_handler.bulk_create_phenotypes(formatted_data)
            logger.success(f"Phenotype Seed: {result['inserted_count']} added, {result['skipped_count']} skipped.")
    except Exception as e:
        logger.error(f"Phenotype Seed failed: {e}")

def main():
    load_dotenv()
    
    try:
        config = Config.from_env()
        deps = create_dependencies(config)
        
        logger.info("--- Starting Database Seeding ---")
        
        # Run GWAS Seeding
        seed_gwas_library(deps['gwas_library'], config.gwas_manifest_path)
        
        # Run Phenotype Seeding
        seed_phenotypes(deps['phenotypes'], config.phenotypes_catalog_path)
        
        logger.info("--- Seeding Check Complete ---")
        
    except Exception as e:
        logger.error(f"Critical error during seeding: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()