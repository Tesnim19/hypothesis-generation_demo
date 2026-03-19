'''
prev config.py separated into 2 here and depedencies
this file for getting config from env
and dependencies.py replaces the create_dependencies function 
'''

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict 

class Settings(BaseSettings):
    # map files(required)
    ensembl_hgnc_map : str = None
    hgnc_ensembl_map : str = None
    go_map : str = None

    # paths
    embedding_model : str = "w601sxs/b1ade-embed-kd"
    plink_dir_37 : str = "./data/1000Genomes_phase3/plink_format_b37"
    plink_dir_38 : str = "./data/1000Genomes_phase3/plink_format_b38"
    data_dir : str = "./data"
    ontology_cache_dir : str = "./data/ontology"

    # infrastructure
    mongodb_uri : str = None
    db_name : str = None

    # for internal communication
    host : str = "0.0.0.0"
    port : int = 5000

    # external services
    swipl_host : str = "localhost"
    swipl_port : int = 4242

    # Harmonization workflow configuration
    harmonizer_ref_dir_37 : str = "/app/data/harmonizer_ref/b37"
    harmonizer_ref_dir_38 : str = "/app/data/harmonizer_ref/b38"
    harmonizer_code_repo : str = "./gwas-sumstats-harmoniser"  # Nextflow workflow
    harmonizer_script_dir : str = "./scripts/1000Genomes_phase3"  # Shell scripts

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore", # ignore extra env vars
        frozen=True, # make settings immutable
    )

    def get_plink_dir(self, ref_genome):
        """Get the PLINK directory for the specified genome build"""
        if ref_genome == "GRCh38":
            return self.plink_dir_38
        elif ref_genome == "GRCh37":
            return self.plink_dir_37
        else:
            raise ValueError(f"Unsupported reference genome: {ref_genome}. Must be 'GRCh37' or 'GRCh38'")

    def get_harmonizer_ref_dir(self, ref_genome):
        """Get the harmonizer reference directory for the specified genome build"""
        if ref_genome == "GRCh38":
            return self.harmonizer_ref_dir_38
        elif ref_genome == "GRCh37":
            return self.harmonizer_ref_dir_37
        else:
            raise ValueError(f"Unsupported reference genome: {ref_genome}. Must be 'GRCh37' or 'GRCh38'")

    def get_plink_file_pattern(self, ref_genome, population, chrom):
        """
        Get the PLINK file pattern for the specified genome build.
        """
        if ref_genome == "GRCh38":
            return f"{population}.chr{chrom}.1KG.GRCh38"
        elif ref_genome == "GRCh37":
            return f"{population}.chr{chrom}.1KG.GRCh38"
            # return f"{population}.{chrom}.1000Gp3.20130502"
        else:
            raise ValueError(f"Unsupported reference genome: {ref_genome}. Must be 'GRCh37' or 'GRCh38'")
    
@lru_cache()
def get_settings():
    """Get settings instance"""
    return Settings()