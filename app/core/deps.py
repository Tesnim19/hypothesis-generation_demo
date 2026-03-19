'''
dependecies for all docker services
for api we have additional one in api/
'''

from functools import lru_cache

from app.db import (
    UserHandler,
    ProjectHandler,
    FileHandler,
    AnalysisHandler,
    EnrichmentHandler,
    HypothesisHandler,
    SummaryHandler, 
    TaskHandler,
    GeneExpressionHandler,
    PhenotypeHandler,
    GWASLibraryHandler
)
from app.core.status_tracker import StatusTracker
from app.services import LLM, create_minio_client_from_env
from app.core.config import Settings, get_settings
from fastapi import Depends
from query_swipl import PrologQuery
from app.services.enrich import Enrich

# WE CAN SPLIY THIS TO 2 DEPS IF HEAVY TO PASS ALL TO THE ROUTERS, 
# ONE FOR DB AND ONE FOR SERVICES, BUT FOR NOW KEEPING IT SIMPLE
@lru_cache()
def create_dependencies(config: Settings) -> dict:
    """Factory function to create all dependencies from config"""    
    
    # Use environment variables for MongoDB connection
    mongodb_uri = config.mongodb_uri 
    db_name = config.db_name
    

    # Validate MongoDB configuration
    if not mongodb_uri or not db_name:
        raise ValueError("Missing required MongoDB configuration: MONGODB_URI and DB_NAME environment variables must be set")
    
    # Initialize MinIO storage
    minio_storage = create_minio_client_from_env()

    # Intialize status tracker with task handler dependency
    tasks_handler = TaskHandler(mongodb_uri, db_name)
    status_tracker_instance = StatusTracker()
    status_tracker_instance.initialize(tasks_handler)

    # Initialize enrichr service & gene expression handler
    gene_expression_handler = GeneExpressionHandler(mongodb_uri, db_name)
    enrichr = Enrich(
        config.ensembl_hgnc_map,
        config.hgnc_ensembl_map,
        config.go_map,
        config.data_dir,
        gene_expression_handler
    )
    
    llm = LLM()
    
    prolog_query = PrologQuery(
        host=config.swipl_host,
        port=config.swipl_port
    )
    return {
        'enrichr': enrichr,
        'llm': llm,
        'prolog_query': prolog_query,
        'users': UserHandler(mongodb_uri, db_name),
        'projects': ProjectHandler(mongodb_uri, db_name),
        'files': FileHandler(mongodb_uri, db_name),
        'analysis': AnalysisHandler(mongodb_uri, db_name),
        'enrichment': EnrichmentHandler(mongodb_uri, db_name),
        'hypotheses': HypothesisHandler(mongodb_uri, db_name),
        'summaries': SummaryHandler(mongodb_uri, db_name),
        'phenotypes': PhenotypeHandler(mongodb_uri, db_name),
        'gwas_library': GWASLibraryHandler(mongodb_uri, db_name),
        'gene_expression': gene_expression_handler,
        'tasks': tasks_handler,
        'storage': minio_storage,
        'status_tracker': status_tracker_instance
    }


def get_dependencies(config: Settings = Depends(get_settings)) -> dict:
    """for dependency injection in routes"""
    return create_dependencies(config)