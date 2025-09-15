import os
import argparse
from llm import LLM
from query_swipl import PrologQuery
from db import (
    UserHandler, ProjectHandler, FileHandler, AnalysisHandler,
    EnrichmentHandler, HypothesisHandler, SummaryHandler, TaskHandler
)

class Config:
    """Centralized configuration for the application"""
    
    def __init__(self):
        self.ensembl_hgnc_map = None
        self.hgnc_ensembl_map = None
        self.go_map = None
        self.swipl_host = "localhost"
        self.swipl_port = 4242
        self.mongodb_uri = None
        self.db_name = None
        self.embedding_model = "w601sxs/b1ade-embed-kd"
        self.plink_dir = "./data/1000Genomes_phase3/plink_format_b37"
        self.data_dir = "./data"
        self.host = "0.0.0.0"
        self.port = 5000
        
        # LLM Configuration
        self.biomedical_llm = "Qwen/Qwen3-4B-Instruct-2507"
        self.jina_embedding_model = "jinaai/jina-embeddings-v3"
        self.llm_temperature = 0.0
        self.llm_max_tokens = 300
        self.llm_attention_dropout = 0.2
        self.llm_hidden_dropout = 0.2
        
        # MC Dropout Configuration
        self.mc_dropout_samples = 5
        self.mc_dropout_max_predictions = 10
        
        # Literature Mining Configuration
        self.use_literature_mining = True
        self.pubmed_max_results = 10
        self.pubmed_rate_limit_delay = 2.0
        
        # FAISS/Chunking Configuration
        self.faiss_k_nearest = 2
        self.chunk_max_size = 500
        self.chunk_total_limit = 5000
        self.min_chunk_size = 500

    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments"""
        config = cls()
        config.ensembl_hgnc_map = args.ensembl_hgnc_map
        config.hgnc_ensembl_map = args.hgnc_ensembl_map
        config.go_map = args.go_map
        config.swipl_host = args.swipl_host
        config.swipl_port = args.swipl_port
        config.embedding_model = getattr(args, 'embedding_model', config.embedding_model)
        # Flask-specific arguments (if present)
        config.host = getattr(args, 'host', config.host)
        config.port = getattr(args, 'port', config.port)
        # Also load MongoDB config from environment
        config.mongodb_uri = os.getenv("MONGODB_URI")
        config.db_name = os.getenv("DB_NAME")
        return config

    @classmethod
    def from_env(cls):
        """Create config from environment variables"""
        config = cls()
        config.ensembl_hgnc_map = os.getenv("ENSEMBL_HGNC_MAP")
        config.hgnc_ensembl_map = os.getenv("HGNC_ENSEMBL_MAP")
        config.go_map = os.getenv("GO_MAP")
        config.swipl_host = os.getenv("SWIPL_HOST", "localhost")
        config.swipl_port = int(os.getenv("SWIPL_PORT", "4242"))
        config.mongodb_uri = os.getenv("MONGODB_URI")
        config.db_name = os.getenv("DB_NAME")
        config.embedding_model = os.getenv("EMBEDDING_MODEL", "w601sxs/b1ade-embed-kd")
        config.plink_dir = os.getenv("PLINK_DIR", "./data/1000Genomes_phase3/plink_format_b37")
        config.data_dir = os.getenv("DATA_DIR", "./data")
        
        # LLM Configuration
        config.biomedical_llm = os.getenv("BIOMEDICAL_LLM", "Qwen/Qwen3-4B-Instruct-2507")
        config.jina_embedding_model = os.getenv("JINA_EMBEDDING_MODEL", "jinaai/jina-embeddings-v3")
        config.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        config.llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "300"))
        config.llm_attention_dropout = float(os.getenv("LLM_ATTENTION_DROPOUT", "0.2"))
        config.llm_hidden_dropout = float(os.getenv("LLM_HIDDEN_DROPOUT", "0.2"))
        
        # MC Dropout Configuration
        config.mc_dropout_samples = int(os.getenv("MC_DROPOUT_SAMPLES", "5"))
        config.mc_dropout_max_predictions = int(os.getenv("MC_DROPOUT_MAX_PREDICTIONS", "10"))
        
        # Literature Mining Configuration
        config.use_literature_mining = os.getenv("USE_LITERATURE_MINING", "true").lower() == "true"
        config.pubmed_max_results = int(os.getenv("PUBMED_MAX_RESULTS", "10"))
        config.pubmed_rate_limit_delay = float(os.getenv("PUBMED_RATE_LIMIT_DELAY", "2.0"))
        
        # FAISS/Chunking Configuration
        config.faiss_k_nearest = int(os.getenv("FAISS_K_NEAREST", "2"))
        config.chunk_max_size = int(os.getenv("CHUNK_MAX_SIZE", "500"))
        config.chunk_total_limit = int(os.getenv("CHUNK_TOTAL_LIMIT", "5000"))
        config.min_chunk_size = int(os.getenv("MIN_CHUNK_SIZE", "500"))
        
        return config


def create_dependencies(config):
    """Factory function to create all dependencies from config"""
    # Import here to avoid circular dependency
    from enrich import Enrich
    
    enrichr = Enrich(
        config.ensembl_hgnc_map,
        config.hgnc_ensembl_map,
        config.go_map
    )
    
    llm = LLM(config=config)
    
    prolog_query = PrologQuery(
        host=config.swipl_host,
        port=config.swipl_port
    )
    
    # Use environment variables for MongoDB connection
    mongodb_uri = config.mongodb_uri 
    db_name = config.db_name
    
    # Validate MongoDB configuration
    if not mongodb_uri or not db_name:
        raise ValueError("Missing required MongoDB configuration: MONGODB_URI and DB_NAME environment variables must be set")
    
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
        'tasks': TaskHandler(mongodb_uri, db_name)
    }
