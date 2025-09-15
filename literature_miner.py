import requests
import json
import time
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from loguru import logger
import re
import numpy as np
from parser import LiteratureParser
import math
import faiss

class LiteratureMiner:
    def __init__(self, config=None):
        """Initialize the literature miner with embedding model"""
        
        if config:
            embedding_model = config.jina_embedding_model
            self.pubmed_max_results = config.pubmed_max_results
            self.rate_limit_delay = config.pubmed_rate_limit_delay
            self.k_nearest = config.faiss_k_nearest
            self.chunk_max_size = config.chunk_max_size
            self.chunk_total_limit = config.chunk_total_limit
            self.min_chunk_size = config.min_chunk_size
            self.use_literature_mining = config.use_literature_mining
        else:
            # Fallback to defaults for backward compatibility
            embedding_model = "jinaai/jina-embeddings-v3"
            self.pubmed_max_results = 10
            self.rate_limit_delay = 2.0
            self.k_nearest = 2
            self.chunk_max_size = 500
            self.chunk_total_limit = 5000
            self.min_chunk_size = 500
            self.use_literature_mining = True

        self.embedder = SentenceTransformer(embedding_model, trust_remote_code=True)
        self.pubmed_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.phenotype_variations_cache = {}  # Cache for phenotype variations
        self.relevant_chunks = []
        logger.info("LiteratureMiner initialized successfully")
        self.parser = LiteratureParser()   
        self.index = None
        self.num_candidate_genes = 0 

    def search_pubmed(self, phenotype, gene, max_results=None):
        """
        Search PubMed for phenotype-gene associations with improved specificity
        Returns: Dict with count and relevant abstracts
        """
        if max_results is None:
            max_results = self.pubmed_max_results
            
        try:
            # More specific search query with gene name and phenotype in title/abstract
            search_url = f"{self.pubmed_base_url}/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": f'"{gene}"[Gene Name] AND "{phenotype}"[Title/Abstract] AND (causal OR mechanism OR association OR function)',
                "retmode": "json",
                "retmax": max_results,
                "sort": "relevance",
                "field": "Title/Abstract",
            }

            response = requests.get(search_url, params=search_params, timeout=10)
            response.raise_for_status()
            search_data = response.json()

            article_ids = search_data["esearchresult"]["idlist"]
            count = len(article_ids)

            if count == 0:
                result = {"count": 0, "abstracts": []}
                return result

            pubmed_articles = self._fetch_pubmed_abstracts(article_ids[:5])  
            relevant_sections = [section["text"] for article in pubmed_articles for section in article["relevant_sections"]]

            result = {
                "count": len(pubmed_articles),
                "relevant_text": relevant_sections,
            }
            
            return result

        except Exception as e:
            logger.error(f"PubMed search failed for {gene}: {str(e)}")
            result = {"count": 0, "abstracts": []}
            return result

    def _fetch_pubmed_abstracts(self, article_ids):
        """Fetch abstracts for given PubMed article IDs"""
        if not article_ids:
            return []

        try:
            # Fetch article details
            fetch_url = f"{self.pubmed_base_url}/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(article_ids),
                "retmode": "xml",
                "rettype": "abstract",
            }

            response = requests.get(fetch_url, params=fetch_params, timeout=15)
            response.raise_for_status()
            articles = self.parser.parse_pubmed_xml(response.text)
            return articles[:5]

        except Exception as e:
            logger.error(f"Failed to fetch PubMed abstracts: {str(e)}")
            return []
    
    def recursive_chunker(self, relevant_texts, gene):
        max_size = (self.chunk_total_limit // self.num_candidate_genes) if self.num_candidate_genes else self.chunk_max_size
        if max_size < self.min_chunk_size: 
            self.k_nearest = 1  
        max_size = max(max_size, self.min_chunk_size)
        
        if not relevant_texts:
            return 
            
        for i in range(len(relevant_texts)):
            text = relevant_texts[i]
            if len(text) <= max_size:
                text = gene + " : " + text
                if i > 0 and len(self.relevant_chunks) > 0:
                    # overlap by last sntence
                    prev_sentences = self.relevant_chunks[-1].split(".")
                    if len(prev_sentences) > 1:
                        text = prev_sentences[-1] + text
                self.relevant_chunks.append(text)
            else:
                partition = math.ceil(len(text) / max_size)
                gap = len(text) // partition
                start, end = 0, gap 
                for j in range(partition):
                    if j > 0 and len(self.relevant_chunks) > 0:
                        prev_sentences = self.relevant_chunks[-1].split(".")
                        if len(prev_sentences) > 1:
                            prev_text = prev_sentences[-1]
                            end -= len(prev_text)
                            chunk_text = gene + " : " + prev_text + text[start:end]
                        else:
                            chunk_text = gene + " : " + text[start:end].strip()
                    else:
                        chunk_text = gene + " : " + text[start:end].strip()

                    if chunk_text:
                        self.relevant_chunks.append(chunk_text)
                    start = end 
                    end += gap  
                    if start >= len(text):
                        break


    def extract_key_findings(self, phenotype, gene, llm=None):
        """
        Extract key findings from abstracts using improved semantic similarity
        """
        if not self.relevant_chunks:
            return []

        try:
            # Create semantic queries for better relevance detection
            query = f"{gene} {phenotype} association mechanism function"
            q_vec = self.embedder.encode([query])  
            q_vec = q_vec.astype('float32')
            faiss.normalize_L2(q_vec)
            
            # retrieve 2 nearest neighbor chunk indices for each gene
            D, I = self.index.search(q_vec, k=self.k_nearest)  
            return {"Chunk_indices" : I[0], "Distance" : D[0]}

        except Exception as e:
            logger.error(f"Failed to extract key findings: {str(e)}")
            return []

    def get_literature_evidence(self, phenotype, candidate_genes, llm=None):
        """
        Get comprehensive literature evidence for all candidate genes using PubMed
        """
        logger.info(f"=== GET_LITERATURE_EVIDENCE START ===")
        logger.info(f"Input - phenotype: '{phenotype}'")
        logger.info(f"Input - candidate_genes: {candidate_genes}")
        self.num_candidate_genes = len(candidate_genes)
        evidence = []
        
        for gene_idx, gene in enumerate(candidate_genes):
            logger.info(f"Processing gene {gene_idx + 1}/{len(candidate_genes)}: {gene}")

            # Reset chunks for each gene
            self.relevant_chunks = []
            
            # Search PubMed 
            pubmed_results = self.search_pubmed(phenotype, gene)
            if "relevant_text" in pubmed_results:
                self.recursive_chunker(pubmed_results["relevant_text"], gene)
            
            num_chunks = len(self.relevant_chunks)
            print(f"Gene {gene}: collected {num_chunks} chunks")
            
            if num_chunks == 0:
                logger.warning(f"No chunks found for gene {gene}")
                time.sleep(self.rate_limit_delay)
                continue
            
            time.sleep(self.rate_limit_delay)

            # Extract key findings with improved method
            chunk_embeddings = self.embedder.encode(self.relevant_chunks)

            # Create fresh FAISS index for this gene
            embeddings = np.array(chunk_embeddings).astype('float32')
            d = embeddings.shape[1]
            # exact search index
            self.index = faiss.IndexFlatIP(d)
            faiss.normalize_L2(embeddings)  # normalize in-place for cosine similarity
            self.index.add(embeddings)
            print(f"Indexed {self.index.ntotal} vectors of dimension {d} for gene {gene}.")

            key_findings = self.extract_key_findings(phenotype, gene, llm)

            chunk_idx_to_distance = [[self.relevant_chunks[x], y] for x, y in zip(key_findings["Chunk_indices"], key_findings["Distance"])]
            evidence.extend(chunk_idx_to_distance)

         
        # print(evidence)
        print(len("".join([x[0] for x in evidence])))
        return evidence

    def format_evidence_for_llm(self, evidences) -> str:
        """
        Format literature evidence for LLM prompt with improved structure
        """
        logger.info(f"=== FORMAT_EVIDENCE_FOR_LLM START ===")

        if not evidences:
            logger.warning("No evidence to format")
            return "Literature Evidence:\nNo relevant literature found."
        
        evidence_text = "Literature Evidence:\n" + " ".join([x[0] for x in evidences])
        print("length of Evidence Text:", len(evidence_text))

        logger.info(f"=== FORMAT_EVIDENCE_FOR_LLM END ===")
        return evidence_text