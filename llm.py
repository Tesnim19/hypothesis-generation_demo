import json
import os
import re
import scipy
import torch
import scipy.spatial
import numpy as np
from typing import List, Optional, Dict, Union
import time
from collections import defaultdict, Counter
from loguru import logger
from pydantic import BaseModel
from scipy.spatial.distance import cosine
from literature_miner import LiteratureMiner
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]

def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
                texts.append(passage)
        elif title is not None:
             titles.append(title)
             texts.append("")
    return {"title": titles, "text": texts}


class GoTerm(BaseModel):
    rank: int
    name: str
    reason: str


class Response(BaseModel):
    terms: List[GoTerm]

class LLM():

    def __init__(self, config=None, llm=None, temperature=None):
        super().__init__()
        
        # Use config if provided, otherwise use defaults or parameters
        if config:
            model_name = config.biomedical_llm
            embedding_model = config.jina_embedding_model
            self.temperature = config.llm_temperature
            self.max_tokens = config.llm_max_tokens
            self.mc_dropout_samples = config.mc_dropout_samples
            self.mc_dropout_max_predictions = config.mc_dropout_max_predictions
            attention_dropout = config.llm_attention_dropout
            hidden_dropout = config.llm_hidden_dropout
        else:
            # Fallback to parameters or defaults for backward compatibility
            model_name = llm if llm else "Qwen/Qwen3-4B-Instruct-2507"
            embedding_model = "jinaai/jina-embeddings-v3"
            self.temperature = temperature if temperature is not None else 0.0
            self.max_tokens = 300
            self.mc_dropout_samples = 5
            self.mc_dropout_max_predictions = 10
            attention_dropout = 0.2
            hidden_dropout = 0.2
        
        self.embedder = SentenceTransformer(embedding_model, trust_remote_code=True)
        self.llm_type = model_name
        self.literature_miner = LiteratureMiner(config=config)
        
        # Load model config & instantiate model
        model_config = Qwen3Config.from_pretrained(model_name)
        model_config.attention_dropout = attention_dropout
        model_config.hidden_dropout = hidden_dropout

        self.llm = AutoModelForCausalLM.from_pretrained(model_name, config=model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Predictive confidence and entropy
    def predictive_confidence(self, genes, reasons):
        if not genes:
            return {}
        
        counts = Counter(genes)
        total = sum(counts.values())
        final_gene, max_count = counts.most_common(1)[0]
        
        confidence = (max_count / total) * 100
        
        # Predictive entropy (normalized by number of MC samples)
        probs = np.array([c / total for c in counts.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(total)
        norm_entropy = (entropy / max_entropy) * 100 if max_entropy > 0 else 0
        
        # Vote margin
        if len(counts) > 1:
            _, second_count = counts.most_common(2)[1]
            vote_margin = ((max_count - second_count) / total) * 100
        else:
            vote_margin = 100.0
        
        return {
            "Predicted causal gene": final_gene,
            "Confidence": f"{confidence:.2f}%",
            "Predictive entropy": f"{norm_entropy:.2f}%",
            "Vote margin": f"{vote_margin:.2f}%",
            "Predictions": genes,
            "Reasons for predicted gene": reasons[final_gene][0]
        }

    def mc_dropout(self, messages, num_samples=None): 
        if num_samples is None:
            num_samples = self.mc_dropout_samples
            
        self.llm.train()
        predictions = []
        reasons = defaultdict(list)

        for _ in range(num_samples):
           logger.debug(f"MC Dropout iteration {_}/{num_samples} started")
           parsed = self.get_causal_gene_with_reason(messages)
           if parsed and "causal_gene" in parsed and "reason" in parsed:
               predictions.append(parsed["causal_gene"].upper())
               reasons[parsed["causal_gene"].upper()].append(parsed["reason"])
           else:
               logger.error(f"No valid reason generated, Iteration Passed")

           if len(predictions) >= self.mc_dropout_max_predictions:
               break

        self.llm.eval()
        return self.predictive_confidence(predictions, reasons)
      
    
    def _chat(self, messages: List[dict]) -> Optional[Union[Dict, List]]:
        """
        Internal method to send messages to LLM and parse JSON response.
        Returns parsed JSON dict/list or None if parsing fails.
        """
        with torch.no_grad():
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)

            logger.debug("Sending messages to LLM for generation")
            gen_ids = self.llm.generate(**model_inputs, max_new_tokens=self.max_tokens, do_sample=False)
            output_ids = gen_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            gen_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            logger.debug(f"LLM generated text: {gen_text[:200]}..." if len(gen_text) > 200 else f"LLM generated text: {gen_text}")
            
            match = re.search(r"(\{.*?\}|\[.*?\])", gen_text, re.DOTALL)
            if match:
                json_text = match.group(0)
                try:
                    parsed = json.loads(json_text)
                except json.JSONDecodeError:
                    parsed = None
            else:
                parsed = None

            return parsed

    def get_causal_gene_with_reason(self, messages):
        gen_text = self._chat(messages)
        return gen_text
        
    def predict_casual_gene(self, phenotype, genes, prev_gene=None, rule=None, use_literature=None):
        """
        Given a variant, a list of candidate genes and a phenotype, query the LLM to predict the causal gene
        """
        if use_literature is None:
            use_literature = hasattr(self.literature_miner, 'use_literature_mining') and self.literature_miner.use_literature_mining
            
        genes = sorted(genes)
        genes_fmt = []
        for gene in genes:
            genes_fmt.append("{" + gene + "}")

        genes_str = ",".join(genes_fmt)
        literature_evidence = ""
        if use_literature:
            try:
                logger.info(f"Searching literature for {phenotype} and {len(genes)} candidate genes")
                evidence_list = self.literature_miner.get_literature_evidence(phenotype, genes, self)
                literature_evidence = self.literature_miner.format_evidence_for_llm(evidence_list)
                logger.info(f"Found {len(evidence_list)} literature evidence chunks")
            except Exception as e:
                logger.warning(f"Literature mining failed: {str(e)}. Continuing without literature evidence.")
                literature_evidence = ""

        
        constraint_text = ""
        if rule:
            assert prev_gene is not None, "A previous gene must be provided when a rule is given."
            constraint_text = (
                f"\n\nIMPORTANT CONSTRAINT: Your previous prediction of '{prev_gene}' was invalid due to the rule: "
                f"'{rule}'. Your new prediction MUST be consistent with this rule.")

        system_prompt = """
You are a specialized genetics research assistant AI. 
Your task is to determine the most likely causal gene for a given phenotype from a list of candidates, based on provided literature. 

Follow these rules:
- Analyze the phenotype, candidate list, and literature.
- Identify the single best candidate.
- Write a justification of at least 50 words, using evidence only from the literature.
- Output ONLY a valid JSON object, no markdown, no explanations, nothing else.

Format:
{
  "causal_gene": "GENE_SYMBOL",
  "reason": "50 word explanation citing the evidence"
}
"""

        response = ""

        messages = [
            {
                "role": "system",
                "content":system_prompt,
            },
            {
                "role": "user",
                "content": f"""Phenotype: {phenotype}
        Candidate genes: {",".join(genes)}
        Literature Evidence:
        {literature_evidence}{constraint_text}""",
            },
        ]
        
        logger.debug(f"Predicting causal gene for phenotype '{phenotype}' with {len(genes)} candidates")
        logger.debug(f"Literature evidence length: {len(literature_evidence)} chars, Rule constraint: {bool(rule)}")
        response = self.mc_dropout(messages)
        return response

    def get_phenotype_variations(self, phenotype):
        """
        Get phenotype variations using LLM for intelligent expansion
        """
        try:
            # Create prompt for phenotype variations
            prompt = f"""
            Given the phenotype "{phenotype}", generate a JSON list of common variations, synonyms, and related terms that might appear in scientific literature.

            Include:
            1. Common medical synonyms
            2. Abbreviations used in literature
            3. Related medical terms
            4. Different spellings or forms

            Return ONLY a valid JSON array of strings, no explanations.

            Example for "diabetes":
            ["diabetic", "diabetes mellitus", "DM", "T1D", "T2D", "glucose metabolism disorder"]

            For "{phenotype}":
            """

            response = ""

            system_prompt = "You are a medical terminology expert. Return only valid JSON arrays."
            messages = [
            {
                "role": "system",
                "content":system_prompt,
            },
            {
                "role": "user",
                "content":prompt
            }]

            time.sleep(1)
            response = self._chat(messages)  # Returns parsed dict or None

            if response and isinstance(response, list):
                logger.info(f"LLM generated {len(response)} phenotype variations for '{phenotype}': {response}")
                return response
            else:
                logger.warning(f"Invalid response from LLM for phenotype '{phenotype}': {response}")
                # Fallback to basic variations
                return self._get_basic_phenotype_variations(phenotype)

        except Exception as e:
            logger.error(f"Failed to get LLM phenotype variations for '{phenotype}': {str(e)}")
            # Fallback to basic variations
            return self._get_basic_phenotype_variations(phenotype)

    def _get_basic_phenotype_variations(self, phenotype: str) -> List[str]:
        """
        Fallback method to provide basic phenotype variations when LLM fails
        """
        # Return basic variations: original, lowercase, and capitalized
        return [phenotype, phenotype.lower(), phenotype.capitalize()]

    def get_relevant_go(self, phentoype, enrich_tbl, k=10):
        """
        Given a phenotype, a sequence variant and an enrichment analysis table, get the top k relevant GO terms relevant to the phenotype by prompting the LLM using RAG
        :param phentoype: GWAS Phenotype/Trait
        :param variant: Sequence Variant
        :param enrich_tbl: Table containing the over-presentation test expected columns are ID, Term, Desc, Adjusted P-val
        :return: dict obj containing the k relevant GO terms, their p-val and the reason why the LLM thinks they are relevant to the phenotype
        """
        # tmp_file = tempfile.NamedTemporaryFile("w+")
        # df.drop(columns=["ID", "Adjusted P-value", "Genes"], inplace=False)
        # df.to_csv(tmp_file, index=False)

        # Embed the GO terms and their descriptions.
        res = self._retrieve_top_k_go_terms(phentoype, enrich_tbl, k)

        return res

    def _embed_dataset(self, batch):
        combined_text = []
        for title, text in zip(batch["title"], batch["text"]):
            combined_text.append(" [SEP] ".join([title, text]))

        return {"embeddings": self.embedder.encode(combined_text)}

    def _retrieve_top_k_go_terms(self, query, data, k):

        texts = [
            f"{row['Term'].strip()} [SEP] {row['Desc'].strip()}"
            for _, row in data.iterrows()
        ]

        embeddings = self.embedder.encode(texts, convert_to_tensor=False)
        query_embedding = self.embedder.encode(query, convert_to_tensor=False)

        similarities = [1 - cosine(e, query_embedding) for e in embeddings]

        data["similarity"] = similarities

        res = data.sort_values("similarity", ascending=False).head(k)

        # Format result
        subset_go = []
        for i, (_, row) in enumerate(res.iterrows(), start=1):
            go_entry = {
                "id": row["ID"].strip(),
                "name": row["Term"].strip(),
                "genes": row["Genes"].split(";"),
                "p": row["Adjusted P-value"],
                "rank": i,
            }
            subset_go.append(go_entry)
        logger.debug(subset_go)
        return subset_go

    def get_structured_response(self, response, enrich_table):
        """
        Use outlines to generate a structured response to a prompt
        :param prompt: Prompt to use
        :param enrich_table: Enrichment table
        :return:
        """
        # model = models.openai("gpt-4-0163", api_key=openai.api_key)
        # generator = outlines.generate.json(model, Response)
        # # rng = torch.Generator(device="cuda")
        # # rng.manual_seed(42)
        # response = generator(prompt)
        subset_go = {
            "ID": [],
            "Name": [],
            "Rank": [],
            "Reason": [],
            "Genes": [],
            "Adjusted P-value": [],
        }

        for res in response.terms:
            row = enrich_table[enrich_table["Term"].str.contains(res.name, case=False)]
            if len(row) == 0:
                print(f"Couldn't find {res['Name']}")
                continue
            elif len(row) > 1:
                row = row.head(1)
            go_id, name, rank, reason, pval, genes = (
                row["ID"].iloc[0],
                row["Term"].iloc[0],
                res.rank,
                res.reason,
                row["Adjusted P-value"].iloc[0],
                row["Genes"].iloc[0],
            )
            if go_id not in subset_go["ID"]:
                subset_go["ID"].append(go_id)
                subset_go["Name"].append(name)
                subset_go["Rank"].append(rank)
                subset_go["Reason"].append(reason)
                subset_go["Genes"].append(genes)
                subset_go["Adjusted P-value"].append(pval)

        return subset_go

    def summarize_graph(self, graph):
        system_prompt = f"""You are an expert in biology and genetics. You have been provided with a graph provides a hypothesis for the connection of a SNP to a phenotype in terms of genes and Go terms.
                   Your task is to summarize the graph in 150 words or less. Return your response in JSON format with the key 'summary'. Don't add any additional information to the response."""

        query = f"Graph: {graph}"

        response = ""
        messages = [
            {
                "role": "system",
                "content":system_prompt,
            },
            {
                "role": "user",
                "content":query
            }]
        response = self._chat(messages) 

        logger.debug(response)
        if response and isinstance(response, dict) and "summary" in response:
            return response["summary"]
        else:
            raise ValueError(f"Invalid JSON response from LLM: {response}")

    def chat(self, query, graph):
        """
        Given a graph as a context, chat with the LLM
        """

        system_prompt = f"""You are an expert in biology and genetics.
        Use the provided graph, which describes a potential hypothesis as to why a SNP is causally related to a phenotype, as a context and answer the query. Your answer should be 100 words or less.

        Return your response in JSON format. JSON key should be `response`. Don't add any additional information to the response."""

        query = f"Graph: {graph}\nQuery: {query}"

        combined_query = f"{system_prompt}\n\nUser:\n{query}"
        messages = [
            {
                "role": "system",
                "content":system_prompt,
            },
            {
                "role": "user",
                "content":query
            }]
        response = self._chat(messages) 

        logger.debug(f"Chat response received: {response}")
        if response and isinstance(response, dict):
            return response.get("response")
        else:
            raise ValueError(f"Invalid JSON response from LLM: {response}")