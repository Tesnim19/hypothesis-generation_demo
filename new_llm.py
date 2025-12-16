import json
from typing import List

import scipy.spatial

from pydantic import BaseModel
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
import google.generativeai as genai
import openai
import scipy
import os
from loguru import logger
import re

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from literature_miner import LiteratureMiner

model = SentenceTransformer("all-MiniLM-L6-v2")


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
    return {"title": titles, "text": texts}


class GoTerm(BaseModel):
    rank: int
    name: str
    reason: str


class Response(BaseModel):
    terms: List[GoTerm]


class LLM:

    def __init__(self, llm="gemini", temperature=0.0):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.temperature = temperature
        self.llm_type = llm

        # Initialize literature miner
        self.literature_miner = LiteratureMiner()

        if llm == "gpt4":
            # Check that the openai key is available
            try:
                openai_api_key = os.getenv("OPENAI_API_KEY")
                openai.api_key = openai_api_key
                self.llm = OpenAI(
                    api_key=openai_api_key, temperature=temperature, model="gpt-4-0613"
                )
                # self.llm = OpenAI(api_key=openai_api_key, temperature=temperature, model="gpt-35-turbo-0613")
            except KeyError:
                raise ValueError("Please set the OPENAI_API_KEY environment variable")
        elif llm == "claude":
            # Check that Anthropic key is available
            try:
                anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
                self.llm = Anthropic(
                    api_key=anthropic_api_key,
                    temperature=temperature,
                    model="claude-3-5-sonnet-20240620",
                )
            except KeyError:
                raise ValueError(
                    "Please set the ANTHROPIC_API_KEY environment variable"
                )
        elif llm == "gemini":
            google_api_key = os.getenv("GEMINI_API_KEY")
            if not google_api_key:
                raise ValueError("Please set the GEMINI_API_KEY environment variable")
            genai.configure(api_key=google_api_key)
            self.llm = genai.GenerativeModel("gemini-1.5-flash")
        else:
            raise ValueError(f"Unsupported LLM: {llm}")

    def _chat(self, messages: List[ChatMessage]) -> str:
        if isinstance(self.llm, OpenAI) or isinstance(self.llm, Anthropic):
            return self.llm.chat(messages).message.content
        elif self.llm_type == "gemini":
            full_prompt = "\n".join(
                [f"{m.role.upper()}: {m.content}" for m in messages]
            )
            full_prompt += "\n\nPlease return *only* a JSON object and nothing else."
            response = self.llm.generate_content(full_prompt)
            raw = response.text

            m = re.search(r"(\{.*\})", raw, re.DOTALL)
            if not m:
                raw = raw.strip("`").replace("json\n", "").strip()
                m = re.search(r"(\[.*?\])", raw, re.DOTALL)
                if not m:
                    raise ValueError(
                        f"Could not find a JSON object in the model output:\n{raw!r}"
                    )
            return m.group(1)

        else:
            raise NotImplementedError("Unsupported LLM interface")

    def predict_casual_gene(
        self, phenotype, genes, prev_gene=None, rule=None, use_literature=True
    ):
        """
        Given a variant, a list of candidate genes and a phenotype, query the LLM to predict the causal gene
        Now enhanced with literature mining capabilities
        """
        genes = sorted(genes)
        genes_fmt = []
        for gene in genes:
            genes_fmt.append("{" + gene + "}")

        genes_str = ",".join(genes_fmt)

        # Get literature evidence if enabled
        literature_evidence = ""
        evidence_dict = {}  # Store evidence dict for reuse
        if use_literature:
            try:
                logger.info(
                    f"Searching literature for {phenotype} and {len(genes)} candidate genes"
                )
                evidence_dict = self.literature_miner.get_literature_evidence(
                    phenotype, genes, self
                )
                literature_evidence = self.literature_miner.format_evidence_for_llm(
                    evidence_dict
                )
                logger.info(f"Found literature evidence for {len(evidence_dict)} genes")
            except Exception as e:
                logger.warning(
                    f"Literature mining failed: {str(e)}. Continuing without literature evidence."
                )
                literature_evidence = ""

        if rule is None:
            system_prompt = """You are an expert in biology and genetics.
                            Your task is to identify likely causal genes within a locus for a given GWAS phenotype based on literature evidence.

                            From the list, provide the likely causal gene (matching one of the given genes), confidence (0: very unsure to 1: very confident), and a brief reason (50 words or less) for your choice.

                            Return your response in JSON format, excluding the GWAS phenotype name and gene list in the locus. JSON keys should be 'causal_gene','confidence','reason'.
                            Don't add any additional information to the response.

                        """
        else:
            assert (
                prev_gene is not None
            ), "Previous gene must be provided when rule is provided"
            system_prompt = f"""You are an expert in biology and genetics.
                            Your task is to identify likely causal genes within a locus for a given GWAS phenotype based on literature evidence.

                            From the list, provide the likely causal gene (matching one of the given genes), confidence (0: very unsure to 1: very confident), and a brief reason (50 words or less) for your choice.
                            
                            You previously identified {prev_gene} as a causal gene. Your prediction couldn't be verified by the following prolog rule:
                            
                            {rule}
                            
                            Make sure your prediction is consistent with the rule.
                            Return your response in JSON format, excluding the GWAS phenotype name and gene list in the locus. JSON keys should be 'causal_gene','confidence','reason'.
                            Don't add any additional information to the response.
                        """

        # Build the query with literature evidence
        query = f"GWAS Phenotype: {phenotype}\nGenes: {genes_str}"

        # Add literature evidence if available
        if literature_evidence:
            query += f"\n\n{literature_evidence}"
            logger.info("Enhanced prediction with literature evidence")

        print(f"Query: {query}")
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=query),
        ]
        response = self._chat(messages)
        logger.debug(response)
        try:
            result = json.loads(response)

            # If we have literature evidence, adjust confidence based on evidence
            if literature_evidence and use_literature and evidence_dict:
                try:
                    predicted_gene = result.get("causal_gene", "")

                    if predicted_gene in evidence_dict:
                        evidence = evidence_dict[predicted_gene]
                        # Boost confidence if there's literature evidence
                        if evidence["pubmed_count"] > 0:
                            original_confidence = result.get("confidence", 0.5)
                            boosted_confidence = min(original_confidence * 1.2, 1.0)
                            result["confidence"] = boosted_confidence
                            result[
                                "reason"
                            ] += f" (Literature evidence: {evidence['pubmed_count']} articles)"
                            logger.info(
                                f"Boosted confidence for {predicted_gene} based on literature evidence"
                            )
                except Exception as e:
                    logger.warning(
                        f"Failed to adjust confidence with literature evidence: {str(e)}"
                    )

            return result
        except json.JSONDecodeError:
            raise ValueError("Json error when decoding")

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

            # Use the existing chat method
            messages = [
                ChatMessage(
                    role="system",
                    content="You are a medical terminology expert. Return only valid JSON arrays.",
                ),
                ChatMessage(role="user", content=prompt),
            ]

            
            import time

            time.sleep(1)  

            response = self._chat(messages)

            
            try:
                variations = json.loads(response)
                logger.info(
                    f"LLM generated {len(variations)} phenotype variations for '{phenotype}': {variations}"
                )
                return variations
            except:
                logger.warning(
                    f"Could not find a JSON array in the model output for phenotype '{phenotype}': {response}"
                )
                # Fallback to basic variations
                return self._get_basic_phenotype_variations(phenotype)

        except Exception as e:
            logger.error(
                f"Failed to get LLM phenotype variations for '{phenotype}': {str(e)}"
            )
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
        # df.drop(columns=["ID", "Adjusted P-value", "Genes"], inplace=True)
        # df.to_csv(tmp_file, index=False)

        # Embed the GO terms and their descriptions.
        res = self._retrieve_top_k_go_terms(phentoype, enrich_tbl, k)

        return res

    def _embed_dataset(self, batch):
        combined_text = []
        for title, text in zip(batch["title"], batch["text"]):
            combined_text.append(" [SEP] ".join([title, text]))

        return {"embeddings": self.embed_model.encode(combined_text)}

    def _retrieve_top_k_go_terms(self, query, data, k):

        texts = [
            f"{row['Term'].strip()} [SEP] {row['Desc'].strip()}"
            for _, row in data.iterrows()
        ]

        embeddings = self.model.encode(texts, convert_to_tensor=False)
        query_embedding = self.model.encode(query, convert_to_tensor=False)

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
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=query),
        ]
        response = self._chat(messages)
        logger.debug(response)
        try:
            return json.loads(response)["summary"]
        except json.JSONDecodeError:
            raise ValueError("Json error when decoding")

    def chat(self, query, graph):
        """
        Given a graph as a context, chat with the LLM
        """

        system_prompt = f"""You are an expert in biology and genetics. 
        Use the provided graph, which describes a potential hypothesis as to why a SNP is causally related to a phenotype, as a context and answer the query. Your answer should be 100 words or less.
        
        Return your response in JSON format. JSON key should be `response`. Don't add any additional information to the response."""

        query = f"Graph: {graph}\nQuery: {query}"
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=query),
        ]
        response = self._chat(messages)
        print(f"LLM Response: {response}")
        return json.loads(response).get("response")
