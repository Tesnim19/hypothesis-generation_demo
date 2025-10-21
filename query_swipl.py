from loguru import logger
import requests

class PrologQuery:

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server = f"http://{self.host}:{self.port}"

    def get_candidate_genes(self, variant_id):
        """
        Given a SNP, get candidate genes that are proximal to it.
        """
        logger.info(f"Getting candidate genes for variant {variant_id}")
        payload = {"rsid": variant_id}
        res = requests.get(f"{self.server}/api/hypgen/candidate_genes", params=payload)
        if not res.ok:
            logger.error(f"Prolog server error for variant {variant_id}: {res.status_code} - {res.text}")
            raise RuntimeError(f"get_candidate_genes failed. Prolog server response: {res.text}")
        
        try:
            result = res.json()
            genes = [g.upper() for g in result["candidate_genes"]]
            return genes
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from Prolog server for variant {variant_id}")
            logger.error(f"Response status: {res.status_code}")
            logger.error(f"Response text: {res.text}")
            raise RuntimeError(f"get_candidate_genes failed. Invalid JSON response from Prolog server. Response: {res.text[:500]}") from e
    
    def get_relevant_gene_proof(self, variant_id, samples):
        payload = {"rsid": variant_id, "samples": samples}
        res = requests.get(f"{self.server}/api/hypgen", params=payload)
        if not res.ok:
            logger.error(f"Prolog server error for variant {variant_id}: {res.status_code} - {res.text}")
            raise RuntimeError(f"get_relevant_gene_proof failed. Prolog server response: {res.text}")
        
        try:
            result = res.json()
            logger.debug(f"Received response: {result}")
            logger.info(f"Found {len(result) if isinstance(result, list) else 'unknown'} gene proof items")
            return result
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from Prolog server for variant {variant_id}")
            logger.error(f"Response status: {res.status_code}")
            logger.error(f"Response text: {res.text}")
            raise RuntimeError(f"get_relevant_gene_proof failed. Invalid JSON response from Prolog server. Response: {res.text[:500]}") from e
    
    def execute_query(self, query):
        logger.info(f"Executing Prolog query: {query}")
        payload = {"query": query}

        res = requests.get(f"{self.server}/api/query", params=payload)
        if not res.ok:
            logger.error(f"Prolog server error for query '{query}': {res.status_code} - {res.text}")
            raise RuntimeError(f"execute_query failed. Prolog server response: {res.text}")
        
        try:
            result = res.json()
            
            # Check if the response contains an error from Prolog
            if isinstance(result, dict) and 'error' in result:
                logger.error(f"Prolog query error for '{query}': {result['error']}")
                raise RuntimeError(f"Prolog query failed: {result['error']}")
            
            return result
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from Prolog server for query '{query}'")
            logger.error(f"Response status: {res.status_code}")
            logger.error(f"Response text: {res.text}")
            
            # Try to clean the response by removing header lines and parse again
            try:
                # Split by lines and find where JSON starts (look for { or [)
                lines = res.text.strip().split('\n')
                json_start_idx = -1
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith('{') or stripped.startswith('['):
                        json_start_idx = i
                        break
                
                if json_start_idx >= 0:
                    # Join from the JSON start onwards
                    json_text = '\n'.join(lines[json_start_idx:])
                    import json
                    result = json.loads(json_text)
                    
                    # Check if this cleaned result contains a Prolog error
                    if isinstance(result, dict) and 'error' in result:
                        logger.error(f"Prolog query error for '{query}': {result['error']}")
                        raise RuntimeError(f"Prolog query failed: {result['error']}")
                    
                    logger.warning(f"Successfully parsed JSON after cleaning response for query '{query}'")
                    return result
                else:
                    raise RuntimeError(f"execute_query failed. Invalid JSON response from Prolog server. Response: {res.text[:500]}") from e
            except Exception as clean_error:
                raise RuntimeError(f"execute_query failed. Invalid JSON response from Prolog server. Response: {res.text[:500]}") from e
    
    def get_gene_ids(self, gene_names):

        logger.info(f"Getting gene IDs for genes: {gene_names}")
        
        # Query each gene name individually
        gene_ids = []
        for gene_name in gene_names:
            query = f"gene_id('{gene_name}', X)"
            try:
                result = self.execute_query(query)
                if result and len(result) > 0:
                    gene_ids.append(result[0])
                else:
                    logger.warning(f"No gene ID found for gene name: {gene_name}")
                    gene_ids.append(gene_name)  
            except Exception as e:
                logger.error(f"Error getting gene ID for {gene_name}: {e}")
                gene_ids.append(gene_name)
        
        return gene_ids