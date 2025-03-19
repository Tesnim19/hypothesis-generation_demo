from prefect import flow
from prefect.task_runners import ConcurrentTaskRunner
from prefect.context import get_run_context
import time
from datetime import datetime, timezone
from uuid import uuid4
from loguru import logger
from utils import emit_task_update
from status_tracker import TaskState
from tasks import check_enrich, get_candidate_genes, predict_causal_gene, get_relevant_gene_proof, retry_predict_causal_gene, retry_get_relevant_gene_proof, create_enrich_data 
from tasks import check_hypothesis, get_enrich, get_gene_ids, execute_gene_query, execute_variant_query,summarize_graph, create_hypothesis, execute_phenotype_query

### Enrichment Flow
@flow(log_prints=True)
async def enrichment_flow(enrichr, llm, prolog_query, db, current_user_id, phenotype, variant, hypothesis_id):
    try:
        enrich = check_enrich(db, current_user_id, phenotype, variant, hypothesis_id)
        if enrich:
            print("Retrieved enrich data from saved db")
            return {"id": enrich.get('id')}, 200

        candidate_genes = get_candidate_genes(prolog_query, variant, hypothesis_id)
        time.sleep(3)
        causal_gene = predict_causal_gene(llm, phenotype, candidate_genes, hypothesis_id)
        time.sleep(3)
        causal_graph, proof = get_relevant_gene_proof(prolog_query, variant, causal_gene, hypothesis_id)

        # mock causal_graph
        causal_graph = {'nodes': [{'id': 'rs1421085', 'type': 'snp'}, {'id': 'ensg00000177508', 'type': 'gene'}, {'id': 'rs1421085', 'type': 'snp'}, {'id': 'ensg00000177508', 'type': 'gene'}, {'id': 'rs1421085', 'type': 'snp'}, {'id': 'chr16_53741418_53785410_grch38', 'type': 'super_enhancer'}, {'id': 'chr16_53741418_53785410_grch38', 'type': 'super_enhancer'}, {'id': 'ensg00000140718', 'type': 'gene'}, {'id': 'rs1421085', 'type': 'snp'}, {'id': 'ensg00000125798', 'type': 'gene'}, {'id': 'ensg00000125798', 'type': 'gene'}, {'id': 'ensg00000177508', 'type': 'gene'}, {'id': 'ensg00000125798', 'type': 'gene'}, {'id': 'chr16_53744537_53744917_grch38', 'type': 'tfbs'}, {'id': 'chr16_53744537_53744917_grch38', 'type': 'tfbs'}, {'id': 'chr16_53741418_53785410_grch38', 'type': 'super_enhancer'}], 'edges': [{'source': 'rs1421085', 'target': 'ensg00000177508', 'label': 'eqtl_association'}, {'source': 'rs1421085', 'target': 'ensg00000177508', 'label': 'in_tad_with'}, {'source': 'rs1421085', 'target': 'chr16_53741418_53785410_grch38', 'label': 'in_regulatory_region'}, {'source': 'chr16_53741418_53785410_grch38', 'target': 'ensg00000140718', 'label': 'associated_with'}, {'source': 'rs1421085', 'target': 'ensg00000125798', 'label': 'alters_tfbs'}, {'source': 'ensg00000125798', 'target': 'ensg00000177508', 'label': 'regulates'}, {'source': 'ensg00000125798', 'target': 'chr16_53744537_53744917_grch38', 'label': 'binds_to'}, {'source': 'chr16_53744537_53744917_grch38', 'target': 'chr16_53741418_53785410_grch38', 'label': 'overlaps_with'}]}

        if causal_graph is None:
            causal_gene = retry_predict_causal_gene(llm, phenotype, candidate_genes, proof, causal_gene, hypothesis_id)
            causal_graph, proof = retry_get_relevant_gene_proof(prolog_query, variant, causal_gene, hypothesis_id)
            print("Retried causal gene: ", causal_gene)
            print("Retried causal graph: ", causal_graph)

        enrich_tbl = enrichr.run(causal_gene)
        relevant_gos = llm.get_relevant_go(phenotype, enrich_tbl)

        enrich_id = create_enrich_data(db, variant, phenotype, causal_gene, relevant_gos, causal_graph, current_user_id, hypothesis_id)
        
        return {"id": enrich_id}, 201
    except Exception as e:
        logger.error(f"Error in enrichment flow: {str(e)}")
        raise

@flow(name="async_enrichment_process", task_runner=ConcurrentTaskRunner())
async def async_enrichment_process(enrichr, llm, prolog_query, db, current_user_id, phenotype, variant, hypothesis_id):
    """Wrapper flow to run enrichment process"""
    try:
        flow_result = await enrichment_flow(
            enrichr=enrichr,
            llm=llm,
            prolog_query=prolog_query,
            db=db,
            current_user_id=current_user_id,
            phenotype=phenotype,
            variant=variant,
            hypothesis_id=hypothesis_id
        )
        
        # Update hypothesis with enrichment ID
        db.update_hypothesis(hypothesis_id, {
            "enrich_id": flow_result[0].get('id'),
            "status": "completed",
            "updated_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds') + "Z",
        })

        logger.info(f"Enrichment flow completed: {flow_result}")
        print(f"Enrichment flow completed: {flow_result}")
        
        return flow_result
        
    except Exception as e:
        logger.error(f"Enrichment flow failed: {str(e)}")
        
        # Update hypothesis with error state
        db.update_hypothesis(hypothesis_id, {
            "status": "failed",
            "error": str(e),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds') + "Z",
        })

        # Emit failure update
        emit_task_update(
            hypothesis_id=hypothesis_id,
            task_name="Enrichment",
            state=TaskState.FAILED,
            error=str(e),
            progress=0
        )
        raise

### Hypothesis Flow
@flow(log_prints=True)
def hypothesis_flow(current_user_id, hypothesis_id, enrich_id, go_id, db, prolog_query, llm):
    hypothesis = check_hypothesis(db, current_user_id, enrich_id, go_id, hypothesis_id)
    if hypothesis:
        print("Retrieved hypothesis data from saved db")
        return {"summary": hypothesis.get('summary'), "graph": hypothesis.get('graph')}, 200

    enrich_data = get_enrich(db, current_user_id, enrich_id, hypothesis_id)
    if not enrich_data:
        return {"message": "Invalid enrich_id or access denied."}, 404

    go_term = [go for go in enrich_data["GO_terms"] if go["id"] == go_id]
    go_name = go_term[0]["name"]
    causal_gene = enrich_data['causal_gene']
    variant_id = enrich_data['variant']
    phenotype = enrich_data['phenotype']
    coexpressed_gene_names = go_term[0]["genes"]
    causal_graph = enrich_data['causal_graph']

    print(f"Enrich data: {enrich_data}")

    # causal_gene_id = get_gene_ids(prolog_query, [causal_gene.lower()], hypothesis_id)[0]
    time.sleep(3)
    causal_gene_id = get_gene_ids(1, hypothesis_id)[0]
    time.sleep(3)
    coexpressed_gene_ids = get_gene_ids(2, hypothesis_id)
    # coexpressed_gene_ids = get_gene_ids(prolog_query, [g.lower() for g in coexpressed_gene_names], hypothesis_id)

    nodes, edges = causal_graph["nodes"], causal_graph["edges"]

    gene_nodes = [n for n in nodes if n["type"] == "gene"]
    gene_ids = [n['id'] for n in gene_nodes]
    gene_entities = [f"gene({id})" for id in gene_ids]
    query = f"maplist(gene_name, {gene_entities}, X)".replace("'", "")

    gene_names = execute_gene_query(prolog_query, query, hypothesis_id)
    for id, name, node in zip(gene_ids, gene_names, gene_nodes):
        node["id"] = id
        node["name"] = name.upper()
    
    variant_nodes = [n for n in nodes if n["type"] == "snp"]
    variant_rsids = [n['id'] for n in variant_nodes]
    variant_entities = [f"snp({id})" for id in variant_rsids]
    query = f"maplist(variant_id, {variant_entities}, X)".replace("'", "")

    time.sleep(3)
    variant_ids = execute_variant_query(prolog_query, query, hypothesis_id)
    for variant_id, rsid, node in zip(variant_ids, variant_rsids, variant_nodes):
        variant_id = variant_id.replace("'", "")
        node["id"] = variant_id
        node["name"] = rsid
        source_edges = [e for e in edges if e["source"] == rsid]
        target_edges = [e for e in edges if e["target"] == rsid]
        for edge in source_edges:
            edge["source"] = variant_id
        for edge in target_edges:
            edge["target"] = variant_id
            
    nodes.append({"id": go_id, "type": "go", "name": go_name})
    time.sleep(3)
    phenotype_id = execute_phenotype_query(prolog_query, phenotype, hypothesis_id)

    nodes.append({"id": phenotype_id, "type": "phenotype", "name": phenotype})
    edges.append({"source": go_id, "target": phenotype_id, "label": "involved_in"})
    for gene_id, gene_name in zip(coexpressed_gene_ids, coexpressed_gene_names):
        nodes.append({"id": gene_id, "type": "gene", "name": gene_name})
        edges.append({"source": gene_id, "target": go_id, "label": "enriched_in"})
        edges.append({"source": causal_gene_id, "target": gene_id, "label": "coexpressed_with"})

    causal_graph = {"nodes": nodes, "edges": edges}
    summary = summarize_graph(llm, causal_graph, hypothesis_id)
    
    hypothesis_id = create_hypothesis(db, enrich_id, go_id, variant_id, phenotype, causal_gene, causal_graph, summary, current_user_id, hypothesis_id)
    
    return {"summary": summary, "graph": causal_graph}, 201