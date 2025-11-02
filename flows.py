import asyncio
import os
import time
from flask import json
from loguru import logger
from prefect import flow
from status_tracker import TaskState
import multiprocessing as mp
from uuid import uuid4
from analysis_tasks import create_guaranteed_demo_credible_set

from tasks import (
    check_enrich, create_enrich_data, get_candidate_genes, predict_causal_gene, 
    get_relevant_gene_proof, retry_predict_causal_gene, retry_get_relevant_gene_proof,
    check_hypothesis, get_enrich, get_gene_ids, execute_gene_query, execute_variant_query,
    summarize_graph, create_hypothesis, execute_phenotype_query
)

from analysis_tasks import (
    munge_sumstats_preprocessing, filter_significant_variants, run_cojo_per_chromosome, create_region_batches, finemap_region_batch_worker,
    save_sumstats_for_workers, cleanup_sumstats_file
)

from project_tasks import (
    save_analysis_state_task, create_analysis_result_task, 
    get_project_analysis_path_task
)

import pandas as pd
from datetime import datetime, timezone
from prefect.task_runners import ThreadPoolTaskRunner

from utils import emit_task_update
from config import Config, create_dependencies
from status_tracker import TaskState

def _extract_causal_gene_for_enrichment(graph, candidate_genes):
    """Extract causal gene from graph for enrichment analysis, fallback to first candidate gene"""
    if not graph:
        return candidate_genes[0] if candidate_genes else "UNKNOWN"
    
    nodes = graph.get("nodes", [])
    gene_nodes = [n for n in nodes if n.get("type") == "gene"]
    
    if gene_nodes:
        # Return the first gene from the graph
        gene_name = gene_nodes[0].get("name", gene_nodes[0].get("id", ""))
        return gene_name.upper() if gene_name else (candidate_genes[0] if candidate_genes else "UNKNOWN")
    
    # Fallback to first candidate gene
    return candidate_genes[0] if candidate_genes else "UNKNOWN"

### Enrichment Flow
@flow(log_prints=True, persist_result=False, task_runner=ThreadPoolTaskRunner(max_workers=4))
def enrichment_flow(current_user_id, phenotype, variant, hypothesis_id, project_id):
    """
    Fully project-based enrichment flow that initializes dependencies from centralized config
    """
    # Initialize dependencies from environment variables
    config = Config.from_env()
    deps = create_dependencies(config)
    
    enrichr = deps['enrichr']
    llm = deps['llm']
    prolog_query = deps['prolog_query']
    hypotheses = deps['hypotheses']
    
    try:
        logger.info(f"Running project-based enrichment for project {project_id}, variant {variant}")
        
        # Check for existing enrichment data
        enrich = check_enrich.submit(deps['enrichment'], current_user_id, variant, phenotype, hypothesis_id).result()
        
        if enrich:
            logger.info("Retrieved enrich data from saved db")
            return {"id": enrich['id']}, 200

        candidate_genes = get_candidate_genes.submit(prolog_query, variant, hypothesis_id).result()
        graphs_list = get_relevant_gene_proof.submit(prolog_query, variant, hypothesis_id).result()

        if not graphs_list or len(graphs_list) == 0:
            graphs_list = retry_get_relevant_gene_proof.submit(prolog_query, variant, hypothesis_id).result()
            logger.info(f"Retried graphs: {len(graphs_list) if graphs_list else 0} graphs received")

        logger.info(f"Creating enrichments for {len(graphs_list) if graphs_list else 0} graphs from Prolog server")

        # Sort graphs by probability (highest first)
        graphs_with_prob = []
        for i, graph in enumerate(graphs_list):
            prob = graph.get('prob', {}).get('value', 0.0)
            graphs_with_prob.append((i, graph, prob))
        
        graphs_with_prob.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"Graph probabilities: {[(i, prob) for i, _, prob in graphs_with_prob]}")

        # Extract causal gene from the highest probability graph for enrichment analysis
        temp_causal_gene = _extract_causal_gene_for_enrichment(graphs_with_prob[0][1] if graphs_with_prob else None, candidate_genes)
        
        enrich_tbl = enrichr.run(temp_causal_gene)
        relevant_gos = llm.get_relevant_go(phenotype, enrich_tbl)

        # Create enrichments for all graphs
        enrichment_data = []
        main_enrichment_id = None
        
        for idx, (original_i, graph, prob) in enumerate(graphs_with_prob):
            # Create enrichment for this graph
            enrich_id = create_enrich_data.submit(
                deps['enrichment'], hypotheses, current_user_id, project_id, variant, 
                phenotype, causal_gene, relevant_gos, {
                    "graph": graph,
                    "graph_index": original_i,
                    "total_graphs": len(graphs_list),
                }, hypothesis_id
            ).result()
            
            enrichment_data.append({
            
            enrichment_data.append({
                "enrich_id": enrich_id,
                "graph_index": original_i,
                "graph_probability": prob
            })
            
            if idx == 0:
                main_enrichment_id = enrich_id

        all_enrich_ids = [e['enrich_id'] for e in enrichment_data]
        
        # Update original hypothesis with main enrichment and children info
        hypotheses.update_hypothesis(hypothesis_id, {
            "enrich_id": main_enrichment_id,
            "child_enrich_ids": all_enrich_ids[1:],
            "status": "enrichment_complete"
        })

        logger.info(f"Created {len(enrichment_data)} enrichments, main: {main_enrichment_id}")
        logger.info(f"Child enrichments (will be processed on-demand): {all_enrich_ids[1:]}")
        logger.info(f"Created {len(enrichment_data)} enrichments, main: {main_enrichment_id}")
        logger.info(f"Child enrichments (will be processed on-demand): {all_enrich_ids[1:]}")
        
        # Return main enrichment
        return {"id": main_enrichment_id}, 200
    except Exception as e:
        logger.error(f"Enrichment flow failed: {str(e)}")
        
        # Update hypothesis with error state
        hypotheses.update_hypothesis(hypothesis_id, {
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
def hypothesis_flow(current_user_id, hypothesis_id, enrich_id, go_id, hypotheses, prolog_query, llm, enrichment):
    
    hypothesis = check_hypothesis(hypotheses, current_user_id, enrich_id, go_id, hypothesis_id)
    if hypothesis:
        logger.info("Retrieved hypothesis data from saved db")
        
        summary = hypothesis.get('summary')
        graph = hypothesis.get('graph')
        
        if summary and graph:
            logger.info(f"Returning existing hypothesis with summary and graph")
            return {"summary": summary, "graph": graph}, 201
        else:
            # If incomplete, log warning and continue with generation
            logger.warning(f"Existing hypothesis {hypothesis_id} missing summary ({bool(summary)}) or graph ({bool(graph)}), regenerating...")
            # Continue to generation below
    
    # Check if this hypothesis has child enrichments
    parent_hypothesis = hypotheses.get_hypotheses(current_user_id, hypothesis_id)
    if parent_hypothesis and 'child_enrich_ids' in parent_hypothesis:
        child_enrich_ids = parent_hypothesis.get('child_enrich_ids', [])
        if child_enrich_ids and len(child_enrich_ids) > 0:
            logger.info(f"Triggering background processing for {len(child_enrich_ids)} child enrichments")
            
            from threading import Thread
            
            # Create deps dict from current context
            deps_for_bg = {
                'hypotheses': hypotheses,
                'enrichment': enrichment,
                'prolog_query': prolog_query,
                'llm': llm
            }
            
            def run_background_hypotheses():
                try:
                    process_child_enrichments_simple(
                        current_user_id, child_enrich_ids, hypothesis_id, deps_for_bg
                    )
                except Exception as bg_e:
                    logger.error(f"Background child hypothesis generation failed: {str(bg_e)}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            bg_thread = Thread(target=run_background_hypotheses)
            bg_thread.start()
            logger.info(f"Background thread started for child enrichments (processing in parallel)")

    enrich_data = get_enrich(enrichment, current_user_id, enrich_id, hypothesis_id)
    if not enrich_data:
        return {"message": "Invalid enrich_id or access denied."}, 404

    go_term = [go for go in enrich_data["GO_terms"] if go["id"] == go_id]
    go_name = go_term[0]["name"]
    causal_gene = enrich_data['causal_gene']
    variant_id = enrich_data['variant']
    phenotype = enrich_data['phenotype']
    coexpressed_gene_names = go_term[0]["genes"]
    causal_graph_data = enrich_data['causal_graph']
    
    graph = causal_graph_data["graph"]
    graph_index = causal_graph_data.get("graph_index", 0)
    total_graphs = causal_graph_data.get("total_graphs", 1)
    
    logger.info(f"Processing graph {graph_index + 1}/{total_graphs} from Prolog server")
    
    graph_prob = graph.get('prob', {}).get('value', 1.0)
    logger.info(f"Processing graph {graph_index + 1}/{total_graphs} with probability {graph_prob}")
    
    causal_graph = graph      
    
    # Extract causal gene from graph
    def extract_causal_gene_from_graph(graph, variant_nodes):
        """Extract the most likely causal gene from the Prolog graph structure"""
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        
        # Get all gene nodes
        gene_nodes = [n for n in nodes if n.get("type") == "gene"]
        if not gene_nodes:
            return None, None
        
        # Strategy 1: Find genes directly connected to SNPs
        snp_ids = [n.get("id", n.get("name", "")) for n in variant_nodes]
        directly_connected_genes = []
        
        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            
            # Check if edge connects SNP to gene
            for snp_id in snp_ids:
                if source == snp_id and any(target == g.get("id", "") for g in gene_nodes):
                    gene_node = next((g for g in gene_nodes if g.get("id", "") == target), None)
                    if gene_node:
                        directly_connected_genes.append(gene_node)
                elif target == snp_id and any(source == g.get("id", "") for g in gene_nodes):
                    gene_node = next((g for g in gene_nodes if g.get("id", "") == source), None)
                    if gene_node:
                        directly_connected_genes.append(gene_node)
        
        # If we found directly connected genes, use the first one
        if directly_connected_genes:
            causal_gene_node = directly_connected_genes[0]
            return causal_gene_node.get("id", ""), causal_gene_node.get("name", "")
        
        # Strategy 2: Use the first gene in the graph (fallback)
        if gene_nodes:
            causal_gene_node = gene_nodes[0]
            return causal_gene_node.get("id", ""), causal_gene_node.get("name", "")
        
        return None, None
    
    coexpressed_gene_ids = get_gene_ids(prolog_query, [g.lower() for g in coexpressed_gene_names], hypothesis_id)

    nodes, edges = causal_graph["nodes"], causal_graph["edges"]
    
    # Process variant nodes first to get their IDs
    
    # Process variant nodes first to get their IDs
    variant_nodes = [n for n in nodes if n["type"] == "snp"]
    variant_rsids = [n['id'] for n in variant_nodes]
    variant_entities = [f"snp({id})" for id in variant_rsids]
    query = f"maplist(variant_id, {variant_entities}, X)".replace("'", "")

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
    
    # Now extract causal gene from the current graph (after variant processing)
    extracted_gene_id, extracted_gene_name = extract_causal_gene_from_graph(causal_graph, variant_nodes)
    
    if extracted_gene_id:
        # Use the gene extracted from the graph
        causal_gene_id = extracted_gene_id.lower()
        causal_gene_name = extracted_gene_name.upper() if extracted_gene_name else extracted_gene_id.upper()
        logger.info(f"Extracted causal gene from graph: {causal_gene_name} (ID: {causal_gene_id})")
    else:
        # Fallback to LLM prediction if extraction fails
        causal_gene_id = causal_gene.lower()
        causal_gene_names = execute_gene_query(prolog_query, f"maplist(gene_name, [gene({causal_gene_id})], X)", hypothesis_id)
        causal_gene_name = causal_gene_names[0].upper() if causal_gene_names else causal_gene.upper()
        logger.info(f"Using LLM predicted causal gene as fallback: {causal_gene_name}")

    gene_nodes = [n for n in nodes if n["type"] == "gene"]
    gene_ids = [n['id'] for n in gene_nodes]
    gene_entities = [f"gene({id})" for id in gene_ids]
    query = f"maplist(gene_name, {gene_entities}, X)".replace("'", "")

    gene_names = execute_gene_query(prolog_query, query, hypothesis_id)
    for id, name, node in zip(gene_ids, gene_names, gene_nodes):
        node["id"] = id
        node["name"] = name.upper()
            
    # Add the causal gene node if not present
    if causal_gene_id not in gene_ids:
        nodes.append({"id": causal_gene_id, "type": "gene", "name": causal_gene_name})
    
    nodes.append({"id": go_id, "type": "go", "name": go_name})
    phenotype_result = execute_phenotype_query(prolog_query, phenotype, hypothesis_id)
    
    phenotype_id = phenotype_result[0] if isinstance(phenotype_result, list) and phenotype_result else phenotype_result

    nodes.append({"id": phenotype_id, "type": "phenotype", "name": phenotype})
    edges.append({"source": go_id, "target": phenotype_id, "label": "involved_in"})
    for gene_id, gene_name in zip(coexpressed_gene_ids, coexpressed_gene_names):
        nodes.append({"id": gene_id, "type": "gene", "name": gene_name})
        edges.append({"source": gene_id, "target": go_id, "label": "enriched_in"})
        edges.append({"source": causal_gene_id, "target": gene_id, "label": "coexpressed_with"})

    final_causal_graph = {"nodes": nodes, "edges": edges, "probability": graph_prob}

    summary = summarize_graph(llm, {"nodes": nodes, "edges": edges}, hypothesis_id)

    create_hypothesis(hypotheses, enrich_id, go_id, variant_id, phenotype, causal_gene_name, final_causal_graph, 
                     summary, current_user_id, hypothesis_id)
    
    return {"summary": summary, "graph": final_causal_graph}, 201


### Background Child Enrichment Processing (runs in thread)
def process_child_enrichments_simple(current_user_id, child_enrich_ids, parent_hypothesis_id, deps):
    """
    Process child enrichments in background - each using its FIRST GO term
    Calls hypothesis_flow.fn() directly to avoid Prefect recursion
    NOTE: This runs in a background thread, NOT as a Prefect flow
    """
    hypotheses = deps['hypotheses']
    enrichment = deps['enrichment']
    prolog_query = deps['prolog_query']
    llm = deps['llm']
    
    logger.info(f"Background processing started for {len(child_enrich_ids)} child enrichments")
    
    # Get parent hypothesis to extract project_id
    parent_hypothesis = hypotheses.get_hypotheses(current_user_id, parent_hypothesis_id)
    parent_project_id = parent_hypothesis.get('project_id') if parent_hypothesis else None
    
    try:
        for enrich_id in child_enrich_ids:
            logger.info(f"Processing child enrichment {enrich_id}")
            
            try:
                # Get enrichment data to find first GO term
                enrich_data = get_enrich.fn(enrichment, current_user_id, enrich_id, parent_hypothesis_id)
                if not enrich_data:
                    logger.warning(f"Could not get enrichment data for {enrich_id}")
                    continue
                
                # Get GO terms from enrichment
                go_terms = enrich_data.get("GO_terms", [])
                if not go_terms or len(go_terms) == 0:
                    logger.warning(f"No GO terms found for enrichment {enrich_id}")
                    continue
                
                # Get FIRST GO term only
                go_term = go_terms[0]
                go_id = go_term.get("id")
                if not go_id:
                    logger.warning(f"No GO ID found in first GO term for enrichment {enrich_id}")
                    continue
                
                # Get phenotype and variant from enrichment data
                phenotype = enrich_data.get('phenotype')
                variant = enrich_data.get('variant')
                
                # Check if hypothesis already exists for this enrichment + GO combination
                all_hypotheses = hypotheses.get_hypotheses(current_user_id)
                existing_hyp = None
                if isinstance(all_hypotheses, list):
                    for h in all_hypotheses:
                        if h.get('enrich_id') == enrich_id and h.get('go_id') == go_id:
                            existing_hyp = h
                            break
                
                if existing_hyp:
                    logger.info(f"Hypothesis already exists for enrichment {enrich_id} + GO {go_id}")
                    continue
                
                # Create unique hypothesis ID
                new_hypothesis_id = str(uuid4())
                
                # Create the hypothesis record in DB FIRST (so it exists when hypothesis_flow runs)
                from datetime import datetime, timezone
                hypothesis_data = {
                    "id": new_hypothesis_id,
                    "enrich_id": enrich_id,
                    "go_id": go_id,
                    "phenotype": phenotype,
                    "variant": variant,
                    "status": "pending",
                    "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds') + "Z",
                    "task_history": [],
                }
                
                # Add project_id if parent has one
                if parent_project_id:
                    hypothesis_data["project_id"] = parent_project_id
                hypotheses.create_hypothesis(current_user_id, hypothesis_data)
                logger.info(f"Created hypothesis record {new_hypothesis_id} for child enrichment {enrich_id}")
                
                # Call hypothesis_flow directly (using .fn to bypass Prefect)
                logger.info(f"Generating hypothesis for child enrichment {enrich_id}, GO: {go_id}")
                hypothesis_flow.fn(current_user_id, new_hypothesis_id, enrich_id, go_id, hypotheses, prolog_query, llm, enrichment)
                logger.info(f"Successfully generated background hypothesis {new_hypothesis_id}")
                
            except Exception as hyp_e:
                logger.error(f"Failed to generate hypothesis for enrichment {enrich_id}: {str(hyp_e)}")
                logger.exception(hyp_e)  # Log full traceback
                continue
        
        logger.info(f"Background hypothesis generation completed")
        
    except Exception as e:
        logger.error(f"Background hypothesis generation flow failed: {str(e)}")
        raise


@flow(log_prints=True)
def analysis_pipeline_flow(projects_handler, analysis_handler, mongodb_uri, db_name, user_id, project_id, gwas_file_path, ref_genome="GRCh37", 
                           population="EUR", batch_size=5, max_workers=3,
                           maf_threshold=0.01, seed=42, window=2000, L=-1, 
                           coverage=0.95, min_abs_corr=0.5):
    """
    Complete analysis pipeline flow using Prefect for orchestration
    but multiprocessing for fine-mapping batches (R safety)
    """
    
    logger.info(f"[PIPELINE] Starting Prefect analysis pipeline with multiprocessing fine-mapping")
    logger.info(f"[PIPELINE] Project: {project_id}, User: {user_id}")
    logger.info(f"[PIPELINE] File: {gwas_file_path}")
    logger.info(f"[PIPELINE] Batch size: {batch_size} regions per worker process")
    logger.info(f"[PIPELINE] Max workers: {max_workers}")
    logger.info(f"[PIPELINE] Parameters: maf={maf_threshold}, seed={seed}, window={window}kb, L={L}, coverage={coverage}, min_abs_corr={min_abs_corr}")
    
    try:
        # Get project-specific output directory (using Prefect task)
        output_dir = get_project_analysis_path_task.submit(projects_handler, user_id, project_id).result()
        logger.info(f"[PIPELINE] Using output directory: {output_dir}")
        
        # Save initial analysis state
        initial_state = {
            "status": "Running",
            "stage": "Preprocessing",
            "progress": 10,
            "message": "Starting MungeSumstats preprocessing",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        save_analysis_state_task.submit(projects_handler, user_id, project_id, initial_state).result()
        
        logger.info(f"[PIPELINE] Stage 1: MungeSumstats preprocessing")
        munged_file_result = munge_sumstats_preprocessing.submit(gwas_file_path, output_dir, ref_genome=ref_genome, n_threads=14).result()
        
        # Extract the actual file path from the result
        if isinstance(munged_file_result, tuple):
            munged_df, munged_file = munged_file_result
        else:
            munged_file = munged_file_result
            munged_df = pd.read_csv(munged_file, sep='\t')
        
        # Update analysis state after preprocessing
        preprocessing_state = {
            "status": "Running",
            "stage": "Filtering",
            "progress": 30,
            "message": "Preprocessing completed, filtering significant variants",
            "started_at": initial_state["started_at"]
        }
        save_analysis_state_task.submit(projects_handler, user_id, project_id, preprocessing_state).result()
        
        logger.info(f"[PIPELINE] Stage 2: Loading and filtering variants")
        significant_df_result = filter_significant_variants.submit(munged_df, output_dir).result()
        
        # Extract the actual DataFrame
        if isinstance(significant_df_result, tuple):
            significant_df, _ = significant_df_result
        else:
            significant_df = significant_df_result
        
        # Update analysis state after filtering
        filtering_state = {
            "status": "Running",
            "stage": "Cojo",
            "progress": 50,
            "message": "Filtering completed, running COJO analysis"
        }
        save_analysis_state_task.submit(projects_handler, user_id, project_id, filtering_state).result()
        
        logger.info(f"[PIPELINE] Stage 3: COJO analysis")
       
        config = Config.from_env()
        plink_dir = config.plink_dir
        cojo_result = run_cojo_per_chromosome.submit(significant_df, plink_dir, output_dir, maf_threshold=maf_threshold, population=population).result()
        
        # Extract the actual DataFrame
        if isinstance(cojo_result, tuple):
            cojo_results, _ = cojo_result
        else:
            cojo_results = cojo_result
        
        if cojo_results is None or len(cojo_results) == 0:
            logger.error("[PIPELINE] No COJO results to process")
            # Save failed state
            failed_state = {
                "status": "Failed",
                "stage": "Cojo",
                "progress": 50,
                "message": "COJO analysis failed - no independent signals found",
            }
            save_analysis_state_task.submit(projects_handler, user_id, project_id, failed_state).result()
            return None
        
        # Update analysis state after COJO
        cojo_state = {
            "status": "Running",
            "stage": "Fine_mapping",
            "progress": 70,
            "message": "COJO analysis completed, starting fine-mapping"
        }
        save_analysis_state_task.submit(projects_handler, user_id, project_id, cojo_state).result()
        
        logger.info(f"[PIPELINE] Stage 4: Multiprocessing fine-mapping)")
        logger.info(f"[PIPELINE] Processing {len(cojo_results)} regions with {batch_size} regions per batch")
        
        region_batches = create_region_batches(cojo_results, batch_size=batch_size)
        logger.info(f"[PIPELINE] Created {len(region_batches)} batches for {max_workers} worker processes")
        
        sumstats_temp_file = save_sumstats_for_workers(significant_df, output_dir)
        
        # Prepare batch data for multiprocessing
        batch_data_list = []
        for i, batch in enumerate(region_batches):
            db_params = {
                'uri': mongodb_uri,
                'db_name': db_name
            }
            batch_data = (batch, f"batch_{i}", sumstats_temp_file, {
                'db_params': db_params,
                'user_id': user_id,
                'project_id': project_id,
                'finemap_params': {
                    'seed': seed,
                    'window': window,
                    'L': L,
                    'coverage': coverage,
                    'min_abs_corr': min_abs_corr,
                    'population': population,
                    'ref_genome': ref_genome,
                    'maf_threshold': maf_threshold
                }
            })
            batch_data_list.append(batch_data)
        
        original_method = mp.get_start_method()
        if original_method != 'spawn':
            logger.info(f"[PIPELINE] Switching multiprocessing method from '{original_method}' to 'spawn' to reduce memory usage")
            mp.set_start_method('spawn', force=True)
        
        all_results = []
        successful_batches = 0
        failed_batches = 0
        
        try:
            with mp.Pool(max_workers) as pool:
                try:
                    # Process all batches in parallel
                    batch_results_list = pool.map(finemap_region_batch_worker, batch_data_list)
                    
                    # Collect results
                    for i, batch_results in enumerate(batch_results_list):
                        if batch_results and len(batch_results) > 0:
                            all_results.extend(batch_results)
                            successful_batches += 1
                            logger.info(f"[PIPELINE] Batch {i} completed with {len(batch_results)} regions")
                        else:
                            failed_batches += 1
                            logger.warning(f"[PIPELINE] Batch {i} failed or returned no results")
                            
                except Exception as e:
                    logger.error(f"[PIPELINE] Error in multiprocessing: {str(e)}")
                    raise
                finally:
                    # Clean up temporary sumstats file after all workers are done
                    cleanup_sumstats_file(sumstats_temp_file)
        finally:
            # Restore original multiprocessing method
            if original_method != 'spawn':
                try:
                    mp.set_start_method(original_method, force=True)
                    logger.info(f"[PIPELINE] Restored multiprocessing method to '{original_method}'")
                except:
                    logger.warning(f"[PIPELINE] Could not restore multiprocessing method to '{original_method}'")
        
        # Combine and save results
        if all_results:
            # Always add guaranteed demo variant
            try:
                demo_variant = create_guaranteed_demo_credible_set()
                all_results.append(demo_variant)
                logger.info("[PIPELINE] Added guaranteed demo variant rs1421085")
                
                # Also save demo variant as a proper credible set in database
                from utils import transform_credible_sets_to_locuszoom
                demo_credible_set_data = transform_credible_sets_to_locuszoom(demo_variant)
                
                # Save demo variant as credible set with proper format matching normal credible sets
                analysis_handler.save_credible_set(user_id, project_id, {
                    'set_id': 999,  # Unique set ID for demo
                    'variants': demo_credible_set_data,
                    'coverage': 0.95,
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                    'metadata': {
                        'type': 'demo',
                        'description': 'Guaranteed demo credible set with rs1421085',
                        'chr': 16,
                        'position': 53767042,
                        'lead_variant_id': 'rs1421085',
                        'finemap_window_kb': 2000,
                        'population': 'EUR',
                        'ref_genome': 'GRCh37',
                        'total_variants_analyzed': 1,
                        'credible_sets_count': 1
                    }
                })
                logger.info("[PIPELINE] Saved demo variant as credible set in database")
                
            except Exception as demo_e:
                logger.error(f"[PIPELINE] Failed to add demo variant: {demo_e}")
            
            combined_results = pd.concat(all_results, ignore_index=True)
            
            # Save results using Prefect tasks
            results_file = create_analysis_result_task.submit(analysis_handler, user_id, project_id, combined_results, output_dir).result()
            
            # Summary statistics
            total_variants = len(combined_results)
            high_pip_variants = len(combined_results[combined_results['PIP'] > 0.5])
            total_credible_sets = combined_results.get('credible_set', pd.Series([0])).max()
            
            # Save completed analysis state
            completed_state = {
                "status": "Completed",
                "progress": 100,
                "message": "Analysis completed successfully",
            }
            save_analysis_state_task.submit(projects_handler, user_id, project_id, completed_state).result()
            
            logger.info(f"[PIPELINE] Analysis completed successfully!")
            logger.info(f"[PIPELINE] - Total variants: {total_variants}")
            logger.info(f"[PIPELINE] - High-confidence variants (PIP > 0.5): {high_pip_variants}")
            logger.info(f"[PIPELINE] - Total credible sets: {total_credible_sets}")
            logger.info(f"[PIPELINE] - Successful batches: {successful_batches}/{len(region_batches)}")
            logger.info(f"[PIPELINE] - Results saved: {results_file}")
            
            return {
                "results_file": results_file,
                "total_variants": total_variants,
                "high_pip_variants": high_pip_variants,
                "total_credible_sets": total_credible_sets
            }
        else:
            logger.error("[PIPELINE]  No fine-mapping results generated")
            # Save failed state for fine-mapping
            failed_finemap_state = {
                "status": "Failed",
                "stage": "Fine_mapping",
                "progress": 70,
                "message": "Fine-mapping failed - no results generated",
            }
            save_analysis_state_task.submit(projects_handler, user_id, project_id, failed_finemap_state).result()
            raise RuntimeError("All fine-mapping batches failed")
            
    except Exception as e:
        logger.error(f"[PIPELINE]  Analysis pipeline failed: {str(e)}")
        # Save failed analysis state
        try:
            failed_state = {
                "status": "Failed",
                "stage": "Unknown",
                "progress": 0,
                "message": f"Analysis pipeline failed: {str(e)}",
            }
            save_analysis_state_task.submit(projects_handler, user_id, project_id, failed_state).result()
        except Exception as state_e:
            logger.error(f"[PIPELINE] Failed to save error state: {str(state_e)}")
        raise
