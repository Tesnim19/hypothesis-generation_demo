import os
from threading import Thread, Timer
import uuid
from flask import json, request, send_file
from flask_restful import Resource
from flask_socketio import join_room, leave_room
from socketio_instance import socketio
from auth import socket_token_required, token_required
from datetime import datetime, timezone
from uuid import uuid4
from flows import hypothesis_flow, analysis_pipeline_flow
# finemapping_flow
from run_deployment import invoke_enrichment_deployment
from status_tracker import status_tracker, TaskState
from prefect import flow
from utils import allowed_file, convert_variants_to_object_array, parse_float, parse_int
from loguru import logger
from werkzeug.utils import secure_filename
from utils import serialize_datetime_fields
from project_tasks import count_gwas_records, get_project_with_full_data
from bson import ObjectId
import re
from config import Config
import pandas as pd
import re
import gzip
import glob


class EnrichAPI(Resource):
    def __init__(self, enrichr, llm, prolog_query, enrichment, hypotheses, projects, gene_expression=None):
        self.enrichr = enrichr
        self.llm = llm
        self.prolog_query = prolog_query
        self.enrichment = enrichment
        self.hypotheses = hypotheses
        self.projects = projects
        self.gene_expression = gene_expression

    @token_required
    def get(self, current_user_id):
        # Get the enrich_id from the query parameters
        enrich_id = request.args.get('id')
        project_id = request.args.get('project_id')
        
        if enrich_id:
                    # Fetch a specific enrich by enrich_id and user_id
            enrich = self.enrichment.get_enrich(current_user_id, enrich_id)
            if not enrich:
                return {"message": "Enrich not found or access denied."}, 404
            # Serialize datetime objects before returning
            enrich = serialize_datetime_fields(enrich)
            return enrich, 200
        
        if project_id:
            # Get all enrichments for a specific project
            enrichments = self.enrichment.get_enrich(user_id=current_user_id)
            if isinstance(enrichments, list):
                # Filter by project_id if it exists in the enrichment data
                project_enrichments = [e for e in enrichments if e.get('project_id') == project_id]
                project_enrichments = serialize_datetime_fields(project_enrichments)
                return {"enrichments": project_enrichments}, 200
            else:
                # Handle case where get_enrich returns a single item
                if enrichments and enrichments.get('project_id') == project_id:
                    enrichments = serialize_datetime_fields(enrichments)
                    return {"enrichments": [enrichments]}, 200
                return {"enrichments": []}, 200
          
        # Fetch all enrichments for the current user
        enrich = self.enrichment.get_enrich(user_id=current_user_id)
        # Serialize datetime objects before returning
        enrich = serialize_datetime_fields(enrich)
        return enrich, 200

    @token_required
    def post(self, current_user_id):
        json_data = request.get_json(silent=True) or {}
        
        variant =  request.args.get('variant') or json_data.get('variant')
        project_id = request.args.get('project_id') or json_data.get('project_id')
        seed = int(json_data.get('seed', 42))

        
        if not project_id:
            return {"error": "project_id is required"}, 400
        if not variant:
            return {"error": "variant is required"}, 400
        
        # Validate project exists and get phenotype from project
        project = self.projects.get_projects(current_user_id, project_id)
        if not project:
            return {"error": "Project not found or access denied"}, 404
        
        phenotype = project['phenotype']

        tissue_name = (request.args.get('tissue_name') or (json_data.get('tissue_name') if isinstance(json_data, dict) else None))
        if not tissue_name:
            return {"error": "tissue_name is required"}, 400

        # Validate tissue exists for the project and save selection 
        try:
            available_tissues = self.gene_expression.get_ldsc_results_for_project(current_user_id, project_id, limit=20, format='selection')
            tissue_names = [t.get('tissue_name') for t in (available_tissues or [])]
            if tissue_name not in tissue_names:
                return {"error": f"Invalid tissue selection. Available tissues: {tissue_names}"}, 400
            # Persist selection 
            self.gene_expression.save_tissue_selection(
                current_user_id, project_id, variant, tissue_name
            )
            logger.info(f"Saved tissue selection in /enrich: {tissue_name} for variant {variant} in project {project_id}")
        except Exception as e:
            logger.warning(f"Failed to save/validate tissue selection: {e}")
        
        logger.info(f"Project-based enrichment request for project {project_id}")
        
        # Check for existing hypothesis in project context
        existing_hypothesis = self.hypotheses.get_hypothesis_by_phenotype_and_variant_in_project(
            current_user_id, project_id, phenotype, variant
        )
        
        if existing_hypothesis:
            
            logger.info(f"Re-running enrichment for existing hypothesis {existing_hypothesis['id']}")
            invoke_enrichment_deployment(
                current_user_id=current_user_id, 
                phenotype=phenotype, 
                variant=variant, 
                hypothesis_id=existing_hypothesis['id'],
                project_id=project_id,
                seed=seed
            )
            return {"hypothesis_id": existing_hypothesis['id'], "project_id": project_id}, 202
        
        # Generate hypothesis_id and create with project context 
        hypothesis_id = str(uuid4())
        hypothesis_data = {
            "id": hypothesis_id,
            "project_id": project_id,
            "phenotype": phenotype,
            "variant": variant,
            "variant_rsid": variant,  
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds') + "Z",
            "task_history": [],
        }
        
        self.hypotheses.create_hypothesis(current_user_id, hypothesis_data)
        
        invoke_enrichment_deployment(
            current_user_id=current_user_id,
            phenotype=phenotype,
            variant=variant,
            hypothesis_id=hypothesis_id,
            project_id=project_id,
            seed=seed
        )
        
        return {"hypothesis_id": hypothesis_id, "project_id": project_id}, 202
    
         
    @token_required
    def delete(self, current_user_id):
        enrich_id = request.args.get('id')
        if enrich_id:
            result = self.enrichment.delete_enrich(current_user_id, enrich_id)
            return result, 200
        return {"message": "enrich id is required!"}, 400


class HypothesisAPI(Resource):
    def __init__(self, enrichr, prolog_query, llm, hypotheses, enrichment, gene_expression=None):
        self.enrichr = enrichr
        self.prolog_query = prolog_query
        self.llm = llm
        self.hypotheses = hypotheses
        self.enrichment = enrichment
        self.gene_expression = gene_expression
    
    def _extract_probability(self, hypothesis, user_id):
        """Extract probability from hypothesis graph or enrichment data"""
        probability = None
        
        # First try to get from hypothesis graph
        if hypothesis.get("graph") and isinstance(hypothesis["graph"], dict):
            probability = hypothesis["graph"].get("probability")
        
        # If no probability in hypothesis, try to get from enrichment data
        if probability is None and hypothesis.get("enrich_id"):
            try:
                enrich_data = self.enrichment.get_enrich(user_id, hypothesis["enrich_id"])
                if enrich_data and enrich_data.get("causal_graph"):
                    causal_graph = enrich_data["causal_graph"]
                    if isinstance(causal_graph, dict) and causal_graph.get("graph"):
                        graph = causal_graph["graph"]
                        if isinstance(graph, dict):
                            probability = graph.get('prob', {}).get('value') if isinstance(graph.get('prob'), dict) else None
            except Exception as e:
                logger.warning(f"Could not get enrichment data for hypothesis {hypothesis['id']}: {e}")
        
        return probability
    
    def _get_related_hypotheses(self, current_hypothesis, user_id):
        """Get all hypotheses in the same project with the same variant"""
        all_variant_hypotheses = []
        
        project_id = current_hypothesis.get('project_id')
        variant = current_hypothesis.get('variant') or current_hypothesis.get('variant_id')
        
        if not project_id or not variant:
            # If no project/variant info, return just the current hypothesis
            current_probability = self._extract_probability(current_hypothesis, user_id)
            return [{
                'id': current_hypothesis['id'],
                'causal_gene': current_hypothesis.get('causal_gene'),
                'probability': current_probability,
                'status': current_hypothesis.get('status', 'pending'),
                'go_id': current_hypothesis.get('go_id')
            }]
        
        try:
            # Get all hypotheses for the user
            all_hypotheses = self.hypotheses.get_hypotheses(user_id)
            if isinstance(all_hypotheses, list):
                for h in all_hypotheses:
                    # Include ALL hypotheses with same project and variant (including current)
                    if (h.get('project_id') == project_id and 
                        (h.get('variant') == variant or h.get('variant_id') == variant)):
                        
                        probability = self._extract_probability(h, user_id)
                        all_variant_hypotheses.append({
                            'id': h['id'],
                            'causal_gene': h.get('causal_gene'),
                            'probability': probability,
                            'status': h.get('status', 'pending'),
                            'go_id': h.get('go_id')
                        })
                
                # Sort by confidence (highest first) for better UX
                all_variant_hypotheses.sort(key=lambda x: x.get('probability') or 0, reverse=True)
                
        except Exception as e:
            logger.warning(f"Could not get variant hypotheses: {e}")
            # Fallback: return just the current hypothesis
            current_probability = self._extract_probability(current_hypothesis, user_id)
            all_variant_hypotheses = [{
                'id': current_hypothesis['id'],
                'causal_gene': current_hypothesis.get('causal_gene'),
                'probability': current_probability,
                'status': current_hypothesis.get('status', 'pending'),
                'go_id': current_hypothesis.get('go_id')
            }]
        
        return all_variant_hypotheses

    @token_required
    def get(self, current_user_id):
        # Get the hypothesis_id from the query parameters
        hypothesis_id = request.args.get('id')

        if hypothesis_id:
            # Fetch a specific hypothesis by hypothesis_id and user_id
            hypothesis = self.hypotheses.get_hypotheses(current_user_id, hypothesis_id)
            if not hypothesis:
                return {"message": "Hypothesis not found or access denied."}, 404
            
            # Check if enrichment is complete
            required_fields = ['enrich_id', 'go_id', 'summary', 'graph']
            is_complete = all(field in hypothesis for field in required_fields)
            # Get task history
            task_history = status_tracker.get_history(hypothesis_id)
            for task in task_history:
                task.pop('details', None) 

            # Get only pending tasks from task history
            pending_tasks = [task for task in task_history if task.get('state') == TaskState.STARTED.value]
            last_pending_task = [pending_tasks[-1]] if pending_tasks else [] 
            logger.info(f"last_pending_task: {last_pending_task}")

            if is_complete:
                enrich_id = hypothesis.get('enrich_id')
                enrich_data = self.enrichment.get_enrich(current_user_id, enrich_id)
                # Remove 'causal_graph' field from enrich_data if it exists
                if isinstance(enrich_data, dict):
                    enrich_data.pop('causal_graph', None)
                
                # Extract confidence for current hypothesis
                confidence = self._extract_probability(hypothesis, current_user_id)
                
                # Get related hypotheses with same variant in project
                related_hypotheses = self._get_related_hypotheses(hypothesis, current_user_id)
                
                # Log for debugging
                logger.info(f"Hypothesis {hypothesis_id}: confidence={confidence}, variant_hypotheses_count={len(related_hypotheses)}")
                print(f"[HYPOTHESIS_API] Hypothesis {hypothesis_id}: confidence={confidence}, variant_hypotheses_count={len(related_hypotheses)}")
                
                response_data = {
                    'id': hypothesis_id,
                    'variant': hypothesis.get('variant') or hypothesis.get('variant_id'),
                    'enrich_id': enrich_id,
                    'phenotype': hypothesis['phenotype'],
                    "status": "completed",
                    "created_at": hypothesis.get('created_at'),
                    "probability": confidence,
                    "hypotheses": related_hypotheses,
                    "result": enrich_data
                }

                if 'tissue_rankings' in hypothesis:
                    response_data['tissue_rankings'] = hypothesis['tissue_rankings']
                    response_data['enrichment_type'] = hypothesis.get('enrichment_type', 'tissue_enhanced')
                else:
                    response_data['enrichment_type'] = 'standard'

                # Get tissue selection from tissue_selections collection
                selected_tissue = None
                if self.gene_expression:
                    try:
                        variant_id = hypothesis.get('variant_rsid') or hypothesis.get('variant') or hypothesis.get('variant_id')
                        project_id = hypothesis.get('project_id')
                        if variant_id and project_id:
                            tissue_selection = self.gene_expression.get_tissue_selection(
                                current_user_id, project_id, variant_id
                            )
                            if tissue_selection:
                                selected_tissue = tissue_selection.get('tissue_name')
                                logger.info(f"Retrieved tissue selection for completed hypothesis {hypothesis_id} using variant_id={variant_id}: {selected_tissue}")
                            else:
                                logger.info(f"No tissue selection found for completed hypothesis {hypothesis_id}, variant_id={variant_id}")
                    except Exception as ts_e:
                        logger.warning(f"Could not get tissue selection for completed hypothesis {hypothesis_id}: {ts_e}")
                
                response_data['tissue_selected'] = selected_tissue

                # Serialize datetime objects before returning
                response_data = serialize_datetime_fields(response_data)
                return response_data, 200

            latest_state = status_tracker.get_latest_state(hypothesis_id)
            
            # Extract confidence for current hypothesis (even if pending)
            confidence = self._extract_probability(hypothesis, current_user_id)
            
            # Get related hypotheses with same variant in project
            related_hypotheses = self._get_related_hypotheses(hypothesis, current_user_id)
            
            # Log for debugging
            logger.info(f"Pending Hypothesis {hypothesis_id}: confidence={confidence}, variant_hypotheses_count={len(related_hypotheses)}")
            print(f"[HYPOTHESIS_API] Pending Hypothesis {hypothesis_id}: confidence={confidence}, variant_hypotheses_count={len(related_hypotheses)}")
            
            status_data = {
                'id': hypothesis_id,
                'variant': hypothesis.get('variant') or hypothesis.get('variant_id'),
                'phenotype': hypothesis['phenotype'],
                'status': 'pending',
                "created_at": hypothesis.get('created_at'),
                'task_history': last_pending_task,
                "probability": confidence,
                "hypotheses": related_hypotheses,
            }

            if 'tissue_rankings' in hypothesis:
                status_data['tissue_rankings'] = hypothesis['tissue_rankings']
                status_data['causal_gene'] = hypothesis.get('causal_gene')
                status_data['enrichment_stage'] = hypothesis.get('enrichment_stage')
                
                # Check if tissue results are ready but enrichment is still ongoing
                if hypothesis.get('enrichment_stage') == 'tissue_analysis_complete':
                    status_data['tissue_results_ready'] = True
                    
            if 'enrich_id' in hypothesis and hypothesis.get('enrich_id') is not None:
                enrich_id = hypothesis.get('enrich_id')
                status_data['enrich_id'] = enrich_id
                enrich_data = self.enrichment.get_enrich(current_user_id, enrich_id)
                if isinstance(enrich_data, dict):
                    enrich_data.pop('causal_graph', None)
                status_data['result'] = enrich_data
                

            # Check for failed state
            if latest_state and latest_state.get('state') == 'failed':
                status_data['status'] = 'failed'
                status_data['error'] = latest_state.get('error')

            # Get tissue selection from tissue_selections collection
            selected_tissue = None
            if self.gene_expression:
                try:
                    # Use variant_rsid for lookup, fall back to variant if not available
                    variant_id = hypothesis.get('variant_rsid') or hypothesis.get('variant') or hypothesis.get('variant_id')
                    project_id = hypothesis.get('project_id')
                    if variant_id and project_id:
                        tissue_selection = self.gene_expression.get_tissue_selection(
                            current_user_id, project_id, variant_id
                        )
                        if tissue_selection:
                            selected_tissue = tissue_selection.get('tissue_name')
                            logger.info(f"Retrieved tissue selection for pending hypothesis {hypothesis_id} using variant_id={variant_id}: {selected_tissue}")
                        else:
                            logger.info(f"No tissue selection found for pending hypothesis {hypothesis_id}, variant_id={variant_id}")
                except Exception as ts_e:
                    logger.warning(f"Could not get tissue selection for pending hypothesis {hypothesis_id}: {ts_e}")
            
            status_data['tissue_selected'] = selected_tissue

            # Serialize datetime objects before returning
            status_data = serialize_datetime_fields(status_data)
            return status_data, 200

        # Fetch all hypotheses for the current user
        hypotheses = self.hypotheses.get_hypotheses(user_id=current_user_id)
        
        # Filter and format the response for all hypotheses
        formatted_hypotheses = []
        for hypothesis in hypotheses:
            # Get only pending tasks from task history
            pending_tasks = [
                task for task in status_tracker.get_history(hypothesis['id']) 
                if task.get('state') == TaskState.STARTED.value
            ]
            last_pending_task = [pending_tasks[-1]] if pending_tasks else []
            
            formatted_hypothesis = {
                'id': hypothesis['id'],
                'phenotype': hypothesis.get('phenotype', 'Unknown'),
                'variant': hypothesis.get('variant') or hypothesis.get('variant_id'),
                'created_at': hypothesis.get('created_at'),
                'status': hypothesis.get('status'),
                'task_history': last_pending_task
            }
            
            if 'enrich_id' in hypothesis and hypothesis.get('enrich_id') is not None:
                 formatted_hypothesis['enrich_id'] = hypothesis.get('enrich_id')
            if 'biological_context' in hypothesis and hypothesis.get('biological_context') is not None:
                formatted_hypothesis['biological_context'] = hypothesis.get('biological_context')
            if 'causal_gene' in hypothesis and hypothesis.get('causal_gene') is not None:
                formatted_hypothesis['causal_gene'] = hypothesis.get('causal_gene')
                
            formatted_hypotheses.append(formatted_hypothesis)
        
        # Serialize datetime objects before returning
        formatted_hypotheses = serialize_datetime_fields(formatted_hypotheses)
        return formatted_hypotheses, 200
        

    @token_required
    def post(self, current_user_id):
        enrich_id = request.args.get('id')
        go_id = request.args.get('go')

        # Get the hypothesis associated with this enrichment
        hypothesis = self.hypotheses.get_hypothesis_by_enrich(current_user_id, enrich_id)
        if not hypothesis:
            return {"message": "No hypothesis found for this enrichment"}, 404
        
        hypothesis_id = hypothesis['id']
        
        # Run the Prefect flow and return the result
        flow_result = hypothesis_flow(current_user_id, hypothesis_id, enrich_id, go_id, self.hypotheses, self.prolog_query, self.llm, self.enrichment)

        return flow_result[0], flow_result[1]

    
    @token_required
    def delete(self, current_user_id):
        hypothesis_id = request.args.get('hypothesis_id')
        if hypothesis_id:
            return self.hypotheses.delete_hypothesis(current_user_id, hypothesis_id)
        return {"message": "Hypothesis ID is required"}, 400
        
    
class BulkHypothesisDeleteAPI(Resource):
    def __init__(self, hypotheses):
        self.hypotheses = hypotheses
        
    @token_required
    def post(self, current_user_id):
        data = request.get_json()
        
        if not data or 'hypothesis_ids' not in data:
            return {"message": "hypothesis_ids is required in request body"}, 400
            
        hypothesis_ids = data.get('hypothesis_ids')
        
        # Validate the list of IDs
        if not isinstance(hypothesis_ids, list):
            return {"message": "hypothesis_ids must be a list"}, 400
            
        if not hypothesis_ids:
            return {"message": "hypothesis_ids list cannot be empty"}, 400
            
        # Call the bulk delete method
        result, status_code = self.hypotheses.bulk_delete_hypotheses(current_user_id, hypothesis_ids)

        return result, status_code

class ChatAPI(Resource):
    def __init__(self, llm, hypotheses):
        self.llm = llm
        self.hypotheses = hypotheses

    @token_required
    def post(self, current_user_id):
        query = request.form.get('query')
        hypothesis_id = request.form.get('hypothesis_id')

        hypothesis = self.hypotheses.get_hypotheses(current_user_id, hypothesis_id)
        print(f"Hypothesis: {hypothesis}")
        
        if not hypothesis:
            return {"error": "Hypothesis not found or access denied"}, 404
        
        graph = hypothesis.get('graph')
        response = self.llm.chat(query, graph)
        response = {"response": response}
        return response

def init_socket_handlers(hypotheses_handler):
    logger.info("Initializing socket handlers...")
    
    # Store active timers for cleanup
    client_timers = {}
    
    @socketio.on('connect')
    def handle_connect(auth=None):

        try:
            logger.info("Client attempting to connect")
            client_id = request.sid
            
            # Log authentication data for debugging
            if auth:
                logger.info(f"Auth data received: {type(auth)}")
            
            # Check for authorization header
            auth_header = request.headers.get('Authorization')
            if auth_header:
                logger.info(f"Authorization header present: {auth_header[:20]}...")
            
            logger.info(f"Client connected: {client_id}")
            
            # Set a reasonable timeout for all connections (extended for better UX)
            inactivity_timer = Timer(600, lambda: socketio.server.disconnect(client_id))  # 10 minutes
            inactivity_timer.start()
            
            # Store the timer for potential cleanup
            client_timers[client_id] = inactivity_timer
            
            return True
                
        except Exception as e:
            logger.error(f"Error in handle_connect: {str(e)}")
            return False

    @socketio.on('disconnect')
    def handle_disconnect():
        try:
            client_id = request.sid
            logger.info(f"Client disconnected: {client_id}")
            
            # Clean up any rooms the client was in
            try:
                # Get all rooms for this client
                rooms = socketio.server.rooms(client_id)
                if rooms:
                    logger.info(f"Cleaning up rooms for client {client_id}: {list(rooms)}")
                    # Leave all rooms
                    for room in rooms:
                        if room != client_id:  
                            leave_room(room)
                            logger.info(f"Client {client_id} left room: {room}")
            except Exception as room_e:
                logger.warning(f"Could not clean up rooms for {client_id}: {room_e}")
            
            # Cancel any pending timers for this client
            if client_id in client_timers:
                try:
                    client_timers[client_id].cancel()
                    del client_timers[client_id]
                    logger.info(f"Cancelled timeout timer for client: {client_id}")
                except Exception as timer_e:
                    logger.warning(f"Could not cancel timer for {client_id}: {timer_e}")
            
        except Exception as e:
            logger.error(f"Error in handle_disconnect: {str(e)}")

    @socketio.on('task_update')
    @socket_token_required  
    def handle_task_update(data, current_user_id=None):
        """
        Handle task updates from Prefect clients and broadcast to appropriate rooms
        """
        try:
            logger.info(f"Received task update from client: {data}")
            
            target_room = data.get('target_room')
            if not target_room:
                logger.error("No target_room specified in task update")
                return
            
            # Remove the target_room from data before broadcasting
            broadcast_data = {k: v for k, v in data.items() if k != 'target_room'}
            
            # Broadcast to the specific room
            socketio.emit('task_update', broadcast_data, room=target_room)
            logger.info(f"Broadcasted task update to room: {target_room}")
            
        except Exception as e:
            logger.error(f"Error handling task update: {str(e)}")

    @socketio.on('subscribe_hypothesis')
    @socket_token_required
    def handle_subscribe(data, current_user_id=None):
        try:
            logger.info(f"Received subscribe request: {data}")
            # Handle both string and dict input
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON string: {data}")
                    return {'error': 'Invalid JSON format'}, 400
            
            # Validate input
            if not isinstance(data, dict) or 'hypothesis_id' not in data:
                logger.error(f"Invalid data format: {data}")
                return {'error': 'Expected format: {"hypothesis_id": "value"}'}, 400
                
            hypothesis_id = data.get('hypothesis_id')
            if not hypothesis_id:
                logger.error("Missing hypothesis_id")
                return {'error': 'hypothesis_id is required'}, 400
            
            response_data = {
                'hypothesis_id': hypothesis_id,
                'timestamp': datetime.now(timezone.utc).isoformat(timespec='milliseconds') + "Z",
            }
            # Join a room specific to this hypothesis
            room = f"hypothesis_{hypothesis_id}"    
            join_room(room)
            logger.info(f"Joined room: {room}")

            # Verify room membership
            try:
                rooms = socketio.server.rooms(request.sid)
                logger.info(f"Current rooms for client {request.sid}: {rooms}")
            except Exception as room_e:
                logger.warning(f"Could not verify room membership: {room_e}")
            
           
            if current_user_id is None:
                hypothesis = hypotheses_handler.get_hypothesis_by_id(hypothesis_id)
            else:
                hypothesis = hypotheses_handler.get_hypotheses(current_user_id, hypothesis_id)
                
            if not hypothesis:
                logger.error(f"Hypothesis not found: {hypothesis_id}")
                raise ValueError("Hypothesis not found or access denied")

            # Get task history
            task_history = status_tracker.get_history(hypothesis_id)
            response_data['task_history'] = task_history
                
            # Check if hypothesis is complete
            required_fields = ['enrich_id', 'go_id', 'summary', 'graph']
            is_complete = all(field in hypothesis for field in required_fields)  
            
            if is_complete:
                response_data.update({
                    'status': 'completed',
                    'result': hypothesis,
                    'progress': 100
                })
            else:
                latest_state = status_tracker.get_latest_state(hypothesis_id)
                progress = status_tracker.calculate_progress(task_history)
                current_task = latest_state['task'] if latest_state else None
                error = latest_state.get('error') if latest_state and latest_state.get('state') == TaskState.FAILED else None
                
                response_data.update({
                    'status': 'pending',
                    'progress': progress
                })

                if 'tissue_rankings' in hypothesis:
                    response_data['tissue_rankings'] = hypothesis['tissue_rankings']
                    response_data['tissue_results_ready'] = True
                    response_data['causal_gene'] = hypothesis.get('causal_gene')
                    response_data['enrichment_stage'] = hypothesis.get('enrichment_stage')

            if current_task:
                response_data['current_task'] = current_task
            if error:
                response_data['error'] = error
            
            
            logger.info(f"Emitting task_update: {response_data}")
            socketio.emit('task_update', response_data, room=room)
            return {"status": "subscribed", "room": room}
            
        except Exception as e:
            logger.error(f"Error in handle_subscribe: {str(e)}")
            return {"error": str(e)}, 500

class ProjectsAPI(Resource):
    """
    API endpoint for managing projects
    """
    def __init__(self, projects, files, analysis, hypotheses, enrichment, gene_expression):
        self.projects = projects
        self.files = files
        self.analysis = analysis
        self.hypotheses = hypotheses
        self.enrichment = enrichment
        self.gene_expression = gene_expression
    
    @token_required
    def get(self, current_user_id):
        """Get all projects for a user or comprehensive data for a specific project"""
        project_id = request.args.get('id')
        
        if project_id:
            response_data, status_code = get_project_with_full_data(
                self.projects, self.analysis, self.hypotheses, self.enrichment, 
                current_user_id, project_id, gene_expression_handler=self.gene_expression
            )
            if status_code == 200:
                response_data = serialize_datetime_fields(response_data)
            return response_data, status_code
        
        projects = self.projects.get_projects(current_user_id)
        enhanced_projects = []
        
        for project in projects:
            enhanced_project = {
                "id": project["id"],
                "name": project["name"],
                "phenotype": project.get("phenotype", ""),
                "created_at": project.get("created_at"),
            }
            
            # Add GWAS file information using files handler
            try:
                file_metadata = self.files.get_file_metadata(current_user_id, project["gwas_file_id"])
                if file_metadata:
                    # Use stored download URL or generate fallback
                    enhanced_project["gwas_file"] = file_metadata.get("download_url", f"/download/{project['gwas_file_id']}")
                    
                    # Use stored record count or calculate fallback
                    enhanced_project["gwas_records_count"] = file_metadata.get("record_count", 0)
                    
                    # If record count is missing, calculate and update it
                    if enhanced_project["gwas_records_count"] == 0:
                        gwas_records_count = count_gwas_records(file_metadata["file_path"])
                        enhanced_project["gwas_records_count"] = gwas_records_count
                        # Update the database with calculated count
                        self.files.file_metadata_collection.update_one(
                            {'_id': ObjectId(project["gwas_file_id"])},
                            {'$set': {'record_count': gwas_records_count}}
                        )
                else:
                    enhanced_project["gwas_file"] = None
                    enhanced_project["gwas_records_count"] = 0
            except Exception as file_e:
                logger.warning(f"Could not load file metadata for project {project['id']}: {file_e}")
                enhanced_project["gwas_file"] = None
                enhanced_project["gwas_records_count"] = 0
            
            # Add analysis status
            try:
                analysis_state = self.projects.load_analysis_state(current_user_id, project["id"])
                if analysis_state:
                    enhanced_project["status"] = analysis_state.get("status", "Not_started")
                else:
                    enhanced_project["status"] = "Not_started"  # Default for projects without analysis state
            except Exception as state_e:
                logger.warning(f"Could not load analysis state for project {project['id']}: {state_e}")
                enhanced_project["status"] = "Completed"
            
            # Get analysis parameters from project (stored during creation)
            enhanced_project["population"] = project.get("population")
            enhanced_project["ref_genome"] = project.get("ref_genome")
            
            # Extract credible sets and variants counts
            total_credible_sets_count = 0
            total_variants_count = 0
            
            try:
                credible_sets_raw = self.analysis.get_credible_sets_for_project(current_user_id, project["id"])
                if credible_sets_raw:
                    if isinstance(credible_sets_raw, list) and credible_sets_raw:
                        # Calculate totals from credible sets
                        total_credible_sets_count = len(credible_sets_raw) if credible_sets_raw else 0
                        total_variants_count = sum(cs.get("variants_count", 0) for cs in credible_sets_raw) if credible_sets_raw else 0
    
            except Exception as cs_e:
                logger.warning(f"Could not load credible sets for project {project['id']}: {cs_e}")
            
            # Add counts to project
            enhanced_project["total_credible_sets_count"] = total_credible_sets_count
            enhanced_project["total_variants_count"] = total_variants_count
            
            # Count hypotheses for this project
            hypothesis_count = 0
            try:
                all_hypotheses = self.hypotheses.get_hypotheses(current_user_id)
                if isinstance(all_hypotheses, list):
                    hypothesis_count = len([h for h in all_hypotheses if h.get('project_id') == project["id"]])
                elif all_hypotheses and all_hypotheses.get('project_id') == project["id"]:
                    hypothesis_count = 1
            except Exception as hyp_e:
                logger.warning(f"Could not count hypotheses for project {project['id']}: {hyp_e}")
            
            enhanced_project["hypothesis_count"] = hypothesis_count
            
            enhanced_projects.append(enhanced_project)
        
        # Serialize datetime objects in all projects
        enhanced_projects = serialize_datetime_fields(enhanced_projects)
        return {"projects": enhanced_projects}, 200
    
    @token_required
    def post(self, current_user_id):
        """Create a new project"""
        data = request.get_json()
        
        if not data or 'name' not in data or 'gwas_file_id' not in data or 'phenotype' not in data:
            return {"error": "Missing required fields: name, gwas_file_id, phenotype"}, 400
        
        try:
            project_id = self.projects.create_project(
                current_user_id, 
                data['name'], 
                data['gwas_file_id'],
                data['phenotype'],
                population=data.get('population'),
                ref_genome=data.get('ref_genome'),
                analysis_parameters=data.get('analysis_parameters')
            )
            
            project = self.projects.get_projects(current_user_id, project_id)
            # Serialize datetime objects before returning
            project = serialize_datetime_fields(project)
            return {"project": project}, 201
        except Exception as e:
            return {"error": f"Failed to create project: {str(e)}"}, 500
    
    @token_required
    def delete(self, current_user_id):
        """Delete a project"""
        project_id = request.args.get('id')
        if not project_id:
            return {"error": "Project ID is required"}, 400
        
        success = self.projects.delete_project(current_user_id, project_id)
        if success:
            return {"message": "Project deleted successfully"}, 200
        return {"error": "Project not found or access denied"}, 404
class BulkProjectDeleteAPI(Resource):
    def __init__(self, projects):
        self.projects = projects
        
    @token_required
    def post(self, current_user_id):
        data = request.get_json()
        
        if not data or 'project_ids' not in data:
            return {"message": "project_ids is required in request body"}, 400
            
        project_ids = data.get('project_ids')
        
        # Validate the list of IDs
        if not isinstance(project_ids, list):
            return {"message": "project_ids must be a list"}, 400
            
        if not project_ids:
            return {"message": "project_ids list cannot be empty"}, 400
            
        # Call the bulk delete method
        result = self.projects.bulk_delete_projects(current_user_id, project_ids)
        
        if result and isinstance(result, dict):
            if result['success']:
                return {
                    "message": f"Successfully deleted {result['deleted_count']} project(s)",
                    "deleted_count": result['deleted_count'],
                    "total_requested": result['total_requested']
                }, 200
            else:
                return {
                    "message": f"Partially deleted {result['deleted_count']}/{result['total_requested']} project(s)",
                    "deleted_count": result['deleted_count'],
                    "total_requested": result['total_requested'],
                    "errors": result['errors']
                }, 207  # Multi-status
        else:
            return {"error": "Failed to delete projects"}, 500

class AnalysisPipelineAPI(Resource):
    def __init__(self, projects, files, analysis, gene_expression, config):
        self.projects = projects
        self.files = files
        self.analysis = analysis
        self.gene_expression = gene_expression
        self.config = config

    @token_required
    def post(self, current_user_id):
        try:
            # Get form data and file
            project_name = request.form.get('project_name')
            phenotype = request.form.get('phenotype')
            ref_genome = request.form.get('ref_genome', 'GRCh37')
            population = request.form.get('population', 'EUR')
            
            max_workers = parse_int(request.form.get('max_workers'), 3, 'max_workers')
            
            # Check the mode: uploaded file or predefined file
            is_uploaded = request.form.get('is_uploaded', 'false').lower() == 'true'
            
            # Fine-mapping parameters with defaults
            maf_threshold = parse_float(request.form.get('maf_threshold'), 0.01, 'maf_threshold')
            seed = parse_int(request.form.get('seed'), 42, 'seed')
            window = parse_int(request.form.get('window'), 2000, 'window')
            L = parse_int(request.form.get('L'), -1, 'L')
            coverage = parse_float(request.form.get('coverage'), 0.95, 'coverage')
            min_abs_corr = parse_float(request.form.get('min_abs_corr'), 0.5, 'min_abs_corr')
            batch_size = parse_int(request.form.get('batch_size'), 5, 'batch_size')
            
            # Validate required fields
            if not project_name:
                return {"error": "project_name is required"}, 400
            
            if not phenotype:
                return {"error": "phenotype is required"}, 400
            
            # Handle based on the upload flag
            if not is_uploaded:
                # Mode: Predefined file - gwas_file contains the file ID from dropdown
                predefined_file_id = request.form.get('gwas_file')  # File ID as string
                
                if not predefined_file_id:
                    return {"error": "gwas_file is required (file ID for predefined files)"}, 400
                
                logger.info(f"[API] Using predefined GWAS file ID: {predefined_file_id}")
                
                # Find the predefined file in data/raw/
                data_dir = getattr(self.config, 'data_dir', 'data')
                raw_data_path = os.path.join(data_dir, 'raw')
                
                # Look for the file with various possible extensions since ID doesn't include extension
                possible_extensions = ['.tsv', '.tsv.gz', '.tsv.bgz', '.txt', '.txt.gz', '.csv', '.csv.gz']
                gwas_file_path = None
                filename = None
                
                for ext in possible_extensions:
                    candidate_path = os.path.join(raw_data_path, f"{predefined_file_id}{ext}")
                    if os.path.exists(candidate_path):
                        gwas_file_path = candidate_path
                        filename = f"{predefined_file_id}{ext}"
                        break
                
                if not gwas_file_path:
                    return {"error": f"Predefined GWAS file not found for ID: {predefined_file_id}"}, 404
                
                logger.info(f"[API] Found predefined file: {filename} at {gwas_file_path}")
                file_size = os.path.getsize(gwas_file_path)
                file_id = str(uuid.uuid4())
                
            else:
                # Mode: Uploaded file - gwas_file contains the actual file
                if 'gwas_file' not in request.files:
                    return {"error": "No GWAS file uploaded"}, 400
                
                gwas_file = request.files['gwas_file']
                if gwas_file.filename == '':
                    return {"error": "No file selected"}, 400
            
            logger.info(f"[API] Starting analysis pipeline")
            logger.info(f"[API] Project: {project_name}")
            logger.info(f"[API] Phenotype: {phenotype}")
            if not is_uploaded:
                logger.info(f"[API] Predefined file: {filename} (ID: {predefined_file_id})")
            else:
                logger.info(f"[API] Uploaded file: {gwas_file.filename}")
            logger.info(f"[API] Reference: {ref_genome}, Population: {population}")
            logger.info(f"[API] Fine-mapping params: maf={maf_threshold}, seed={seed}, window={window}kb, L={L}, coverage={coverage}, min_abs_corr={min_abs_corr}")
            
            # Validate file (only for uploaded files)
            if is_uploaded and not allowed_file(gwas_file.filename):
                return {"error": "Invalid file format. Supported: .tsv, .txt, .csv, .gz, .bgz"}, 400
            
            # Validate parameters
            if ref_genome not in ["GRCh37", "GRCh38"]:
                return {"error": "Reference genome must be GRCh37 or GRCh38"}, 400
            
            if population not in ["EUR", "AFR", "AMR", "EAS", "SAS"]:
                return {"error": "Population must be one of: EUR, AFR, AMR, EAS, SAS"}, 400
            
            if max_workers < 1 or max_workers > 16:
                return {"error": "Max workers must be between 1-16"}, 400
            
            # Validate fine-mapping parameters
            if maf_threshold < 0.001 or maf_threshold > 0.5:
                return {"error": "MAF threshold must be between 0.001-0.5"}, 400
            
            if seed < 1 or seed > 999999:
                return {"error": "Seed must be between 1-999999"}, 400
            
            if window > 10000:
                return {"error": "Fine-mapping window shouldn't be greater than 10000 kb"}, 400
            
            if L != -1 and (L < 1 or L > 50):
                return {"error": "L must be -1 (auto) or between 1-50"}, 400
            
            if coverage < 0.5 or coverage > 0.999:
                return {"error": "Coverage must be between 0.5-0.999"}, 400
            
            if min_abs_corr < 0.5 or min_abs_corr > 1.0:
                return {"error": "Min absolute correlation must be between 0.5-1.0"}, 400
            
            if batch_size < 1 or batch_size > 20:
                return {"error": "Batch size must be between 1-20"}, 400
            
            # === FILE HANDLING AND PROJECT CREATION  ===
            if not is_uploaded:
                # Use predefined file (already set gwas_file_path, filename, file_size above)
                logger.info(f"Using predefined GWAS file: {filename} (Path: {gwas_file_path})")
                file_path = gwas_file_path
                gwas_records_count = count_gwas_records(file_path)
            else:
                # Handle uploaded file
                # Generate secure filename and file ID
                filename = secure_filename(gwas_file.filename)
                file_id = str(uuid.uuid4())
                
                # Create user upload directory
                user_upload_dir = os.path.join('data', 'uploads', str(current_user_id))
                os.makedirs(user_upload_dir, exist_ok=True)
                
                # File path for saving
                file_path = os.path.join(user_upload_dir, f"{file_id}_{filename}")
                
                logger.info(f"Starting upload for file {filename} (ID: {file_id})")
                start_time = datetime.now()
                
                # Save file
                gwas_file.save(file_path)
                file_size = os.path.getsize(file_path)
                gwas_file_path = file_path
                
                gwas_records_count = count_gwas_records(file_path)
            
            # Create file metadata in database with record count through files handler
            if is_uploaded:
                original_filename = gwas_file.filename
            else:
                original_filename = filename
            file_metadata_id = self.files.create_file_metadata(
                user_id=current_user_id,
                filename=filename,
                original_filename=original_filename,
                file_path=file_path,
                file_type='gwas',
                file_size=file_size,
                record_count=gwas_records_count,
                download_url=f"/download/{str(uuid.uuid4())}"
            )
            
            # Prepare analysis parameters
            analysis_parameters = {
                'maf_threshold': maf_threshold,
                'seed': seed,
                'window': window,
                'L': L,
                'coverage': coverage,
                'min_abs_corr': min_abs_corr,
                'batch_size': batch_size,
                'max_workers': max_workers
            }
            
            # Create project with analysis parameters
            project_id = self.projects.create_project(
                user_id=current_user_id,
                name=project_name,
                gwas_file_id=file_metadata_id,
                phenotype=phenotype,
                population=population,
                ref_genome=ref_genome,
                analysis_parameters=analysis_parameters
            )
            
            # Save metadata to file system
            metadata_dir = os.path.join('data', 'metadata', str(current_user_id))
            os.makedirs(metadata_dir, exist_ok=True)
            
            metadata = {
                'file_id': file_metadata_id,
                'user_id': current_user_id,
                'filename': filename,
                'original_filename': original_filename,
                'file_path': file_path,
                'file_type': 'gwas',
                'upload_date': str(datetime.now()),
                'file_size': file_size,
                'project_id': project_id
            }
            
            with open(os.path.join(metadata_dir, f"{file_metadata_id}.json"), 'w') as f:
                json.dump(metadata, f)
            
            if is_uploaded:
                total_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Completed: {filename} in {total_time:.1f} seconds")
            else:
                logger.info(f"Using predefined file: {filename}")
            
            logger.info(f"[API] Created project {project_id} with file {file_metadata_id}")
            
            # Start pipeline in background thread
            def run_pipeline_background(
                proj_id=project_id,
                user_id=current_user_id,
                gwas_path=file_path,
                ref_gen=ref_genome,
                pop=population,
                batch=batch_size,
                workers=max_workers,
                maf=maf_threshold,
                seed_val=seed,
                win=window,
                L_val=L,
                cov=coverage,
                min_corr=min_abs_corr
            ):
                try:
                    logger.info(f"[API] Running analysis pipeline for project {proj_id}")
                    
                    # Run the analysis pipeline flow directly
                    credible_sets = analysis_pipeline_flow(
                        projects_handler=self.projects,
                        analysis_handler=self.analysis,
                        gene_expression=self.gene_expression,
                        mongodb_uri=self.config.mongodb_uri,
                        db_name=self.config.db_name,
                        user_id=user_id,
                        project_id=proj_id,
                        gwas_file_path=gwas_path,
                        ref_genome=ref_gen,
                        population=pop,
                        batch_size=batch,
                        max_workers=workers,
                        maf_threshold=maf,
                        seed=seed_val,
                        window=win,
                        L=L_val,
                        coverage=cov,
                        min_abs_corr=min_corr
                    )
                    
                    logger.info(f"[API] Analysis pipeline for project {proj_id} completed successfully")
                    if credible_sets and isinstance(credible_sets, dict):
                        logger.info(f"[API] Generated {credible_sets.get('total_variants', 0)} variants in {credible_sets.get('total_credible_sets', 0)} credible sets")
                    else:
                        logger.info(f"[API] Analysis completed but no credible sets generated")
                    
                except Exception as e:
                    logger.error(f"[API] Analysis pipeline for project {proj_id} failed: {str(e)}")
            
            # Start background thread
            background_thread = Thread(target=run_pipeline_background)
            background_thread.start()
            
            logger.info(f"[API] Analysis pipeline started for project {project_id}")
            
            return {
                "status": "started",
                "project_id": project_id,
                "file_id": file_metadata_id,
                "message": "Analysis pipeline started successfully",
            }, 202
            
        except Exception as e:
            logger.error(f"[API] Error starting analysis pipeline: {str(e)}")
            return {"error": f"Error starting analysis pipeline: {str(e)}"}, 500

class LDSCResultsAPI(Resource):
    """API endpoint for getting LDSC tissue analysis results"""
    
    def __init__(self, gene_expression, projects):
        self.gene_expression = gene_expression
        self.projects = projects
    
    @token_required
    def get(self, current_user_id):
        """Get LDSC results for a project"""
        project_id = request.args.get('project_id')
        
        if not project_id:
            return {"error": "project_id is required"}, 400
        
        try:
            # Validate project access
            project = self.projects.get_projects(current_user_id, project_id)
            if not project:
                return {"error": "Project not found or access denied"}, 404
            
            # Get LDSC analysis status and results
            status = self.gene_expression.check_gene_expression_status(project_id, current_user_id)
            
            response_data = {
                "project_id": project_id,
                "ldsc_status": status['status'],
                "has_results": status['has_data'],
                "analysis_date": status.get('created_at')
            }
            
            if status['has_data']:
                # Get detailed LDSC results
                ldsc_results = self.gene_expression.get_gene_expression_results(
                    project_id=project_id,
                    user_id=current_user_id
                )
                
                if ldsc_results and ldsc_results[0]['ldsc_results']:
                    tissue_results = ldsc_results[0]['ldsc_results']
                    
                    response_data.update({
                        "total_tissues": len(tissue_results),
                        "significant_tissues": len([t for t in tissue_results if t.get('p_value', 1) < 0.05]),
                        "top_10_tissues": tissue_results[:10],
                        "all_tissues": tissue_results
                    })
                else:
                    response_data["tissues"] = []
            
            return serialize_datetime_fields(response_data), 200
            
        except Exception as e:
            logger.error(f"Error getting LDSC results: {str(e)}")
            return {"error": f"Error retrieving LDSC results: {str(e)}"}, 500

 

class FileDownloadAPI(Resource):
    """
    API endpoint for downloading files
    """
    def __init__(self, files):
        self.files = files
    
    @token_required
    def get(self, current_user_id, file_id):
        """Download a file by file_id"""
        try:
            logger.info(f"[DOWNLOAD] Download request for file {file_id} by user {current_user_id}")
            
            # Get file metadata
            file_metadata = self.files.get_file_metadata(current_user_id, file_id)
            if not file_metadata:
                logger.warning(f"[DOWNLOAD] File metadata not found for file {file_id}")
                return {"error": "File not found or access denied"}, 404
            
            # Check if file exists on disk
            file_path = file_metadata['file_path']
            if not os.path.exists(file_path):
                logger.error(f"[DOWNLOAD] File not found on disk: {file_path}")
                return {"error": "File not found on disk"}, 404
            
            # Get original filename for download
            original_filename = file_metadata.get('original_filename', file_metadata.get('filename', 'download'))
            
            logger.info(f"[DOWNLOAD] Serving file: {original_filename} (Path: {file_path})")
            
            # Return file for download
            return send_file(
                file_path,
                as_attachment=True,
                download_name=original_filename,
                mimetype='application/octet-stream'  # Generic binary file type
            )
            
        except Exception as e:
            logger.error(f"[DOWNLOAD] Error downloading file {file_id}: {str(e)}")
            return {"error": f"Download failed: {str(e)}"}, 500

class CredibleSetsAPI(Resource):
    """
    API endpoint for fetching credible sets
    """
    def __init__(self, analysis):
        self.analysis = analysis

    @token_required
    def get(self, current_user_id):
        """Get credible set details by credible set ID or lead variant ID"""
        project_id = request.args.get('project_id')
        credible_set_id = request.args.get('credible_set_id')
        
        if not project_id:
            return {"error": "project_id is required"}, 400
        
        if not credible_set_id:
            return {"error": "Credible_set_id is required"}, 400
        
        try:
            credible_set = self.analysis.get_credible_set_by_id(current_user_id, project_id, credible_set_id)
            if not credible_set:
                return {"message": "No credible set found with this ID"}, 404
            
            # Extract variants data 
            variants_data = credible_set.get("variants_data", {})
            if not variants_data:
                return {"message": "No variants data found for this credible set"}, 404
            
            variants = variants_data.get("data", {})
            
            # Convert from object-with-arrays format to array-of-objects format
            variants_array = convert_variants_to_object_array(variants)

            variants_array = serialize_datetime_fields(variants_array)
            return {
                "variants": variants_array
            }, 200

        except Exception as e:
            logger.error(f"Error fetching credible set: {str(e)}")
            return {"error": f"Failed to fetch credible set: {str(e)}"}, 500


class GWASFilesAPI(Resource):
    """
    API endpoint for automatically discovering GWAS files and extracting their metadata
    """
    def __init__(self, config, phenotypes):
        self.config = config
        self.phenotypes = phenotypes

    def _extract_file_metadata(self, file_path):
        """Extract metadata from a GWAS file by examining its content"""
        
        filename = os.path.basename(file_path)
        
        try:
            # Extract phenotype ID from filename
            phenotype_id = None
            match = re.match(r'^(\d+)_', filename)
            if match:
                phenotype_id = match.group(1)
            
            # Determine file type and how to open it
            if file_path.endswith('.bgz'):
                # Use gzip for bgz files
                with gzip.open(file_path, 'rt') as f:
                    header = f.readline().strip().split('\t')
                    # Read a few lines to get sample info
                    lines = [f.readline().strip() for _ in range(3)]
            elif file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt') as f:
                    header = f.readline().strip().split('\t')
                    lines = [f.readline().strip() for _ in range(3)]
            else:
                with open(file_path, 'r') as f:
                    header = f.readline().strip().split('\t')
                    lines = [f.readline().strip() for _ in range(3)]
            
            # Extract sample size from n_complete_samples column if available
            sample_size = "Unknown"
            if 'n_complete_samples' in header:
                try:
                    n_idx = header.index('n_complete_samples')
                    for line in lines:
                        if line:
                            fields = line.split('\t')
                            if len(fields) > n_idx and fields[n_idx]:
                                sample_size = f"~{int(float(fields[n_idx])):,}"
                                break
                except (ValueError, IndexError):
                    pass
            
            # Extract population from filename patterns
            population = "Unknown"
            if "both_sexes" in filename:
                population = "Mixed"
            elif "male" in filename:
                population = "Male"
            elif "female" in filename:
                population = "Female"
            elif re.search(r'(eur|european)', filename.lower()):
                population = "EUR"
            elif re.search(r'(afr|african)', filename.lower()):
                population = "AFR"
            elif re.search(r'(eas|east_asian)', filename.lower()):
                population = "EAS"
            elif re.search(r'(sas|south_asian)', filename.lower()):
                population = "SAS"
            elif re.search(r'(amr|american)', filename.lower()):
                population = "AMR"
            else:
                # Default to EUR for UK Biobank files
                if phenotype_id and phenotype_id.startswith(('2', '50', '23')):
                    population = "EUR"
            
            # Determine genome build
            genome_build = "GRCh37"  # Default
            if "hg38" in filename.lower() or "grch38" in filename.lower():
                genome_build = "GRCh38"
            
            # Determine if it's raw or processed
            is_raw = "_raw" in filename or (not "_munged" in filename and not "_processed" in filename)
            
            # Get basic phenotype name from known mappings or use ID
            phenotype_name = self._get_phenotype_name(phenotype_id) if phenotype_id else "Unknown"
            
            return {
                "phenotype_id": phenotype_id,
                "phenotype_name": phenotype_name,
                "population": population,
                "sample_size": sample_size,
                "genome_build": genome_build,
                "is_raw": is_raw,
                "header_columns": header,
                "file_size": os.path.getsize(file_path)
            }
            
        except Exception as e:
            logger.warning(f"Could not extract metadata from {filename}: {str(e)}")
            return {
                "phenotype_id": phenotype_id,
                "phenotype_name": f"Phenotype {phenotype_id}" if phenotype_id else "Unknown",
                "population": "Unknown",
                "sample_size": "Unknown",
                "genome_build": "Unknown",
                "is_raw": "_raw" in filename,
                "header_columns": [],
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }

    def _get_phenotype_name(self, phenotype_id):
        """Get phenotype name from known UK Biobank field IDs"""
        # Known UK Biobank phenotype mappings
        known_phenotypes = {
            "21001": "Body Mass Index (BMI)",
            "50": "Standing Height",
            "23104": "Body Fat Percentage", 
            "21002": "Weight",
            "23105": "Basal Metabolic Rate",
            "23106": "Impedance",
            "23107": "Arm Fat-free Mass",
            "23108": "Arm Fat Mass",
            "23109": "Arm Predicted Mass",
            "48": "Waist Circumference",
            "49": "Hip Circumference"
        }
        return known_phenotypes.get(phenotype_id, f"Phenotype {phenotype_id}")

    def _generate_file_description(self, filename, metadata):
        """Generate description based on file metadata"""
        
        phenotype_name = metadata['phenotype_name']
        
        if "_sig" in filename:
            return f"Genome-wide significant variants for {phenotype_name}"
        elif "_chr" in filename:
            chr_match = re.search(r'_chr(\d+)', filename)
            chr_num = chr_match.group(1) if chr_match else "X"
            return f"{phenotype_name} variants on chromosome {chr_num}"

    def get(self):
        """Automatically discover GWAS files and extract their metadata"""
        try:
            
            # Get data directory
            data_dir = getattr(self.config, 'data_dir', 'data')
            raw_data_path = os.path.join(data_dir, 'raw')
            
            if not os.path.exists(raw_data_path):
                return {"gwas_files": [], "total_files": 0}, 200
            
            # Scan for all potential GWAS files
            patterns = [
                os.path.join(raw_data_path, '*.tsv'),
                os.path.join(raw_data_path, '*.tsv.gz'),
                os.path.join(raw_data_path, '*.tsv.bgz'),
                os.path.join(raw_data_path, '*.txt'),
                os.path.join(raw_data_path, '*.txt.gz'),
                os.path.join(raw_data_path, '*.csv'),
                os.path.join(raw_data_path, '*.csv.gz')
            ]
            
            all_files = []
            for pattern in patterns:
                all_files.extend(glob.glob(pattern))
            
            # Remove duplicates and sort
            all_files = sorted(list(set(all_files)))
            
            gwas_files = []
            
            for file_path in all_files:
                filename = os.path.basename(file_path)
                
                # Skip obviously non-GWAS files
                if any(skip in filename.lower() for skip in ['readme', 'log', 'tmp', 'temp']):
                    continue
                
                # Extract metadata from the file
                metadata = self._extract_file_metadata(file_path)
                
                # Generate file ID
                file_id = filename
                for ext in ['.tsv.bgz', '.tsv.gz', '.txt.gz', '.csv.gz', '.tsv', '.txt', '.csv']:
                    if file_id.endswith(ext):
                        file_id = file_id[:-len(ext)]
                        break
                
                # Create file entry
                gwas_file_entry = {
                    "id": file_id,
                    "name": metadata['phenotype_name'],
                    "filename": filename,
                    "data_field": metadata['phenotype_id'],
                    "phenotype": metadata['phenotype_name'],
                    "population": metadata['population'],
                    "sample_size": metadata['sample_size'],
                    "genome_build": metadata['genome_build'],
                    "file_path": file_path,
                    "file_size_mb": round(metadata['file_size'] / (1024 * 1024), 2),
                    "is_raw": metadata['is_raw'],
                    "url": f"/gwas-files/download/{file_id}",
                    "description": self._generate_file_description(filename, metadata)
                }
                
                gwas_files.append(gwas_file_entry)
            
            
            return {
                "gwas_files": gwas_files,
                "total_files": len(gwas_files)
            }, 200
            
        except Exception as e:
            logger.error(f"Error discovering GWAS files: {str(e)}")
            return {"error": f"Failed to discover GWAS files: {str(e)}"}, 500


class PhenotypesAPI(Resource):
    """
    API endpoint for getting and loading phenotypes
    """
    def __init__(self, phenotypes):
        self.phenotypes = phenotypes

    def get(self):
        """Get phenotypes with pagination to prevent memory issues"""
        try:
            # Get query parameters
            phenotype_id = request.args.get('id')
            search_term = request.args.get('search')
            limit = request.args.get('limit', type=int)
            skip = request.args.get('skip', 0, type=int)
            
            if phenotype_id:
                # Get specific phenotype
                phenotype = self.phenotypes.get_phenotypes(phenotype_id=phenotype_id)
                if not phenotype:
                    return {"error": "Phenotype not found"}, 404
                return serialize_datetime_fields({"phenotype": phenotype}), 200
            
            # Set reasonable default limit to prevent memory issues
            if limit is None:
                limit = 100
                logger.info(f"No limit specified, using default limit of {limit} for memory protection")
            
            # Get phenotypes with pagination
            phenotypes = self.phenotypes.get_phenotypes(
                limit=limit, 
                skip=skip, 
                search_term=search_term
            )
            
            # Get total count for pagination
            total_count = self.phenotypes.count_phenotypes(search_term=search_term)
            
            response = {
                "phenotypes": phenotypes,
                "total_count": total_count,
                "skip": skip,
                "limit": limit,
                "has_more": (skip + len(phenotypes)) < total_count,
                "next_skip": skip + len(phenotypes) if (skip + len(phenotypes)) < total_count else None
            }
            
            if search_term:
                response["search_term"] = search_term
            
            return serialize_datetime_fields(response), 200
            
        except Exception as e:
            logger.error(f"Error getting phenotypes: {str(e)}")
            return {"error": f"Failed to get phenotypes: {str(e)}"}, 500

    def post(self):
        """
        Bulk load phenotypes from JSON data
        
        Expects JSON array with format:
        [
            {"name": "phenotype name", "id": "EFO_1234567"},
            ...
        ]
        """
        try:
            # Get JSON data from request
            data = request.get_json()
            
            if not data:
                return {"error": "No JSON data provided"}, 400
            
            if not isinstance(data, list):
                return {"error": "Expected JSON array of phenotypes"}, 400
            
            # Transform data to match database schema
            # Input format: {"name": "...", "id": "..."}
            # Database format: {"phenotype_name": "...", "id": "..."}
            phenotypes_data = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                # Map "name" to "phenotype_name" for database
                phenotype = {
                    "id": item.get("id", ""),
                    "phenotype_name": item.get("name", "")
                }
                
                # Validate that both fields exist
                if phenotype["id"] and phenotype["phenotype_name"]:
                    phenotypes_data.append(phenotype)
                else:
                    logger.warning(f"Skipping invalid phenotype entry: {item}")
            
            if not phenotypes_data:
                return {"error": "No valid phenotypes found in JSON data"}, 400
            
            # Bulk insert phenotypes
            logger.info(f"Loading {len(phenotypes_data)} phenotypes into database...")
            result = self.phenotypes.bulk_create_phenotypes(phenotypes_data)
            
            response = {
                "message": "Phenotypes loaded successfully",
                "inserted_count": result['inserted_count'],
                "skipped_count": result['skipped_count'],
                "total_provided": len(phenotypes_data)
            }
            
            logger.info(f"Phenotype load complete: {result['inserted_count']} inserted, {result['skipped_count']} skipped")
            
            return response, 201
            
        except Exception as e:
            logger.error(f"Error loading phenotypes: {str(e)}")
            return {"error": f"Failed to load phenotypes: {str(e)}"}, 500


class GWASFileDownloadAPI(Resource):
    """
    API endpoint for downloading predefined GWAS files
    """
    def __init__(self, config):
        self.config = config

    def get(self, file_id):
        """Download a predefined GWAS file by file_id"""
        try:
            logger.info(f"[GWAS DOWNLOAD] Download request for file {file_id}")
            
            # Get data directory
            data_dir = getattr(self.config, 'data_dir', 'data')
            raw_data_path = os.path.join(data_dir, 'raw')
            
            # Scan for matching files dynamically
            possible_extensions = ['.tsv', '.tsv.gz', '.tsv.bgz']
            file_path = None
            original_filename = None
            
            for ext in possible_extensions:
                candidate_path = os.path.join(raw_data_path, f"{file_id}{ext}")
                if os.path.exists(candidate_path):
                    file_path = candidate_path
                    original_filename = f"{file_id}{ext}"
                    break
            
            if not file_path:
                logger.error(f"[GWAS DOWNLOAD] File not found for ID: {file_id}")
                return {"error": "File not found"}, 404
            
            # Generate a user-friendly download name
            download_name = original_filename.replace('_munged.gwas.imputed_v3.both_sexes', '_GWAS')
            
            logger.info(f"[GWAS DOWNLOAD] Serving file: {download_name} (Path: {file_path})")
            
            # Return file for download
            return send_file(
                file_path,
                as_attachment=True,
                download_name=download_name,
                mimetype='text/tab-separated-values'
            )
            
        except Exception as e:
            logger.error(f"[GWAS DOWNLOAD] Error downloading file {file_id}: {str(e)}")
            return {"error": f"Download failed: {str(e)}"}, 500

