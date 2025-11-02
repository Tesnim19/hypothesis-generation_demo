from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from datetime import datetime, timezone
import os
import json
from uuid import uuid4
import pandas as pd
from loguru import logger

class Database:
    def __init__(self, uri, db_name):
        # Store connection parameters for multiprocessing
        self.uri = uri
        self.db_name = db_name
        
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        
        # Existing collections
        self.users_collection = self.db['users']
        self.hypothesis_collection = self.db['hypotheses']
        self.enrich_collection = self.db['enrich']
        self.task_updates_collection = self.db['task_updates'] 
        self.summary_collection = self.db['summaries']
        self.processing_collection = self.db['processing_status']
        
        # New collections for project-based structure
        self.projects_collection = self.db['projects']
        self.file_metadata_collection = self.db['file_metadata']
        self.analysis_results_collection = self.db['analysis_results']
        self.credible_sets_collection = self.db['credible_sets']
        
        # Gene expression analysis collections
        self.gene_expression_runs_collection = self.db['gene_expression_runs']
        self.ldsc_results_collection = self.db['ldsc_results']
        self.coexpression_results_collection = self.db['coexpression_results']
        self.pathway_results_collection = self.db['pathway_results']
        self.tissue_mappings_collection = self.db['tissue_mappings']

    # ==================== USER METHODS ====================
    def create_user(self, email, password):
        if self.users_collection.find_one({'email': email}):
            return {'message': 'User already exists'}, 400
        
        hashed_password = generate_password_hash(password)
        self.users_collection.insert_one({'email': email, 'password': hashed_password})
        return {'message': 'User created successfully'}, 201

    def verify_user(self, email, password):
        user = self.users_collection.find_one({'email': email})
        if user and check_password_hash(user['password'], password):
            return {'message': 'Logged in successfully', 'user_id': str(user['_id'])}, 200
        return {'message': 'Invalid credentials'}, 401

    # ==================== PROJECT METHODS ====================
    def create_project(self, user_id, name, gwas_file_id, phenotype, population=None, ref_genome=None, analysis_parameters=None):
        """Create a new project"""
        project_data = {
            'user_id': user_id,
            'name': name,
            'phenotype': phenotype,
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
            'status': 'active',
            'gwas_file_id': gwas_file_id,
            'population': population,
            'ref_genome': ref_genome,
            'analysis_parameters': analysis_parameters or {}
        }
        result = self.projects_collection.insert_one(project_data)
        return str(result.inserted_id)

    def get_projects(self, user_id, project_id=None):
        """Get projects for a user"""
        query = {'user_id': user_id}
        if project_id:
            query['_id'] = ObjectId(project_id)
            project = self.projects_collection.find_one(query)
            if project:
                project['id'] = str(project['_id'])
                del project['_id']  
            return project
        
        projects = list(self.projects_collection.find(query))
        for project in projects:
            project['id'] = str(project['_id'])
            del project['_id']
        return projects

    def update_project(self, project_id, data):
        """Update project data"""
        data['updated_at'] = datetime.now(timezone.utc)
        result = self.projects_collection.update_one(
            {'_id': ObjectId(project_id)},
            {'$set': data}
        )
        return result.matched_count > 0

    # def delete_project(self, user_id, project_id):
    #     """Delete a project"""
    #     result = self.projects_collection.delete_one({
    #         '_id': ObjectId(project_id),
    #         'user_id': user_id
    #     })
    #     return result.deleted_count > 0

    def delete_project(self, user_id, project_id):
        """Delete a single project and all associated data"""
        try:
            return self.bulk_delete_projects(user_id, [project_id])
        except Exception as e:
            logger.error(f"Error deleting project {project_id}: {str(e)}")
            return False

    def bulk_delete_projects(self, user_id, project_ids):
        """Delete multiple projects and all associated data"""
        try:
            if not project_ids or not isinstance(project_ids, list):
                return False
            
            deleted_count = 0
            errors = []
            
            for project_id in project_ids:
                try:
                    # Verify project exists and belongs to user
                    project = self.projects_collection.find_one({
                        '_id': ObjectId(project_id),
                        'user_id': user_id
                    })
                    
                    if not project:
                        errors.append(f"Project {project_id} not found or access denied")
                        continue
                    
                    # Delete all associated data
                    self._delete_project_data(user_id, project_id)
                    
                    # Delete the project itself
                    result = self.projects_collection.delete_one({
                        '_id': ObjectId(project_id),
                        'user_id': user_id
                    })
                    
                    if result.deleted_count > 0:
                        deleted_count += 1
                        logger.info(f"Successfully deleted project {project_id} and all associated data")
                    else:
                        errors.append(f"Failed to delete project {project_id}")
                        
                except Exception as e:
                    logger.error(f"Error deleting project {project_id}: {str(e)}")
                    errors.append(f"Error deleting project {project_id}: {str(e)}")
            
            return {
                'deleted_count': deleted_count,
                'total_requested': len(project_ids),
                'errors': errors,
                'success': deleted_count == len(project_ids)
            }
            
        except Exception as e:
            logger.error(f"Error in bulk project deletion: {str(e)}")
            return False

    def _delete_project_data(self, user_id, project_id):
        """Delete all data associated with a project"""
        try:
            # 1. Delete credible sets
            credible_sets_result = self.credible_sets_collection.delete_many({
                'user_id': user_id,
                'project_id': project_id
            })
            logger.info(f"Deleted {credible_sets_result.deleted_count} credible sets for project {project_id}")
            
            # 2. Delete hypotheses and get their IDs for cascade deletion
            hypotheses = list(self.hypothesis_collection.find({
                'user_id': user_id,
                'project_id': project_id
            }))
            hypothesis_ids = [h.get('id') for h in hypotheses if h.get('id')]
            
            hypotheses_result = self.hypothesis_collection.delete_many({
                'user_id': user_id,
                'project_id': project_id
            })
            logger.info(f"Deleted {hypotheses_result.deleted_count} hypotheses for project {project_id}")
            
            # 3. Delete task updates for the hypotheses
            if hypothesis_ids:
                task_updates_result = self.task_updates_collection.delete_many({
                    'hypothesis_id': {'$in': hypothesis_ids}
                })
                logger.info(f"Deleted {task_updates_result.deleted_count} task updates for project {project_id}")
                
                # 4. Delete summaries for the hypotheses
                summaries_result = self.summary_collection.delete_many({
                    'user_id': user_id,
                    'hypothesis_id': {'$in': hypothesis_ids}
                })
                logger.info(f"Deleted {summaries_result.deleted_count} summaries for project {project_id}")
            
            # 5. Delete enrichment data
            enrich_result = self.enrich_collection.delete_many({
                'user_id': user_id,
                'project_id': project_id
            })
            logger.info(f"Deleted {enrich_result.deleted_count} enrichment records for project {project_id}")
            
            # 6. Delete analysis results
            analysis_result = self.analysis_results_collection.delete_many({
                'user_id': user_id,
                'project_id': project_id
            })
            logger.info(f"Deleted {analysis_result.deleted_count} analysis results for project {project_id}")
            
            # 7. Get file metadata for deletion
            project = self.projects_collection.find_one({
                '_id': ObjectId(project_id),
                'user_id': user_id
            })
            
            if project and project.get('gwas_file_id'):
                file_meta = self.file_metadata_collection.find_one({
                    '_id': ObjectId(project['gwas_file_id']),
                    'user_id': user_id
                })
                
                if file_meta:
                    # Delete physical file if it exists
                    file_path = file_meta.get('file_path')
                    if file_path and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            logger.info(f"Deleted physical file: {file_path}")
                        except Exception as file_e:
                            logger.warning(f"Could not delete physical file {file_path}: {file_e}")
                    
                    # Delete file metadata
                    self.file_metadata_collection.delete_one({
                        '_id': ObjectId(project['gwas_file_id']),
                        'user_id': user_id
                    })
                    logger.info(f"Deleted file metadata for project {project_id}")
            
            # 8. Delete analysis state files
            analysis_state_path = self.get_analysis_state_path(user_id, project_id)
            if os.path.exists(analysis_state_path):
                try:
                    os.remove(analysis_state_path)
                    logger.info(f"Deleted analysis state file: {analysis_state_path}")
                except Exception as state_e:
                    logger.warning(f"Could not delete analysis state file {analysis_state_path}: {state_e}")
            
            # 9. Delete analysis results directory
            analysis_dir = self.get_project_analysis_path(user_id, project_id)
            if os.path.exists(analysis_dir):
                try:
                    import shutil
                    shutil.rmtree(analysis_dir)
                    logger.info(f"Deleted analysis directory: {analysis_dir}")
                except Exception as dir_e:
                    logger.warning(f"Could not delete analysis directory {analysis_dir}: {dir_e}")
                    
        except Exception as e:
            logger.error(f"Error deleting project data for {project_id}: {str(e)}")
            raise

    # ==================== FILE METADATA METHODS ====================
    def create_file_metadata(self, user_id, filename, original_filename, file_path, file_type, file_size, md5_hash=None, record_count=None, download_url=None):
        """Create file metadata entry"""
        file_data = {
            'user_id': user_id,
            'filename': filename,
            'original_filename': original_filename,
            'file_path': file_path,
            'file_type': file_type,
            'file_size': file_size,
            'upload_date': datetime.now(timezone.utc),
            'md5_hash': md5_hash,
            'record_count': record_count,
            'download_url': download_url
        }
        result = self.file_metadata_collection.insert_one(file_data)
        return str(result.inserted_id)

    def get_file_metadata(self, user_id, file_id=None):
        """Get file metadata"""
        query = {'user_id': user_id}
        if file_id:
            query['_id'] = ObjectId(file_id)
            file_meta = self.file_metadata_collection.find_one(query)
            if file_meta:
                file_meta['_id'] = str(file_meta['_id'])
            return file_meta
        
        files = list(self.file_metadata_collection.find(query))
        for file_meta in files:
            file_meta['_id'] = str(file_meta['_id'])
        return files

    def delete_file_metadata(self, user_id, file_id):
        """Delete file metadata"""
        result = self.file_metadata_collection.delete_one({
            '_id': ObjectId(file_id),
            'user_id': user_id
        })
        return result.deleted_count > 0

    # ==================== ANALYSIS RESULTS METHODS ====================
    def create_analysis_result(self, project_id, population, gene_types_identified, result_path=None):
        """Create analysis result entry"""
        analysis_data = {
            'project_id': project_id,
            'population': population,
            'analysis_date': datetime.now(timezone.utc),
            'status': 'completed',
            'gene_types_identified': gene_types_identified,
            'result_path': result_path
        }
        result = self.analysis_results_collection.insert_one(analysis_data)
        return str(result.inserted_id)

    def get_analysis_results(self, project_id, analysis_id=None):
        """Get analysis results for a project"""
        query = {'project_id': project_id}
        if analysis_id:
            query['_id'] = ObjectId(analysis_id)
            analysis = self.analysis_results_collection.find_one(query)
            if analysis:
                analysis['_id'] = str(analysis['_id'])
            return analysis
        
        results = list(self.analysis_results_collection.find(query))
        for analysis in results:
            analysis['_id'] = str(analysis['_id'])
        return results

    def update_analysis_result(self, analysis_id, data):
        """Update analysis result"""
        result = self.analysis_results_collection.update_one(
            {'_id': ObjectId(analysis_id)},
            {'$set': data}
        )
        return result.matched_count > 0

    def save_analysis_results(self, user_id, project_id, results_data):
        """Save analysis results to database"""
        try:
            # Create analysis result entry
            analysis_data = {
                'user_id': user_id,
                'project_id': project_id,
                'analysis_date': datetime.now(timezone.utc),
                'status': 'completed',
                'results_data': results_data
            }
            result = self.analysis_results_collection.insert_one(analysis_data)
            logger.info(f"Saved analysis results for project {project_id}: {len(results_data)} variants")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
            raise

    # ==================== CREDIBLE SETS METHODS ====================
    def save_credible_set(self, user_id, project_id, credible_set_data):
        """Save a single credible set with its own lead variant"""
        try:
            # Extract lead variant info from the credible set data
            variants_data = credible_set_data.get('variants', {}).get('data', {})
            if not variants_data or not variants_data.get('variant'):
                raise ValueError("No variant data found in credible set")
            
            # Find lead variant (highest posterior probability)
            variants = variants_data['variant']
            posterior_probs = variants_data['posterior_prob']
            max_idx = posterior_probs.index(max(posterior_probs))
            lead_variant_id = variants[max_idx]
            
            # Create lead variant info
            lead_variant = {
                'id': lead_variant_id,
                'rs_id': variants_data.get('rs_id', [None] * len(variants))[max_idx],
                'beta': variants_data['beta'][max_idx],
                'chromosome': str(variants_data['chromosome'][max_idx]),
                'log_pvalue': variants_data['log_pvalue'][max_idx],
                'position': variants_data['position'][max_idx],
                'ref_allele': variants_data['ref_allele'][max_idx],
                'minor_allele': variants_data['minor_allele'][max_idx],
                'ref_allele_freq': variants_data['ref_allele_freq'][max_idx],
                'posterior_prob': variants_data['posterior_prob'][max_idx]
            }
            
            # Create credible set document
            credible_set_doc = {
                'user_id': user_id,
                'project_id': project_id,
                'lead_variant_id': lead_variant_id,
                'coverage': credible_set_data.get('coverage'),
                'variants_count': len(variants),
                'completed_at': credible_set_data.get('completed_at'),
                'lead_variant': lead_variant,
                'variants_data': credible_set_data.get('variants'),
                'metadata': credible_set_data.get('metadata', {}),
                'created_at': datetime.now(timezone.utc),
                'type': 'credible_set'
            }
            
            result = self.credible_sets_collection.insert_one(credible_set_doc)
            logger.info(f"Saved credible set with lead variant {lead_variant_id} in project {project_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving credible set: {str(e)}")
            raise

    def get_credible_sets_for_project(self, user_id, project_id):
        """Get all credible sets for a project"""
        try:
            query = {
                'user_id': user_id,
                'project_id': project_id,
                'type': 'credible_set'
            }
            
            results = list(self.credible_sets_collection.find(query))
            credible_sets = []
            
            for result in results:
                credible_set = {
                    '_id': str(result['_id']),
                    'coverage': result.get('coverage'),
                    'variants_count': result.get('variants_count'),
                    'completed_at': result.get('completed_at'),
                    'lead_variant': result.get('lead_variant')
                }
                credible_sets.append(credible_set)
            
            return credible_sets
        except Exception as e:
            logger.error(f"Error getting credible sets for project: {str(e)}")
            raise

    def get_credible_set_by_lead_variant(self, user_id, project_id, lead_variant_id):
        """Get credible set data by lead variant ID"""
        try:
            query = {
                'user_id': user_id,
                'project_id': project_id,
                'lead_variant_id': lead_variant_id,
                'type': 'credible_set'
            }
            
            result = self.credible_sets_collection.find_one(query)
            if result:
                result['_id'] = str(result['_id'])
                return result
            return None
        except Exception as e:
            logger.error(f"Error getting credible set by lead variant: {str(e)}")
            raise

    def get_credible_set_by_id(self, user_id, project_id, credible_set_id):
        """Get credible set data by credible set ID"""
        try:
            from bson import ObjectId
            query = {
                '_id': ObjectId(credible_set_id),
                'user_id': user_id,
                'project_id': project_id,
                'type': 'credible_set'
            }
            
            result = self.credible_sets_collection.find_one(query)
            if result:
                result['_id'] = str(result['_id'])
                return result
            return None
        except Exception as e:
            logger.error(f"Error getting credible set by ID: {str(e)}")
            raise

    def create_enrich(self, user_id, project_id, variant, phenotype, causal_gene, go_terms, causal_graph):
        """Create enrichment entry with project references"""
        enrich_data = {
            'id': str(uuid4()),
            'user_id': user_id,
            'project_id': project_id,
            'variant': variant,
            'phenotype': phenotype,
            'causal_gene': causal_gene,
            'GO_terms': go_terms,
            'causal_graph': causal_graph,
            'created_at': datetime.now(timezone.utc)
        }
        result = self.enrich_collection.insert_one(enrich_data)
        return enrich_data['id']

    def get_enrich_by_lead_variant(self, user_id, lead_variant_id, variant, phenotype):
        """Check if enrichment exists for lead variant, variant, and phenotype"""
        return self.enrich_collection.find_one({
            'user_id': user_id,
            'lead_variant_id': lead_variant_id,
            'variant': variant,
            'phenotype': phenotype
        })

    def get_hypotheses_by_project(self, user_id, project_id):
        """Get all hypotheses for a project"""
        hypotheses = list(self.hypothesis_collection.find({
            'user_id': user_id,
            'project_id': project_id
        }))
        for hypothesis in hypotheses:
            hypothesis['_id'] = str(hypothesis['_id'])
        return hypotheses

    # ==================== UTILITY METHODS ====================
    def get_project_analysis_path(self, user_id, project_id):
        """Get the analysis path for a project"""
        return f"data/projects/{user_id}/{project_id}/analysis"

    def get_analysis_state_path(self, user_id, project_id):
        """Get the analysis state file path"""
        return f"data/states/{user_id}/{project_id}/analysis_state.json"

    def save_analysis_state(self, user_id, project_id, state_data):
        """Save analysis state to file"""
        state_path = self.get_analysis_state_path(user_id, project_id)
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        
        with open(state_path, 'w') as f:
            json.dump(state_data, f, default=str)

    def load_analysis_state(self, user_id, project_id):
        """Load analysis state from file"""
        state_path = self.get_analysis_state_path(user_id, project_id)
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                return json.load(f)
        return None

    def update_analysis_state(self, project_id, user_id, update_data):
        """Update analysis state by merging new data with existing state"""
        # Load existing state
        existing_state = self.load_analysis_state(user_id, project_id)
        if existing_state is None:
            existing_state = {}
        
        # Merge update data
        existing_state.update(update_data)
        
        # Save updated state
        self.save_analysis_state(user_id, project_id, existing_state)
        logger.info(f"Updated analysis state for project {project_id} with keys: {list(update_data.keys())}")
        return existing_state

    def create_hypothesis(self, user_id, data):
        data['user_id'] = user_id
        result = self.hypothesis_collection.insert_one(data)
        return {'message': 'Hypothesis created', 'id': str(result.inserted_id)}, 201

    def get_hypotheses(self, user_id=None, hypothesis_id=None):
        query = {}
        
        if user_id:
            query['user_id'] = user_id
        if hypothesis_id:
            query['id'] = hypothesis_id
            hypothesis = self.hypothesis_collection.find_one(query)
            if hypothesis:
                hypothesis["_id"] = str(hypothesis["_id"])
            else:
                logger.info("No document found for the given hypothesis id.")
            return hypothesis

        hypotheses = list(self.hypothesis_collection.find(query))
        for hypothesis in hypotheses:
            hypothesis['_id'] = str(hypothesis['_id'])

        return hypotheses if hypotheses else []

    def check_hypothesis(self, user_id=None, enrich_id=None, go_id=None):
        query = {}
        
        if user_id:
            query['user_id'] = user_id
        if enrich_id:
            query['enrich_id'] = enrich_id
        if go_id:
            query['go_id'] = go_id
        
        hypothesis = self.hypothesis_collection.find_one(query)
        
        return hypothesis is not None
    
    def check_enrich(self, user_id=None, phenotype=None, variant_id=None):
        query = {}
        
        if user_id:
            query['user_id'] = user_id
        if phenotype:
            query['phenotype'] = phenotype
        if variant_id:
            query['variant'] = variant_id
        
        enrich = self.enrich_collection.find_one(query)
        
        return enrich is not None

    def get_hypothesis_by_enrich_and_go(self, enrich_id, go_id, user_id=None):
        query = {
            'enrich_id': enrich_id,
            'go_id': go_id,
            'user_id': user_id
        }
        hypothesis = self.hypothesis_collection.find_one(query)
        if hypothesis:
            hypothesis['_id'] = str(hypothesis['_id'])

        return hypothesis

    def get_enrich_by_phenotype_and_variant(self, phenotype, variant_id, user_id=None):
        query = {
            'phenotype': phenotype,
            'variant': variant_id,
            'user_id': user_id
        }
        
        enrich = self.enrich_collection.find_one(query)
        
        if enrich:
            enrich['_id'] = str(enrich['_id'])
        
        return enrich

    def get_enrich(self, user_id=None, enrich_id=None):
        query = {}
        
        if user_id:
            query['user_id'] = user_id
        if enrich_id:
            query['id'] = enrich_id
            enrich = self.enrich_collection.find_one(query)  
            if enrich:
                enrich['_id'] = str(enrich['_id'])
            else:
                logger.info("No document found for the given enrich_id.")
            return enrich

        enriches = list(self.enrich_collection.find(query))
        for enrich in enriches:
            enrich['_id'] = str(enrich['_id'])

        return enriches if enriches else []

    def delete_hypothesis(self, user_id, hypothesis_id):
        result = self.hypothesis_collection.delete_one({'id': hypothesis_id, 'user_id': user_id})
        if result.deleted_count > 0:
            return {'message': 'Hypothesis deleted'}, 200
        return {'message': 'Hypothesis not found or not authorized'}, 404
    
    def bulk_delete_hypotheses(self, user_id, hypothesis_ids):
        """
        Delete multiple hypotheses by their IDs for a specific user.
        """
        if not hypothesis_ids or not isinstance(hypothesis_ids, list):
            return {'message': 'Invalid hypothesis_ids format. Expected a non-empty list.'}, 400

        results = {'successful': [], 'failed': []}

        # Bulk delete
        bulk_result = self.hypothesis_collection.delete_many({
            'id': {'$in': hypothesis_ids}, 
            'user_id': user_id
        })

        # Check if all were deleted
        if bulk_result.deleted_count == len(hypothesis_ids):
            return {
                'message': f'All {bulk_result.deleted_count} hypotheses deleted successfully',
                'deleted_count': bulk_result.deleted_count,
                'successful': hypothesis_ids,
                'failed': []
            }, 200

        # Identify which ones failed
        deleted_ids = set(hypothesis_ids[:bulk_result.deleted_count])  # Approximate success count
        failed_ids = list(set(hypothesis_ids) - deleted_ids)

        return {
            'message': f"{bulk_result.deleted_count} hypotheses deleted successfully, {len(failed_ids)} failed",
            'deleted_count': bulk_result.deleted_count,
            'successful': list(deleted_ids),
            'failed': [{'id': h_id, 'reason': 'Not found or not authorized'} for h_id in failed_ids]
        }, 207 if deleted_ids else 404  # Use 207 for partial success
    
    def delete_enrich(self, user_id, enrich_id):
        result = self.enrich_collection.delete_one({'id': enrich_id, 'user_id': user_id})
        if result.deleted_count > 0:
            return {'message': 'Enrich deleted'}, 200
        return {'message': 'Enrich not found or not authorized'}, 404
    
    # def add_task_update(self, hypothesis_id, task_name, state, details=None, error=None):
    #     task_update = {
    #         "hypothesis_id": hypothesis_id,
    #         "task_name": task_name,
    #         "state": state.value,
    #         "timestamp": datetime.now(timezone.utc).isoformat(timespec='milliseconds') + "Z",
    #         "details": details if details else {},
    #         "error": error if error else None,
    #         "progress": progress if progress is not None else 0
    #     }
      
    #     self.task_updates_collection.insert_one(task_update)

    def get_task_history(self, hypothesis_id):
        task_history = list(self.task_updates_collection.find({"hypothesis_id": hypothesis_id}))
        
        for update in task_history:
            update["_id"] = str(update["_id"])
        return task_history
    
    def get_latest_task_state(self, hypothesis_id):
        task_history = list(self.task_updates_collection.find({"hypothesis_id": hypothesis_id}).sort("timestamp", -1).limit(1))
        if task_history:
            return task_history[0]
        return None

    def update_hypothesis(self, hypothesis_id, data):
        # Remove _id if present in data to avoid modification errors
        if '_id' in data:
            del data['_id']
        
        result = self.hypothesis_collection.update_one(
            {'id': hypothesis_id},
            {'$set': data}
        )
        
        if result.matched_count > 0:
            return {'message': 'Hypothesis updated successfully'}, 200
        return {'message': 'Hypothesis not found'}, 404

    def save_task_history(self, hypothesis_id, task_history):
        """Save complete task history to DB"""
        # Delete existing history first
        self.task_updates_collection.delete_many({"hypothesis_id": hypothesis_id})
        
        # Insert new history as a batch
        if task_history:
            self.task_updates_collection.insert_many([
                {**update, "hypothesis_id": hypothesis_id}
                for update in task_history
            ])
    
    def get_hypothesis_by_phenotype_and_variant(self, user_id, phenotype, variant):
        return self.hypothesis_collection.find_one({
            'user_id': user_id,
            'phenotype': phenotype,
            'variant': variant
        })

    def get_hypothesis_by_enrich(self, user_id, enrich_id):
        return self.hypothesis_collection.find_one({
            'user_id': user_id,
            'enrich_id': enrich_id
        })

    def get_hypothesis_by_id(self, hypothesis_id):
        """
        Get hypothesis by ID without user filtering - used by system services
        """
        hypothesis = self.hypothesis_collection.find_one({'id': hypothesis_id})
        if hypothesis:
            hypothesis['_id'] = str(hypothesis['_id'])
        return hypothesis

    def get_hypothesis_by_phenotype_and_variant_in_project(self, user_id, project_id, phenotype, variant):
        """Get hypothesis by phenotype, variant, and project (no credible_set_id needed)"""
        return self.hypothesis_collection.find_one({
            'user_id': user_id,
            'project_id': project_id,
            'phenotype': phenotype,
            'variant': variant
        })
    
    def create_summary(self, user_id, hypothesis_id, summary_data):
        summary_doc = {
            "user_id": user_id,
            "hypothesis_id": hypothesis_id,
            "summary": summary_data,
        }
        result = self.summary_collection.insert_one(summary_doc)
        
        return {
            "summary_id": str(result.inserted_id),
            "user_id": user_id,
            "hypothesis_id": hypothesis_id,
            "summary": summary_data
        }, 201
    
    def check_summary(self, user_id, hypothesis_id):
        query = {}
        
        if user_id:
            query['user_id'] = user_id
        if hypothesis_id:
            query['hypothesis_id'] = hypothesis_id
        
        summary = self.summary_collection.find_one(query)

        return summary

    def check_global_summary(self, variant_input):
        query = {"variant": variant_input}
        summary = self.summary_collection.find_one(query)
        if summary:
            summary["_id"] = str(summary["_id"])
        return summary

    def create_global_summary(self, variant_input, summary_data):
        summary_doc = {
            "variant": variant_input,
            "summary": summary_data,
        }
        result = self.summary_collection.insert_one(summary_doc)
        return {
            "summary_id": str(result.inserted_id),
            "variant": variant_input,
            "summary": summary_data,
        }
    
    def get_summary(self, user_id, summary_id):
        if not user_id or not summary_id:
            print("Missing user_id or summary_id")
            return None
        
        query = {
        "user_id": user_id,
        "_id": ObjectId(summary_id) 
        }
        summary =  self.summary_collection.find_one(query)
        if summary:
            summary["_id"] = str(summary["_id"])

        return summary
    
    def check_processing_status(self, variant_input):
        return self.processing_collection.find_one({"variant": variant_input})

    def set_processing_status(self, variant_input, status):
        if status:
            self.processing_collection.update_one(
                {"variant": variant_input},
                {"$set": {"status": "processing"}},
                upsert=True
            )
        else:
            self.processing_collection.delete_one({"variant": variant_input})
    
    # ==================== GENE EXPRESSION METHODS ====================
    """
    Database additions - Add these methods to your Database class in db.py
    Copy these methods into your existing Database class
    """

    def create_gene_expression_run(self, gwas_file, gene_of_interest, project_id, user_id):
        """Create a new gene expression analysis run"""
        run_data = {
            'id': str(uuid4()),
            'gwas_file': gwas_file,
            'gene_of_interest': gene_of_interest,
            'project_id': project_id,
            'user_id': user_id,
            'status': 'running',
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc)
        }
        result = self.gene_expression_runs_collection.insert_one(run_data)
        logger.info(f"Created gene expression run {run_data['id']} for gene {gene_of_interest}")
        return run_data['id']


    def update_gene_expression_run_status(self, analysis_run_id, status):
        """Update gene expression analysis run status"""
        result = self.gene_expression_runs_collection.update_one(
            {'id': analysis_run_id},
            {'$set': {
                'status': status,
                'updated_at': datetime.now(timezone.utc)
            }}
        )
        logger.info(f"Updated gene expression run {analysis_run_id} status to {status}")
        return result.matched_count > 0


    def save_ldsc_results(self, analysis_run_id, ldsc_results_data):
        """Save LDSC results to database"""
        try:
            results_docs = []
            for idx, result in enumerate(ldsc_results_data):
                doc = {
                    'id': str(uuid4()),
                    'analysis_run_id': analysis_run_id,
                    'tissue_name': result.get('Name', ''),
                    'coefficient': result.get('Coefficient'),
                    'coefficient_se': result.get('Coefficient_std_error'),
                    'p_value': result.get('Coefficient_P_value'),
                    'rank_order': idx + 1,
                    'created_at': datetime.now(timezone.utc)
                }
                results_docs.append(doc)
            
            if results_docs:
                self.ldsc_results_collection.insert_many(results_docs)
                logger.info(f"Saved {len(results_docs)} LDSC results for analysis {analysis_run_id}")
            return len(results_docs)
        except Exception as e:
            logger.error(f"Error saving LDSC results: {str(e)}")
            raise


    def save_tissue_mappings(self, analysis_run_id, tissue_mapping_results):
        """Save tissue mapping results"""
        try:
            mapping_docs = []
            for gtex_tissue, mapping_data in tissue_mapping_results.items():
                doc = {
                    'id': str(uuid4()),
                    'analysis_run_id': analysis_run_id,
                    'gtex_tissue_name': mapping_data.get('gtex_tissue_name', ''),
                    'cellxgene_parent_ontology_name': mapping_data.get('cellxgene_parent_ontology_name', ''),
                    'cellxgene_descendant_ontology_name': mapping_data.get('cellxgene_descendant_ontology_name', ''),
                    'match_type': mapping_data.get('match_type', ''),
                    'mapping_notes': mapping_data.get('notes', ''),
                    'created_at': datetime.now(timezone.utc)
                }
                mapping_docs.append(doc)
            
            if mapping_docs:
                self.tissue_mappings_collection.insert_many(mapping_docs)
                logger.info(f"Saved {len(mapping_docs)} tissue mappings for analysis {analysis_run_id}")
            return len(mapping_docs)
        except Exception as e:
            logger.error(f"Error saving tissue mappings: {str(e)}")
            raise


    def save_coexpression_results(self, analysis_run_id, hgnc_converted_results):
        """Save co-expression results"""
        try:
            results_docs = []
            for tissue_name, results in hgnc_converted_results.items():
                # Save positive correlations
                for rank, (gene_symbol, correlation) in enumerate(results.get('top_positive_hgnc', [])):
                    doc = {
                        'id': str(uuid4()),
                        'analysis_run_id': analysis_run_id,
                        'tissue_name': tissue_name,
                        'gene_symbol': gene_symbol,
                        'correlation': correlation,
                        'correlation_type': 'positive',
                        'rank_order': rank + 1,
                        'created_at': datetime.now(timezone.utc)
                    }
                    results_docs.append(doc)
                
                # Save negative correlations
                for rank, (gene_symbol, correlation) in enumerate(results.get('top_negative_hgnc', [])):
                    doc = {
                        'id': str(uuid4()),
                        'analysis_run_id': analysis_run_id,
                        'tissue_name': tissue_name,
                        'gene_symbol': gene_symbol,
                        'correlation': correlation,
                        'correlation_type': 'negative',
                        'rank_order': rank + 1,
                        'created_at': datetime.now(timezone.utc)
                    }
                    results_docs.append(doc)
            
            if results_docs:
                self.coexpression_results_collection.insert_many(results_docs)
                logger.info(f"Saved {len(results_docs)} co-expression results for analysis {analysis_run_id}")
            return len(results_docs)
        except Exception as e:
            logger.error(f"Error saving co-expression results: {str(e)}")
            raise


    def save_pathway_results(self, analysis_run_id, pathway_results):
        """Save pathway enrichment results"""
        try:
            results_docs = []
            for tissue_name, df in pathway_results.items():
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        doc = {
                            'id': str(uuid4()),
                            'analysis_run_id': analysis_run_id,
                            'tissue_name': tissue_name,
                            'pathway_term': row.get('Term', ''),
                            'pathway_id': row.get('ID', ''),
                            'adjusted_p_value': row.get('Adjusted P-value'),
                            'odds_ratio': row.get('Odds Ratio'),
                            'overlap_count': row.get('Overlap'),
                            'pathway_size': row.get('Gene_set_size'),
                            'created_at': datetime.now(timezone.utc)
                        }
                        results_docs.append(doc)
            
            if results_docs:
                self.pathway_results_collection.insert_many(results_docs)
                logger.info(f"Saved {len(results_docs)} pathway results for analysis {analysis_run_id}")
            return len(results_docs)
        except Exception as e:
            logger.error(f"Error saving pathway results: {str(e)}")
            raise


    def get_gene_expression_results(self, project_id=None, user_id=None, gene_of_interest=None):
        """Get gene expression analysis results"""
        query = {}
        if project_id:
            query['project_id'] = project_id
        if user_id:
            query['user_id'] = user_id
        if gene_of_interest:
            query['gene_of_interest'] = gene_of_interest
        
        runs = list(self.gene_expression_runs_collection.find(query).sort('created_at', -1))
        
        results = []
        for run in runs:
            run['_id'] = str(run['_id'])
            analysis_run_id = run['id']
            
            # Get LDSC results
            ldsc_results = list(self.ldsc_results_collection.find(
                {'analysis_run_id': analysis_run_id}
            ).sort('rank_order', 1))
            
            # Get co-expression results
            coexp_results = list(self.coexpression_results_collection.find(
                {'analysis_run_id': analysis_run_id}
            ).sort([('tissue_name', 1), ('correlation_type', 1), ('rank_order', 1)]))
            
            # Get pathway results
            pathway_results = list(self.pathway_results_collection.find(
                {'analysis_run_id': analysis_run_id}
            ).sort([('tissue_name', 1), ('adjusted_p_value', 1)]))
            
            # Clean up _id fields
            for result_list in [ldsc_results, coexp_results, pathway_results]:
                for result in result_list:
                    result['_id'] = str(result['_id'])
            
            results.append({
                'run_info': run,
                'ldsc_results': ldsc_results,
                'coexpression_results': coexp_results,
                'pathway_results': pathway_results
            })
        
        return results


    def check_gene_expression_status(self, project_id, user_id):
        """Check gene expression analysis status for a project"""
        latest_run = self.gene_expression_runs_collection.find_one(
            {'project_id': project_id, 'user_id': user_id},
            sort=[('created_at', -1)]
        )
        
        if not latest_run:
            return {
                'status': 'not_started',
                'has_data': False,
                'analysis_count': 0
            }
        
        # Count results
        analysis_run_id = latest_run['id']
        ldsc_count = self.ldsc_results_collection.count_documents({'analysis_run_id': analysis_run_id})
        coexp_count = self.coexpression_results_collection.count_documents({'analysis_run_id': analysis_run_id})
        pathway_count = self.pathway_results_collection.count_documents({'analysis_run_id': analysis_run_id})
        
        return {
            'status': latest_run['status'],
            'has_data': latest_run['status'] == 'completed',
            'gene_of_interest': latest_run['gene_of_interest'],
            'created_at': latest_run['created_at'],
            'ldsc_count': ldsc_count,
            'coexpression_count': coexp_count,
            'pathway_count': pathway_count
        }


    def get_coexpressed_genes_for_enrichment(self, gene_of_interest, project_id=None, min_correlation=0.5):
        """Get co-expressed genes for enrichment analysis"""
        # Find latest completed run
        run_query = {'gene_of_interest': gene_of_interest, 'status': 'completed'}
        if project_id:
            run_query['project_id'] = project_id
        
        latest_run = self.gene_expression_runs_collection.find_one(
            run_query, sort=[('created_at', -1)]
        )
        
        if not latest_run:
            return []
        
        # Get co-expressed genes
        coexp_query = {
            'analysis_run_id': latest_run['id'],
            'correlation_type': 'positive',
            'correlation': {'$gte': min_correlation}
        }
        
        coexp_results = list(self.coexpression_results_collection.find(
            coexp_query
        ).sort('correlation', -1))
        
        # Return unique gene symbols
        seen_genes = set()
        unique_genes = []
        for result in coexp_results:
            gene_symbol = result['gene_symbol']
            if gene_symbol not in seen_genes:
                unique_genes.append(gene_symbol)
                seen_genes.add(gene_symbol)
        
        return unique_genes

    # ==================== TISSUE SELECTION METHODS ====================
    
    def save_tissue_selection(self, user_id, project_id, variant_id, tissue_name, tissue_data=None):
        """Save user's tissue selection for a specific variant"""
        try:
            selection_data = {
                'id': str(uuid4()),
                'user_id': user_id,
                'project_id': project_id,
                'variant_id': variant_id,
                'tissue_name': tissue_name,
                'tissue_data': tissue_data or {},
                'created_at': datetime.now(timezone.utc)
            }
            
            # Use upsert to replace any existing selection for this variant
            result = self.db['tissue_selections'].replace_one(
                {
                    'user_id': user_id,
                    'project_id': project_id,
                    'variant_id': variant_id
                },
                selection_data,
                upsert=True
            )
            
            logger.info(f"Saved tissue selection: {tissue_name} for variant {variant_id}")
            return selection_data['id']
            
        except Exception as e:
            logger.error(f"Error saving tissue selection: {str(e)}")
            raise
    
    def get_tissue_selection(self, user_id, project_id, variant_id):
        """Get user's tissue selection for a specific variant"""
        try:
            selection = self.db['tissue_selections'].find_one({
                'user_id': user_id,
                'project_id': project_id,
                'variant_id': variant_id
            })
            
            if selection:
                selection['_id'] = str(selection['_id'])
                
            return selection
            
        except Exception as e:
            logger.error(f"Error getting tissue selection: {str(e)}")
            return None
    
    def get_available_tissues_for_selection(self, user_id, project_id, limit=20):
        """Get available tissues from LDSC analysis for user selection"""
        try:
            # Get the latest LDSC analysis for this project
            latest_run = self.gene_expression_runs_collection.find_one(
                {
                    'user_id': user_id,
                    'project_id': project_id,
                    'gene_of_interest': 'project_analysis',  # Project-level analysis
                    'status': {'$in': ['ldsc_tissue_completed', 'completed']}
                },
                sort=[('created_at', -1)]
            )
            
            if not latest_run:
                return []
            
            # Get LDSC results for tissue selection
            ldsc_results = list(self.ldsc_results_collection.find(
                {'analysis_run_id': latest_run['id']}
            ).sort('p_value', 1).limit(limit))  # Sort by p-value (most significant first)
            
            # Format for frontend selection
            tissues = []
            for result in ldsc_results:
                tissue_data = {
                    'tissue_name': result['tissue_name'],
                    'coefficient': result.get('coefficient'),
                    'p_value': result.get('p_value'),
                    'rank_order': result.get('rank_order'),
                    'is_significant': result.get('p_value', 1) < 0.05
                }
                tissues.append(tissue_data)
            
            return tissues
            
        except Exception as e:
            logger.error(f"Error getting available tissues: {str(e)}")
            return []