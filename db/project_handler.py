from bson.objectid import ObjectId
from datetime import datetime, timezone
import os
import json
from .base_handler import BaseHandler


class ProjectHandler(BaseHandler):
    """Handler for project CRUD operations"""
    
    def __init__(self, uri, db_name):
        super().__init__(uri, db_name)
        self.projects_collection = self.db['projects']
        self.analysis_state_collection = self.db['analysis_states']
    
    def create_project(self, user_id, name, gwas_file_id, phenotype):
        """Create a new project"""
        project_data = {
            'user_id': user_id,
            'name': name,
            'phenotype': phenotype,
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
            'status': 'active',
            'gwas_file_id': gwas_file_id
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

    def delete_project(self, user_id, project_id):
        """Delete a project"""
        result = self.projects_collection.delete_one({
            '_id': ObjectId(project_id),
            'user_id': user_id
        })
        return result.deleted_count > 0

    def save_analysis_state(self, user_id, project_id, state_data):
        """Upsert analysis state into MongoDB"""
        doc = {
            'user_id': user_id,
            'project_id': project_id,
            'state': state_data,
            'updated_at': datetime.now(timezone.utc),
        }

        self.analysis_state_collection.update_one(
            {'user_id': user_id, 'project_id': project_id},
            {'$set': doc},
            upsert=True
        )

    def load_analysis_state(self, user_id, project_id):
        """Load analysis state from MongoDB"""
        doc = self.analysis_state_collection.find_one(
            {'user_id': user_id, 'project_id': project_id}
        )
        if not doc:
            return None
        return doc.get('state')