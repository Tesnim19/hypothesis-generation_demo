"""
GWAS Library Handler

Manages the GWAS library collection in MongoDB, storing metadata for GWAS files
that can be downloaded on-demand and cached in MinIO.
"""

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pymongo import UpdateOne
from loguru import logger
from .base_handler import BaseHandler


class GWASLibraryHandler(BaseHandler):
    """Handler for GWAS library collection"""
    
    def __init__(self, mongodb_uri: str, db_name: str):
        """
        Initialize the GWAS library handler
        
        Args:
            mongodb_uri (str): MongoDB connection URI
            db_name (str): Database name
        """
        super().__init__(mongodb_uri, db_name)
        self.collection = self.db['gwas_library']
        
        # Create indexes for efficient queries
        self._create_indexes()
    
    def _create_indexes(self):
        """Create indexes on commonly queried fields"""
        try:
            # Unique index on file_id (filename is the unique identifier)
            self.collection.create_index('file_id', unique=True)
            
            # Index on phenotype_code for searching (not unique - can be N/A or duplicate)
            self.collection.create_index('phenotype_code')
            
            # Index on display_name for searching
            self.collection.create_index('display_name')
            
            # Index on sex for filtering
            self.collection.create_index('sex')
            
            # Index on downloaded status (cached in MinIO)
            self.collection.create_index('downloaded')
            
            # Text index for full-text search
            self.collection.create_index([
                ('display_name', 'text'),
                ('description', 'text'),
                ('phenotype_code', 'text'),
                ('filename', 'text')
            ])
            
            logger.info("GWAS library indexes created successfully")
        except Exception as e:
            logger.warning(f"Could not create indexes (may already exist): {e}")
    
    @staticmethod
    def _search_substring_query(search_term: str) -> dict:
        """Substring match on key string fields (case-insensitive)."""
        escaped = re.escape(search_term.strip())
        return {
            "$or": [
                {"file_id": {"$regex": escaped, "$options": "i"}},
                {"filename": {"$regex": escaped, "$options": "i"}},
                {"display_name": {"$regex": escaped, "$options": "i"}},
                {"description": {"$regex": escaped, "$options": "i"}},
                {"phenotype_code": {"$regex": escaped, "$options": "i"}},
            ]
        }

    def get_gwas_entry(self, file_id: str) -> Optional[Dict]:
        """
        Get a GWAS entry by file_id (filename)
        
        Args:
            file_id (str): File ID (filename)
            
        Returns:
            dict or None: GWAS entry if found
        """
        try:
            entry = self.collection.find_one({'file_id': file_id})
            
            if entry:
                # Remove MongoDB _id field
                entry.pop('_id', None)
            
            return entry
            
        except Exception as e:
            logger.error(f"Error getting GWAS entry {file_id}: {e}")
            return None
    
    def get_all_gwas_entries(
        self, 
        search_term: Optional[str] = None,
        sex_filter: Optional[str] = None,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict]:
        """
        Get all GWAS entries with optional filtering and pagination
        
        Args:
            search_term (str, optional): Substring search on file_id, filename, display_name, description, phenotype_code
            sex_filter (str, optional): Filter by sex ('both_sexes', 'male', 'female')
            limit (int): Maximum number of entries to return
            skip (int): Number of entries to skip (for pagination)
            
        Returns:
            list: List of GWAS entries
        """
        try:
            # Build query
            query = {}

            if search_term and search_term.strip():
                query.update(self._search_substring_query(search_term))

            if sex_filter:
                query['sex'] = sex_filter

            # Execute query with pagination
            cursor = self.collection.find(query).skip(skip).limit(limit)
            
            # Sort by download count (most popular first) and then by display name
            cursor = cursor.sort([
                ('download_count', -1),
                ('display_name', 1)
            ])
            
            entries = []
            for entry in cursor:
                # Remove MongoDB _id field
                entry.pop('_id', None)
                entries.append(entry)
            
            return entries
            
        except Exception as e:
            logger.error(f"Error getting GWAS entries: {e}")
            return []

    def get_gwas_semantic_candidates(
        self,
        sex_filter: Optional[str] = None,
        max_docs: int = 10_000,
    ) -> List[Dict]:
        """Load GWAS documents for semantic ranking (no text filter), capped at max_docs."""
        try:
            query: Dict = {}
            if sex_filter:
                query["sex"] = sex_filter

            cursor = (
                self.collection.find(query)
                .sort([("file_id", 1)])
                .limit(max_docs)
            )
            entries: List[Dict] = []
            for entry in cursor:
                entry.pop("_id", None)
                entries.append(entry)
            return entries
        except Exception as e:
            logger.error(f"Error loading GWAS semantic candidates: {e}")
            return []

    def bulk_set_gwas_semantic_embeddings(
        self,
        updates: List[tuple[str, List[float], str, str]],
    ) -> None:
        """
        Persist embedding cache for hybrid search.
        Each tuple is (file_id, embedding vector, SentenceTransformer model id, sha256 hex of search text).
        """
        if not updates:
            return
        now = datetime.now(timezone.utc)
        ops: List[UpdateOne] = []
        for file_id, vec, model_id, text_sha in updates:
            if not file_id or not vec:
                continue
            ops.append(
                UpdateOne(
                    {"file_id": file_id},
                    {
                        "$set": {
                            "st_embedding": vec,
                            "st_embed_model": model_id,
                            "st_embed_text_sha256": text_sha,
                            "updated_at": now,
                        }
                    },
                )
            )
        if not ops:
            return
        try:
            self.collection.bulk_write(ops, ordered=False)
        except Exception as e:
            logger.error(f"bulk_set_gwas_semantic_embeddings failed: {e}")

    def get_gwas_keyword_matches_unpaged(
        self,
        search_term: str,
        sex_filter: Optional[str] = None,
        max_results: int = 10_000,
    ) -> List[Dict]:
        """
        All GWAS entries matching substring search (same fields as get_all_gwas_entries),
        sorted like paginated queries, but without skip — capped at max_results.
        """
        try:
            if not search_term or not search_term.strip():
                return []
            query: Dict = {}
            query.update(self._search_substring_query(search_term))
            if sex_filter:
                query["sex"] = sex_filter
            cursor = (
                self.collection.find(query)
                .sort([("download_count", -1), ("display_name", 1)])
                .limit(max_results)
            )
            entries: List[Dict] = []
            for entry in cursor:
                entry.pop("_id", None)
                entries.append(entry)
            return entries
        except Exception as e:
            logger.error(f"Error loading GWAS keyword matches: {e}")
            return []

    def get_entry_count(
        self,
        search_term: Optional[str] = None,
        sex_filter: Optional[str] = None
    ) -> int:
        """
        Get count of GWAS entries matching filters
        
        Args:
            search_term (str, optional): Search term
            sex_filter (str, optional): Filter by sex
            
        Returns:
            int: Count of matching entries
        """
        try:
            query = {}

            if search_term and search_term.strip():
                query.update(self._search_substring_query(search_term))

            if sex_filter:
                query['sex'] = sex_filter

            return self.collection.count_documents(query)
            
        except Exception as e:
            logger.error(f"Error counting GWAS entries: {e}")
            return 0
    
    def update_gwas_entry(self, file_id: str, update_data: Dict) -> bool:
        """
        Update a GWAS entry
        
        Args:
            file_id (str): File ID (filename)
            update_data (dict): Fields to update
            
        Returns:
            bool: True if successful
        """
        try:
            # Add updated_at timestamp
            update_data['updated_at'] = datetime.now(timezone.utc)
            
            result = self.collection.update_one(
                {'file_id': file_id},
                {'$set': update_data}
            )
            
            if result.modified_count > 0:
                logger.info(f"Updated GWAS entry: {file_id}")
                return True
            else:
                logger.warning(f"No changes made to GWAS entry: {file_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error updating GWAS entry {file_id}: {e}")
            return False
    
    def mark_as_downloaded(self, file_id: str, minio_path: str, file_size: int) -> bool:
        """
        Mark a GWAS entry as downloaded and cached in MinIO
        
        Args:
            file_id (str): File ID (filename)
            minio_path (str): MinIO object key where file is cached
            file_size (int): File size in bytes
            
        Returns:
            bool: True if successful
        """
        try:
            update_data = {
                'downloaded': True,
                'minio_path': minio_path,
                'file_size': file_size,
                'last_accessed': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc)
            }
            
            result = self.collection.update_one(
                {'file_id': file_id},
                {'$set': update_data}
            )
            
            if result.modified_count > 0:
                logger.info(f"Marked as downloaded: {file_id} -> s3://{minio_path}")
                return True
            else:
                logger.warning(f"Failed to mark as downloaded: {file_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error marking {file_id} as downloaded: {e}")
            return False
    
    def increment_download_count(self, file_id: str) -> bool:
        """
        Increment the download count for a GWAS entry
        
        Args:
            file_id (str): File ID (filename)
            
        Returns:
            bool: True if successful
        """
        try:
            result = self.collection.update_one(
                {'file_id': file_id},
                {
                    '$inc': {'download_count': 1},
                    '$set': {
                        'last_accessed': datetime.now(timezone.utc),
                        'updated_at': datetime.now(timezone.utc)
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.debug(f"Incremented download count for: {file_id}")
                return True
            else:
                logger.warning(f"Failed to increment download count: {file_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error incrementing download count for {file_id}: {e}")
            return False
    
    def bulk_create_gwas_entries(self, entries: List[Dict]) -> Dict:
        """
        Bulk insert GWAS entries into the collection
        
        Args:
            entries (List[Dict]): List of GWAS entries to insert
            
        Returns:
            dict: Result with inserted_count and skipped_count
        """
        try:
            from datetime import datetime, timezone
            
            inserted_count = 0
            skipped_count = 0
            
            for entry in entries:
                # Add timestamps
                entry['created_at'] = datetime.now(timezone.utc)
                entry['updated_at'] = datetime.now(timezone.utc)
                
                # Add default fields if not present
                entry.setdefault('downloaded', False)
                entry.setdefault('download_count', 0)
                entry.setdefault('minio_path', None)
                entry.setdefault('file_size', None)
                entry.setdefault('last_accessed', None)
                entry.setdefault('sample_size', None)
                entry.setdefault('genome_build', None)
                
                try:
                    # Insert with unique constraint on file_id
                    self.collection.insert_one(entry)
                    inserted_count += 1
                except Exception as e:
                    # Skip duplicates
                    if 'duplicate key error' in str(e).lower():
                        logger.debug(f"Skipping duplicate entry: {entry.get('file_id')}")
                        skipped_count += 1
                    else:
                        logger.warning(f"Error inserting entry {entry.get('file_id')}: {e}")
                        skipped_count += 1
            
            logger.info(f"Bulk insert complete: {inserted_count} inserted, {skipped_count} skipped")
            
            return {
                'inserted_count': inserted_count,
                'skipped_count': skipped_count
            }
            
        except Exception as e:
            logger.error(f"Error during bulk insert: {e}")
            raise