"""
GWAS Library Handler

Manages the GWAS library collection in MongoDB, storing metadata for GWAS files
that can be downloaded on-demand.
"""

from datetime import datetime, timezone
from typing import List, Dict, Optional
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
            
            # Index on downloaded status
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
    
    def create_gwas_entry(self, entry_data: Dict) -> str:
        """
        Create a new GWAS entry
        
        Args:
            entry_data (dict): GWAS entry data
            
        Returns:
            str: file_id of created entry
        """
        try:
            # Add timestamps
            entry_data['created_at'] = datetime.now(timezone.utc)
            entry_data['updated_at'] = datetime.now(timezone.utc)
            
            # Initialize download tracking fields
            if 'downloaded' not in entry_data:
                entry_data['downloaded'] = False
            if 'download_count' not in entry_data:
                entry_data['download_count'] = 0
            if 'local_path' not in entry_data:
                entry_data['local_path'] = None
            if 'last_accessed' not in entry_data:
                entry_data['last_accessed'] = None
            
            # Insert into database
            result = self.collection.insert_one(entry_data)
            
            logger.info(f"Created GWAS entry: {entry_data['file_id']} (phenotype: {entry_data.get('phenotype_code', 'N/A')})")
            return entry_data['file_id']
            
        except Exception as e:
            logger.error(f"Error creating GWAS entry: {e}")
            raise
    
    def bulk_create_gwas_entries(self, entries: List[Dict]) -> Dict:
        """
        Bulk create GWAS entries
        
        Args:
            entries (list): List of GWAS entry dictionaries
            
        Returns:
            dict: Result with inserted_count and skipped_count
        """
        try:
            if not entries:
                return {'inserted_count': 0, 'skipped_count': 0}
            
            # Add timestamps and defaults to all entries
            now = datetime.now(timezone.utc)
            for entry in entries:
                entry['created_at'] = now
                entry['updated_at'] = now
                
                # Initialize download tracking fields
                if 'downloaded' not in entry:
                    entry['downloaded'] = False
                if 'download_count' not in entry:
                    entry['download_count'] = 0
                if 'local_path' not in entry:
                    entry['local_path'] = None
                if 'last_accessed' not in entry:
                    entry['last_accessed'] = None
            
            # Use ordered=False to continue inserting even if some fail (duplicates)
            result = self.collection.insert_many(entries, ordered=False)
            
            inserted_count = len(result.inserted_ids)
            skipped_count = len(entries) - inserted_count
            
            logger.info(f"Bulk insert: {inserted_count} inserted, {skipped_count} skipped (duplicates)")
            
            return {
                'inserted_count': inserted_count,
                'skipped_count': skipped_count
            }
            
        except Exception as e:
            # If it's a bulk write error, extract successful inserts
            if hasattr(e, 'details'):
                inserted_count = e.details.get('nInserted', 0)
                skipped_count = len(entries) - inserted_count
                logger.info(f"Bulk insert (with errors): {inserted_count} inserted, {skipped_count} skipped")
                return {
                    'inserted_count': inserted_count,
                    'skipped_count': skipped_count
                }
            else:
                logger.error(f"Error in bulk insert: {e}")
                raise
    
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
    
    def get_gwas_entry_by_phenotype(self, phenotype_code: str) -> Optional[Dict]:
        """
        Get GWAS entries by phenotype code (may return multiple for different sex/processing)
        
        Args:
            phenotype_code (str): Phenotype code
            
        Returns:
            list: List of GWAS entries matching phenotype code
        """
        try:
            entries = list(self.collection.find({'phenotype_code': phenotype_code}))
            
            for entry in entries:
                entry.pop('_id', None)
            
            return entries
            
        except Exception as e:
            logger.error(f"Error getting GWAS entries for phenotype {phenotype_code}: {e}")
            return []
    
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
            search_term (str, optional): Search term for display_name or description
            sex_filter (str, optional): Filter by sex ('both_sexes', 'male', 'female')
            limit (int): Maximum number of entries to return
            skip (int): Number of entries to skip (for pagination)
            
        Returns:
            list: List of GWAS entries
        """
        try:
            # Build query
            query = {}
            
            if search_term:
                # Use text search
                query['$text'] = {'$search': search_term}
            
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
            
            if search_term:
                query['$text'] = {'$search': search_term}
            
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
    
    def mark_as_downloaded(self, file_id: str, local_path: str, file_size: int) -> bool:
        """
        Mark a GWAS entry as downloaded and update cache information
        
        Args:
            file_id (str): File ID (filename)
            local_path (str): Local file path where file is cached
            file_size (int): File size in bytes
            
        Returns:
            bool: True if successful
        """
        try:
            update_data = {
                'downloaded': True,
                'local_path': local_path,
                'file_size': file_size,
                'last_accessed': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc)
            }
            
            result = self.collection.update_one(
                {'file_id': file_id},
                {'$set': update_data}
            )
            
            if result.modified_count > 0:
                logger.info(f"Marked as downloaded: {file_id} -> {local_path}")
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
    
    def delete_gwas_entry(self, file_id: str) -> bool:
        """
        Delete a GWAS entry
        
        Args:
            file_id (str): File ID (filename)
            
        Returns:
            bool: True if successful
        """
        try:
            result = self.collection.delete_one({'file_id': file_id})
            
            if result.deleted_count > 0:
                logger.info(f"Deleted GWAS entry: {file_id}")
                return True
            else:
                logger.warning(f"GWAS entry not found: {file_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error deleting GWAS entry {file_id}: {e}")
            return False
    
    def clear_collection(self) -> int:
        """
        Clear all entries from the collection (use with caution!)
        
        Returns:
            int: Number of deleted entries
        """
        try:
            result = self.collection.delete_many({})
            deleted_count = result.deleted_count
            logger.warning(f"Cleared GWAS library collection: {deleted_count} entries deleted")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return 0
    
    def get_downloaded_entries(self) -> List[Dict]:
        """
        Get all entries that have been downloaded
        
        Returns:
            list: List of downloaded GWAS entries
        """
        try:
            cursor = self.collection.find({'downloaded': True})
            
            entries = []
            for entry in cursor:
                entry.pop('_id', None)
                entries.append(entry)
            
            return entries
            
        except Exception as e:
            logger.error(f"Error getting downloaded entries: {e}")
            return []
    
    def get_most_popular(self, limit: int = 10) -> List[Dict]:
        """
        Get most popular (most downloaded) GWAS entries
        
        Args:
            limit (int): Number of entries to return
            
        Returns:
            list: List of popular GWAS entries
        """
        try:
            cursor = self.collection.find().sort('download_count', -1).limit(limit)
            
            entries = []
            for entry in cursor:
                entry.pop('_id', None)
                entries.append(entry)
            
            return entries
            
        except Exception as e:
            logger.error(f"Error getting most popular entries: {e}")
            return []
