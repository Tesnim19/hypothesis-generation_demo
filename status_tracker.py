from datetime import datetime, timezone
from enum import Enum
from config import Config, create_dependencies

class TaskState(Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class StatusTracker:
    _instance = None
    _db = None  # Class-level database instance
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.task_history = {}  # hypothesis_id -> list of task updates
            cls._instance.completed_hypotheses = set()
        return cls._instance
    
    @classmethod
    def initialize(cls, task_handler):
        """Initialize the status tracker with a task handler"""
        cls._task_handler = task_handler
        # Also set on the instance for consistency
        if cls._instance:
            cls._instance._task_handler = task_handler
    
    def add_update(self, hypothesis_id, progress, task_name, state, details=None, error=None):
        if not hypothesis_id:
              raise ValueError("Hypothesis ID is required")
        if not isinstance(state, TaskState):
            raise ValueError("Invalid task state provided")

        if hypothesis_id not in self.task_history:
            self.task_history[hypothesis_id] = []
            
        update = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec='milliseconds') + "Z",
            "task": task_name,
            "state": state.value,
            "progress": progress
        }
        
        if details:
            update["details"] = details
        if error:
            update["error"] = error
            
        self.task_history[hypothesis_id].append(update)

        # Persist to DB on completion or failure
        if state in [TaskState.COMPLETED, TaskState.FAILED]:
            if task_name in ["Creating enrich data", "Generating hypothesis"] or task_name.startswith("Verifying existence") and progress == 80:
                self._persist_and_clear(hypothesis_id)
    
    def _ensure_db_connection(self):
        """Ensure database connection is available, initialize if needed"""
        if not hasattr(self, '_db') or self._db is None:
            # Try to get from class variable
            if self.__class__._db is not None:
                self._db = self.__class__._db
            else:
                # Try to create minimal task handler directly first
                try:
                    from db.task_handler import TaskHandler
                    import os
                    mongodb_uri = os.getenv("MONGODB_URI")
                    db_name = os.getenv("DB_NAME", "default")
                    if mongodb_uri and db_name:
                        self._db = TaskHandler(mongodb_uri, db_name)
                        self.__class__._db = self._db
                        return self._db
                except Exception:
                    pass
                    
                # If direct approach fails, try full dependency creation as fallback
                try:
                    config = Config.from_env()
                    deps = create_dependencies(config)
                    self._db = deps['tasks']
                    self.__class__._db = self._db
                except Exception as e:
                    # If everything fails, return None and skip persistence
                    return None
        return self._db

    def _persist_and_clear(self, hypothesis_id):
        """Persist task history to DB and clear from memory"""
        if hypothesis_id in self.task_history:
            # Ensure database connection
            db = self._ensure_db_connection()
            
            # Check if database connection is available
            if db is None:
                # If no database connection available, just clear from memory without persisting
                del self.task_history[hypothesis_id]
                self.completed_hypotheses.add(hypothesis_id)
                return
            
            # Check if task handler is available
            if not hasattr(self.__class__, '_task_handler') or self.__class__._task_handler is None:
                # If no task handler available, just clear from memory without persisting
                del self.task_history[hypothesis_id]
                self.completed_hypotheses.add(hypothesis_id)
                return
                
            # Get existing history from DB
            db_history = self.__class__._task_handler.get_task_history(hypothesis_id) or []
            new_history = self.task_history[hypothesis_id]
            
            # Combine and deduplicate
            combined = db_history + new_history
            deduplicated = {}
            for update in combined:
                key = (update['task'], update['timestamp'])
                deduplicated[key] = update
            
            # Sort by timestamp
            final_history = sorted(deduplicated.values(), key=lambda x: x['timestamp'])
            
            # Save to DB
            self.__class__._task_handler.save_task_history(hypothesis_id, final_history)
            
            # Clear from memory
            del self.task_history[hypothesis_id]
            self.completed_hypotheses.add(hypothesis_id)
    
    def get_history(self, hypothesis_id):
        """Get complete task history from memory and DB without duplicates"""
        memory_history = self.task_history.get(hypothesis_id, [])
        
        # Only get DB history if task handler is available and hypothesis is completed
        db_history = []
        if (hasattr(self.__class__, '_task_handler') and 
            self.__class__._task_handler is not None and 
            hypothesis_id in self.completed_hypotheses):
            db_history = self.__class__._task_handler.get_task_history(hypothesis_id) or []
        
        # Combine histories
        combined_history = memory_history + db_history
        
        if not combined_history:
            return []
        
        # Create a dictionary with timestamp as key to remove duplicates
        deduplicated = {}
        for update in combined_history:
            key = (update['task'], update['timestamp'])
            deduplicated[key] = update
        
        # Convert back to list and sort by timestamp
        sorted_history = sorted(deduplicated.values(), key=lambda x: x['timestamp'])
        
        return sorted_history
    
    def get_latest_state(self, hypothesis_id):
        history = self.task_history.get(hypothesis_id, [])
        return history[-1] if history else None
     
    def calculate_progress(self, task_history):
        """
        Calculate the progress of a task based on its history.

        Args:
            task_history (list): A list of task updates.

        Returns:
            float: A percentage representing the progress (0 to 100).
        """
        if not task_history:
            return 0.0  # No tasks, no progress
        
        # Define task weights and their process group
        enrichment_tasks = {
            "Verifying existence of enrichment data": 10,
            "Getting candidate genes": 10,
            "Predicting causal gene": 20,
            "Getting relevant gene proof": 20,
            "Creating enrich data": 20
        }

        hypothesis_tasks = {
            "Verifying existence of hypothesis data": 2,  # Added weight
            "Getting enrichement data": 2,
            "Getting gene data": 2,
            "Querying gene data": 3,
            "Querying variant data": 3,
            "Querying phenotype data": 3,
            "Generating graph summary": 3,
            "Generating hypothesis": 2
        }

        # Filter only completed tasks
        filtered_history = [
            task for task in task_history 
            if task.get('state') == TaskState.COMPLETED.value
        ]

        enrichment_progress = 0
        hypothesis_progress = 0

        for task in filtered_history:
            task_name = task['task']
            if task_name in enrichment_tasks:
                enrichment_progress += enrichment_tasks[task_name]
            elif task_name in hypothesis_tasks:
                hypothesis_progress += hypothesis_tasks[task_name]

        # Calculate total progress
        total_enrichment_weight = sum(enrichment_tasks.values())  # 80
        total_hypothesis_weight = sum(hypothesis_tasks.values())  # 20

        # Normalize to percentages
        enrichment_percentage = (enrichment_progress / total_enrichment_weight) * 80
        hypothesis_percentage = (hypothesis_progress / total_hypothesis_weight) * 20

        return round(min(enrichment_percentage + hypothesis_percentage, 100), 2)

# Global instance
status_tracker = StatusTracker()