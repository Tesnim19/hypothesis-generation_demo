from datetime import datetime, timezone
from enum import Enum

class TaskState(Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class StatusTracker:
    _instance = None
    _task_handler = None
    _cache = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, task_handler, cache):
        """Initialize the status tracker with a task handler and cache"""
        cls._task_handler = task_handler
        cls._cache = cache
    
    def add_update(self, hypothesis_id, progress, task_name, state, details=None, error=None):
        if not hypothesis_id:
              raise ValueError("Hypothesis ID is required")
        if not isinstance(state, TaskState):
            raise ValueError("Invalid task state provided")
            
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
            
        self._cache.add_update(hypothesis_id, update)

        # Persist to DB on completion or failure
        if state in [TaskState.COMPLETED, TaskState.FAILED]:
            if task_name in ["Creating enrich data", "Generating hypothesis"] or task_name.startswith("Verifying existence") and progress == 80:
                self._persist_and_clear(hypothesis_id)
    
    def _persist_and_clear(self, hypothesis_id):
        """Persist task history to DB and clear from cache """
        # Get existing history from DB
        db_history = self._task_handler.get_task_history(hypothesis_id) or []
        new_history = self._cache.get_history(hypothesis_id) or [] 
        
        # Combine and deduplicate
        combined = db_history + new_history
        deduplicated = {}
        for update in combined:
            key = (update['task'], update['timestamp'])
            deduplicated[key] = update
        
        # Sort by timestamp
        final_history = sorted(deduplicated.values(), key=lambda x: x['timestamp'])
        
        # Save to DB
        self._task_handler.save_task_history(hypothesis_id, final_history)
        # Clear from Redis and mark persisted
        self._cache.clear_history(hypothesis_id)
    
    def get_history(self, hypothesis_id):
        """Get complete task history from memory and DB without duplicates"""
        redis_history = self._cache.get_history(hypothesis_id) or []
        db_history = self._task_handler.get_task_history(hypothesis_id) if self._cache.is_persisted(hypothesis_id) else []
        
        # Combine histories
        combined_history = redis_history + db_history 
        
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
        latest = self._cache.get_latest(hypothesis_id)
        return latest
     
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

    def recover_from_cache(self):
        """startup recover: for each in-progress hypothesis in Redis, persist its history to DB and clear."""
        ids = self._cache.list_inprogress()
        for idx,hyp_id in enumerate(ids):
            latest = self._cache.get_latest(hyp_id)
            if latest:
                latest["timestamp"] = latest.get("timestamp", datetime.now(timezone.utc).isoformat(timespec='milliseconds') + "Z")
                latest["state"] = latest.get("state", TaskState.FAILED.value)
                latest["progress"] = latest.get("progress", 0)
                self._cache.add_update(hyp_id, latest)
        
            self._persist_and_clear(hyp_id)

# Global instance
status_tracker = StatusTracker()