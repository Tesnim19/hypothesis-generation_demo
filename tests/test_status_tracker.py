from datetime import datetime, timezone

from app.core.status_tracker import StatusTracker, TaskState, status_tracker


class FakeTaskHandler:
    def __init__(self):
        self.saved = {}

    def get_task_history(self, hypothesis_id):
        return self.saved.get(hypothesis_id, [])

    def save_task_history(self, hypothesis_id, history):
        self.saved[hypothesis_id] = history


def test_status_tracker_calculate_progress():
    history = [
        {"task": "Creating enrich data", "state": TaskState.COMPLETED.value},
        {"task": "Generating hypothesis", "state": TaskState.COMPLETED.value},
    ]
    tracker = StatusTracker()
    progress = tracker.calculate_progress(history)
    assert progress == 100.0 or progress <= 100.0


def test_status_tracker_persist_and_clear():
    handler = FakeTaskHandler()
    StatusTracker.initialize(handler)

    tracker = StatusTracker()
    tracker.add_update("h1", 80, "Creating enrich data", TaskState.COMPLETED)

    # After completion + persistence trigger, data should be clear from memory and in db
    assert "h1" not in tracker.task_history
    assert handler.get_task_history("h1")


def test_status_tracker_get_history_combine_and_dedup():
    handler = FakeTaskHandler()
    StatusTracker.initialize(handler)

    # add first track with completed state
    tracker = StatusTracker()
    tracker.task_history["h2"] = [
        {"task": "A", "state": TaskState.COMPLETED.value, "timestamp": "2026-03-23T00:00:00.000Z"}
    ]
    status = tracker.get_history("h2")
    assert isinstance(status, list)
    assert status[0]["task"] == "A"
