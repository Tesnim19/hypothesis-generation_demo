import os
from datetime import datetime

import requests
from status_tracker import TaskState

from app.core import utils
from app.core.utils import allowed_file, serialize_datetime_fields, emit_task_update


def test_allowed_file_extensions():
    assert allowed_file("test.tsv")
    assert not allowed_file("test.exe")


def test_serialize_datetime_fields_nested():
    data = {
        "date": datetime(2026, 3, 23, 12, 0, 0),
        "nested": {
            "date": datetime(2026, 3, 23, 13, 0, 0),
        },
        "items": [{"date": datetime(2026, 3, 23, 14, 0, 0)}],
    }
    serialized = serialize_datetime_fields(data)
    assert serialized["date"] == "2026-03-23T12:00:00"
    assert serialized["nested"]["date"] == "2026-03-23T13:00:00"
    assert serialized["items"][0]["date"] == "2026-03-23T14:00:00"


def test_emit_task_update_no_request(monkeypatch):
    # Set service token and mock requests.post to avoid external call
    monkeypatch.setenv("PREFECT_SERVICE_TOKEN", "token")
    monkeypatch.setenv("API_HOST", "localhost")
    monkeypatch.setenv("API_PORT", "5000")

    # Mock requests.post to avoid outbound network e2e
    called = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        called["url"] = url
        called["json"] = json
        class R:
            status_code = 200
        return R()

    monkeypatch.setattr(requests, "post", fake_post)

    # Provide minimal state_tracker history
    # When called with progress=0, it uses status_tracker.calculate_progress and then add_update.
    emit_task_update("h1", "Task A", TaskState.STARTED, progress=10)

    assert "hypothesis_id" in called["json"]
    assert called["json"]["hypothesis_id"] == "h1"
