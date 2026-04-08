import os
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user_id, set_deps
from app.api.routers import router
from app.core.status_tracker import StatusTracker, status_tracker


class MockHypothesisService:
    def __init__(self):
        self.hypotheses = {
            "h1": {
                "id": "h1",
                "user_id": "user1",
                "variant": "rs123",
                "phenotype": "Obesity",
                "enrich_id": "e1",
                "go_id": "GO:0008150",
                "summary": "A test hypothesis",
                "graph": {"nodes": [], "edges": [], "probability": 0.95},
                "status": "completed",
                "project_id": "p1",
            }
        }

    def get_hypotheses(self, user_id=None, hypothesis_id=None):
        if hypothesis_id:
            hyp = self.hypotheses.get(hypothesis_id)
            if hyp and hyp.get("user_id") == user_id:
                return hyp
            return None

        return [v for v in self.hypotheses.values() if v.get("user_id") == user_id]

    def get_hypothesis_by_enrich(self, user_id, enrich_id):
        for h in self.hypotheses.values():
            if h.get("enrich_id") == enrich_id and h.get("user_id") == user_id:
                return h
        return None

    def delete_hypothesis(self, user_id, hypothesis_id):
        hyp = self.hypotheses.get(hypothesis_id)
        if hyp and hyp.get("user_id") == user_id:
            del self.hypotheses[hypothesis_id]
            return {"message": "Hypothesis deleted"}, 200
        return {"message": "Hypothesis not found or not authorized"}, 404

    def bulk_delete_hypotheses(self, user_id, hypothesis_ids):
        if not hypothesis_ids or not isinstance(hypothesis_ids, list):
            return {"message": "Invalid hypothesis_ids format. Expected a non-empty list."}, 400

        found = [hid for hid in hypothesis_ids if hid in self.hypotheses and self.hypotheses[hid].get("user_id") == user_id]
        deleted_count = len(found)
        for hid in found:
            del self.hypotheses[hid]

        if deleted_count == 0:
            return {"message": "No hypotheses found or not authorized"}, 404

        failed = [hid for hid in hypothesis_ids if hid not in found]
        if failed:
            return {
                "message": f"{deleted_count} hypotheses deleted, {len(failed)} failed",
                "deleted_count": deleted_count,
                "successful": found,
                "failed": [{"id": hid, "reason": "Not found or not authorized"} for hid in failed],
            }, 207

        return {
            "message": f"All {deleted_count} hypotheses deleted",
            "deleted_count": deleted_count,
            "successful": found,
            "failed": [],
        }, 200


class MockEnrichmentService:
    def get_enrich(self, user_id, enrich_id):
        if enrich_id == "e1":
            return {
                "id": "e1",
                "causal_graph": {"graph": {"prob": {"value": 0.95}}},
            }
        return None


@pytest.fixture(autouse=True)
def _reset_status_tracker():
    status_tracker_instance = StatusTracker()
    status_tracker_instance.task_history.clear()
    status_tracker_instance.completed_hypotheses.clear()
    if hasattr(StatusTracker, "_task_handler"):
        delattr(StatusTracker, "_task_handler")
    yield


@pytest.fixture
def api_client(monkeypatch):
    app = FastAPI()
    app.include_router(router)

    # override auth
    app.dependency_overrides[get_current_user_id] = lambda: "user1"

    set_deps({
        "hypotheses": MockHypothesisService(),
        "enrichment": MockEnrichmentService(),
    })

    yield TestClient(app)

    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def env_service_token(monkeypatch):
    monkeypatch.setenv("PREFECT_SERVICE_TOKEN", "test-token")
    monkeypatch.setenv("API_HOST", "localhost")
    monkeypatch.setenv("API_PORT", "5000")
    yield