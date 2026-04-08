import pytest

from app.api.dependencies import get_current_user_id
from app.api.routers import router


def test_get_hypothesis_by_id(api_client):
    response = api_client.get("/hypothesis", params={"id": "h1"})
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "h1"
    assert data["status"] == "completed"
    assert data["probability"] == 0.95
    assert data["enrichment_type"] == "standard"


def test_get_all_hypotheses(api_client):
    response = api_client.get("/hypothesis")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert any(item["id"] == "h1" for item in data)


def test_post_hypothesis_missing_go(api_client):
    response = api_client.post("/hypothesis", params={"id": "e1"})
    assert response.status_code == 400


def test_post_hypothesis_not_found(api_client, monkeypatch):
    # Simulate enrichment with no hypothesis available
    class EmptyHypothesisService:
        def get_hypothesis_by_enrich(self, user_id, enrich_id):
            return None

    from app.api.dependencies import set_deps

    set_deps({"hypotheses": EmptyHypothesisService(), "enrichment": object()})

    response = api_client.post("/hypothesis", params={"id": "e1", "go": "GO:0008150"})
    assert response.status_code == 404


def test_post_hypothesis_success(api_client, monkeypatch):
    # Replace actual workflow to focus on routing behavior
    monkeypatch.setattr(
        "app.api.routers.hypothesis.hypothesis_flow",
        lambda current_user_id, hypothesis_id, enrich_id, go_id: (
            {"summary": "test summary", "graph": {"nodes": [], "edges": []}}, 200
        ),
    )

    response = api_client.post("/hypothesis", params={"id": "e1", "go": "GO:0008150"})
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "h1"
    assert data["summary"] == "test summary"
    assert "graph" in data


def test_delete_hypothesis_missing_id(api_client):
    response = api_client.delete("/hypothesis")
    assert response.status_code == 400


def test_delete_hypothesis_success(api_client):
    response = api_client.delete("/hypothesis", params={"hypothesis_id": "h1"})
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Hypothesis deleted"


def test_bulk_delete_hypotheses(api_client):
    response = api_client.post("/hypothesis/delete", json={"hypothesis_ids": ["h1"]})
    assert response.status_code in (200, 207)
    payload = response.json()
    assert "deleted_count" in payload
