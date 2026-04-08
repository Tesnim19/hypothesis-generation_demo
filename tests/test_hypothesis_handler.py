from types import SimpleNamespace

from db.hypothesis_handler import HypothesisHandler


class FakeCollection:
    def __init__(self, docs=None):
        self.docs = docs or []

    # Simulates MoogoDB's methods
    def find(self, query):
        if "id" in query and "$in" in query["id"]:
            return [d for d in self.docs if d["id"] in query["id"]["$in"] and d["user_id"] == query.get("user_id")]
        if query.get("user_id") and query.get("project_id"):
            return [d for d in self.docs if d.get("user_id") == query.get("user_id") and d.get("project_id") == query.get("project_id")]
        return [d for d in self.docs if d.get("user_id") == query.get("user_id")]

    def delete_many(self, query):
        ids = query["id"]["$in"]
        removed = [d for d in self.docs if d["id"] in ids and d["user_id"] == query.get("user_id")]
        self.docs = [d for d in self.docs if d not in removed]
        return SimpleNamespace(deleted_count=len(removed))

    def delete_one(self, query):
        for d in self.docs:
            if d["id"] == query.get("id") and d.get("user_id") == query.get("user_id"):
                self.docs.remove(d)
                return SimpleNamespace(deleted_count=1)
        return SimpleNamespace(deleted_count=0)

    def find_one(self, query):
        for d in self.docs:
            if d.get("id") == query.get("id") and d.get("user_id") == query.get("user_id"):
                return d
        return None


def test_hypothesis_handler_bulk_delete():
    handler = HypothesisHandler.__new__(HypothesisHandler)
    handler.hypothesis_collection = FakeCollection([
        {"id": "h1", "user_id": "u1", "enrich_id": "e1"},
        {"id": "h2", "user_id": "u1", "enrich_id": "e2"},
    ])
    handler.enrich_collection = FakeCollection([
        {"id": "e1", "user_id": "u1"},
        {"id": "e2", "user_id": "u1"},
    ])

    result, status = handler.bulk_delete_hypotheses("u1", ["h1", "h2"])
    assert status == 200
    assert result["deleted_count"] == 2
    assert result["enrichments_deleted"] == 2


def test_hypothesis_handler_delete_hypothesis_not_found():
    handler = HypothesisHandler.__new__(HypothesisHandler)
    handler.hypothesis_collection = FakeCollection([
        {"id": "h1", "user_id": "u1", "enrich_id": "e1"}
    ])
    handler.enrich_collection = FakeCollection([])

    result, status = handler.delete_hypothesis("u1", "h2")
    assert status == 404
    assert "not found" in result["message"].lower()
