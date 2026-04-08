import json

from app.workers.tasks.enrichment import (
    parse_prolog_graphs,
    extract_probability,
    get_related_hypotheses,
    extract_causal_gene_from_graph,
)


class MockEnrichment:
    def get_enrich(self, user_id, enrich_id):
        return {
            "causal_graph": {"graph": {"prob": {"value": 0.45}}}
        }


class MockHypotheses:
    def __init__(self):
        self.calls = 0

    def get_hypotheses(self, user_id):
        self.calls += 1
        return [
            {
                "id": "h1",
                "project_id": "p1",
                "variant": "rs123",
                "causal_gene": "GENE1",
                "go_id": "GO:0008150",
                "status": "completed",
                "graph": {"probability": 0.45},
            },
            {
                "id": "h2",
                "project_id": "p1",
                "variant": "rs123",
                "causal_gene": "GENE2",
                "go_id": "GO:0008151",
                "status": "completed",
                "graph": {"probability": 0.60},
            },
        ]


def test_parse_prolog_graphs_valid_and_invalid():
    raw = {"response": [json.dumps({"a": 1}), "invalid json"]}
    parsed = parse_prolog_graphs(raw)
    assert isinstance(parsed, list)
    assert parsed[0] == {"a": 1}


def test_extract_probability_from_graph():
    hypothesis = {"graph": {"probability": 0.9}, "id": "h1"}
    assert extract_probability(hypothesis, None, "user1") == 0.9


def test_extract_probability_from_enrichment_when_no_graph():
    hypothesis = {"enrich_id": "e1", "id": "h1"}
    assert extract_probability(hypothesis, MockEnrichment(), "user1") == 0.45


def test_get_related_hypotheses_project_variant():
    current = {"id": "h1", "project_id": "p1", "variant": "rs123"}
    related = get_related_hypotheses(current, MockHypotheses(), MockEnrichment(), "user1")
    assert len(related) == 2
    assert related[0]["probability"] >= related[1]["probability"]


def test_get_related_hypotheses_no_project_or_variant():
    current = {"id": "h3"}
    related = get_related_hypotheses(current, MockHypotheses(), MockEnrichment(), "user1")
    assert len(related) == 1


def test_extract_causal_gene_from_graph():
    graph = {
        "nodes": [
            {"id": "rs123", "type": "snp"},
            {"id": "GENE1", "type": "gene", "name": "GENE1"}
        ],
        "edges": [{"source": "rs123", "target": "GENE1", "label": "affects"}],
    }
    gene_id, gene_name = extract_causal_gene_from_graph(graph, [{"id": "rs123"}])
    assert gene_id == "GENE1"
    assert gene_name == "GENE1"
