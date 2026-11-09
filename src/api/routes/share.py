from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.auth import get_current_user_id
from src.api.dependencies import (
    get_enrichment_handler,
    get_gene_expression_handler,
    get_hypothesis_handler,
    get_share_link_handler,
)
from src.api.schemas import ErrorResponse
from src.api.schemas.share import (
    ShareLinkCreateRequest,
    ShareLinkListResponse,
    ShareLinkResponse,
    SharedSnapshot,
)
from src.db import (
    EnrichmentHandler,
    GeneExpressionHandler,
    HypothesisHandler,
    ShareLinkHandler,
)
from src.tasks import extract_probability
from src.utils import normalize_status_responses

router = APIRouter(prefix="/share-links", tags=["share"])

MAX_ACTIVE_LINKS = 20
MAX_RELATED = 20


def _is_completed(hypothesis: dict) -> bool:
    return (
        normalize_status_responses(hypothesis.get("status")) == "Completed"
        and bool(hypothesis.get("summary"))
        and bool(hypothesis.get("graph"))
    )


def _shared_entry(hypothesis: dict, enrichment: EnrichmentHandler, user_id: str) -> dict:
    entry = {
        "causal_gene": hypothesis.get("causal_gene"),
        "probability": extract_probability(hypothesis, enrichment, user_id),
        "status": normalize_status_responses(hypothesis.get("status")),
    }
    if _is_completed(hypothesis):
        enrich_id = hypothesis.get("enrich_id")
        enrich = enrichment.get_enrich(user_id, enrich_id) if enrich_id else None
        go_terms = (enrich or {}).get("GO_terms") or []
        entry.update(
            summary=hypothesis["summary"],
            graph=hypothesis["graph"],
            go_terms=go_terms,
            selected_go_term=next(
                (term for term in go_terms if term.get("id") == hypothesis.get("go_id")),
                None,
            ),
        )
    return entry


def build_share_snapshot(
    hypothesis: dict,
    hypotheses: HypothesisHandler,
    enrichment: EnrichmentHandler,
    gene_expression: GeneExpressionHandler,
) -> dict:
    """Freeze a completed hypothesis and its same-variant siblings into a public payload."""
    user_id = hypothesis["user_id"]
    project_id = hypothesis.get("project_id")
    variant = hypothesis.get("variant") or hypothesis.get("variant_id")
    variant_rsid = hypothesis.get("variant_rsid") or variant

    siblings = []
    tissue = None
    if project_id and variant:
        siblings = [
            h
            for h in hypotheses.get_variant_hypotheses(user_id, project_id, variant)
            if h.get("id") != hypothesis["id"]
        ]
        selection = gene_expression.get_tissue_selection(user_id, project_id, variant_rsid)
        tissue = selection.get("tissue_name") if selection else None

    related = sorted(
        (_shared_entry(h, enrichment, user_id) for h in siblings),
        key=lambda entry: entry["probability"] or 0,
        reverse=True,
    )
    for index, entry in enumerate(related, start=1):
        entry["key"] = f"h{index}"

    main = {
        **_shared_entry(hypothesis, enrichment, user_id),
        "key": "h0",
        "variant": variant,
        "variant_rsid": variant_rsid,
        "phenotype": hypothesis.get("phenotype"),
        "tissue": tissue,
        "warnings": [
            *hypothesis.get("enrichment_warnings", []),
            *hypothesis.get("warnings", []),
        ],
        "generated_at": hypothesis.get("created_at"),
    }
    return SharedSnapshot.model_validate(
        {
            "hypothesis": main,
            "related": related[:MAX_RELATED],
            "related_truncated": len(related) > MAX_RELATED,
        }
    ).model_dump()


@router.post(
    "",
    status_code=201,
    response_model=ShareLinkResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    summary="Create a public share link for a completed hypothesis",
)
async def create_share_link(
    body: ShareLinkCreateRequest,
    current_user_id: str = Depends(get_current_user_id),
    share_links: ShareLinkHandler = Depends(get_share_link_handler),
    hypotheses: HypothesisHandler = Depends(get_hypothesis_handler),
    enrichment: EnrichmentHandler = Depends(get_enrichment_handler),
    gene_expression: GeneExpressionHandler = Depends(get_gene_expression_handler),
):
    hypothesis = hypotheses.get_hypotheses(current_user_id, body.hypothesis_id)
    if not hypothesis:
        raise HTTPException(status_code=404, detail="Hypothesis not found or access denied.")
    if not _is_completed(hypothesis):
        raise HTTPException(status_code=409, detail="Only completed hypotheses can be shared.")
    if share_links.count_active_links(current_user_id, body.hypothesis_id) >= MAX_ACTIVE_LINKS:
        raise HTTPException(
            status_code=409,
            detail=f"A hypothesis can have at most {MAX_ACTIVE_LINKS} active share links.",
        )

    expires_at = (
        datetime.now(timezone.utc) + timedelta(days=body.expires_in_days)
        if body.expires_in_days
        else None
    )
    link = share_links.create_link(
        owner_user_id=current_user_id,
        hypothesis_id=body.hypothesis_id,
        project_id=hypothesis.get("project_id"),
        label=body.label or None,
        expires_at=expires_at,
        snapshot=build_share_snapshot(hypothesis, hypotheses, enrichment, gene_expression),
    )
    return ShareLinkResponse.model_validate(link)


@router.get(
    "",
    response_model=ShareLinkListResponse,
    summary="List a hypothesis's share links",
)
async def list_share_links(
    hypothesis_id: str = Query(...),
    current_user_id: str = Depends(get_current_user_id),
    share_links: ShareLinkHandler = Depends(get_share_link_handler),
):
    return ShareLinkListResponse(links=share_links.list_links(current_user_id, hypothesis_id))


@router.delete(
    "/{link_id}",
    response_model=ShareLinkResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Revoke a share link",
)
async def revoke_share_link(
    link_id: str,
    current_user_id: str = Depends(get_current_user_id),
    share_links: ShareLinkHandler = Depends(get_share_link_handler),
):
    link = share_links.revoke_link(current_user_id, link_id)
    if not link:
        raise HTTPException(status_code=404, detail="Share link not found.")
    return ShareLinkResponse.model_validate(link)
