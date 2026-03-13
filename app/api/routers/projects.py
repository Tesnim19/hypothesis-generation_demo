from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import JSONResponse

from loguru import logger

from app.api.dependencies import get_current_user_id, get_deps
from app.workers.tasks.project import get_project_with_full_data
from app.core.utils import serialize_datetime_fields

router = APIRouter()


@router.get("/projects")
async def get_projects(
    id: str | None = Query(None),
    current_user_id: str = Depends(get_current_user_id),
):
    projects = get_deps()["projects"]
    analysis = get_deps()["analysis"]
    hypotheses = get_deps()["hypotheses"]
    enrichment = get_deps()["enrichment"]
    gene_expression = get_deps().get("gene_expression")

    if id:
        response_data, status_code = get_project_with_full_data(
            projects,
            analysis,
            hypotheses,
            enrichment,
            current_user_id,
            id,
            gene_expression_handler=gene_expression,
        )
        if status_code == 200:
            response_data = serialize_datetime_fields(response_data)
        return JSONResponse(content=response_data, status_code=status_code)

    raw_projects = projects.get_projects(current_user_id)
    files = get_deps()["files"]
    enhanced_projects: list[dict] = []

    for project in raw_projects:
        enhanced: dict = {
            "id": project["id"],
            "name": project["name"],
            "phenotype": project.get("phenotype", ""),
            "created_at": project.get("created_at"),
        }

        file_metadata = files.get_file_metadata(current_user_id, project["gwas_file_id"])
        enhanced["gwas_file"] = file_metadata["download_url"]
        enhanced["gwas_records_count"] = file_metadata["record_count"]

        try:
            analysis_state = projects.load_analysis_state(current_user_id, project["id"])
            enhanced["status"] = (
                analysis_state.get("status", "Not_started")
                if analysis_state
                else "Not_started"
            )
        except Exception as state_e:
            logger.warning(f"Could not load analysis state for project {project['id']}: {state_e}")
            enhanced["status"] = "Completed"

        enhanced["population"] = project.get("population")
        enhanced["ref_genome"] = project.get("ref_genome")

        total_credible_sets = 0
        total_variants = 0
        try:
            credible_sets_raw = analysis.get_credible_sets_for_project(
                current_user_id, project["id"]
            )
            if credible_sets_raw and isinstance(credible_sets_raw, list):
                total_credible_sets = len(credible_sets_raw)
                total_variants = sum(
                    cs.get("variants_count", 0) for cs in credible_sets_raw
                )
        except Exception as cs_e:
            logger.warning(f"Could not load credible sets for {project['id']}: {cs_e}")

        enhanced["total_credible_sets_count"] = total_credible_sets
        enhanced["total_variants_count"] = total_variants

        hypothesis_count = 0
        try:
            all_hyp = hypotheses.get_hypotheses(current_user_id)
            if isinstance(all_hyp, list):
                hypothesis_count = sum(
                    1 for h in all_hyp if h.get("project_id") == project["id"]
                )
            elif all_hyp and all_hyp.get("project_id") == project["id"]:
                hypothesis_count = 1
        except Exception as hyp_e:
            logger.warning(f"Could not count hypotheses for {project['id']}: {hyp_e}")

        enhanced["hypothesis_count"] = hypothesis_count
        enhanced_projects.append(enhanced)

    return {"projects": serialize_datetime_fields(enhanced_projects)}


@router.delete("/projects")
async def delete_project(
    id: str | None = Query(None),
    current_user_id: str = Depends(get_current_user_id),
):
    if not id:
        raise HTTPException(status_code=400, detail="Project ID is required")
    projects = get_deps()["projects"]
    success = projects.delete_project(current_user_id, id)
    if success:
        return {"message": "Project deleted successfully"}
    raise HTTPException(status_code=404, detail="Project not found or access denied")


@router.post("/projects/delete")
async def bulk_delete_projects(
    data: dict = Body(...),
    current_user_id: str = Depends(get_current_user_id),
):
    projects = get_deps()["projects"]
    project_ids = data.get("project_ids")

    if not project_ids:
        raise HTTPException(
            status_code=400, detail="project_ids is required in request body"
        )
    if not isinstance(project_ids, list):
        raise HTTPException(status_code=400, detail="project_ids must be a list")
    if not project_ids:
        raise HTTPException(
            status_code=400, detail="project_ids list cannot be empty"
        )

    result = projects.bulk_delete_projects(current_user_id, project_ids)

    if result and isinstance(result, dict):
        if result["success"]:
            return {
                "message": f"Successfully deleted {result['deleted_count']} project(s)",
                "deleted_count": result["deleted_count"],
                "total_requested": result["total_requested"],
            }
        return JSONResponse(
            content={
                "message": (
                    f"Partially deleted {result['deleted_count']}/{result['total_requested']}"
                    " project(s)"
                ),
                "deleted_count": result["deleted_count"],
                "total_requested": result["total_requested"],
                "errors": result.get("errors"),
            },
            status_code=207,
        )
    raise HTTPException(status_code=500, detail="Failed to delete projects")
