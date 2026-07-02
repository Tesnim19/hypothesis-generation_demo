"""Helpers for injecting demo templates into project API responses."""

from __future__ import annotations

from loguru import logger

from src.db import (
    AnalysisHandler,
    DemoTemplateHandler,
    FileHandler,
    HypothesisHandler,
    ProjectHandler,
)
from src.services.demo.access import resolve_project_access
from src.utils import get_population_label, normalize_status_responses, project_running_task


def get_template_owner_user_id(template: dict) -> str:
    return DemoTemplateHandler.get_owner_user_id(template)


def resolve_hypothesis_data_user_id(
    demo_templates: DemoTemplateHandler,
    hypotheses: HypothesisHandler,
    current_user_id: str,
    hypothesis_id: str,
) -> str | None:
    """Return the user_id to read a hypothesis from, or None if not accessible."""
    hypothesis = hypotheses.get_hypotheses(current_user_id, hypothesis_id)
    if hypothesis:
        return current_user_id

    hypothesis = hypotheses.get_hypothesis_by_id(hypothesis_id)
    if not hypothesis:
        return None

    project_id = hypothesis.get("project_id")
    if not project_id:
        return None

    access = resolve_project_access(demo_templates, current_user_id, project_id)
    if hypothesis.get("user_id") != access.owner_user_id:
        return None
    return access.owner_user_id


def _fork_metadata(
    demo_templates: DemoTemplateHandler, user_id: str, template_project_id: str
) -> dict:
    forked_project_id = demo_templates.get_user_fork(user_id, template_project_id)
    return {
        "has_forked": bool(forked_project_id),
        "forked_project_id": forked_project_id,
    }


def build_project_summary(
    *,
    project_id: str,
    name: str,
    phenotype: str,
    created_at,
    data_user_id: str,
    projects: ProjectHandler,
    analysis: AnalysisHandler,
    hypotheses: HypothesisHandler,
    files: FileHandler,
    population=None,
    ref_genome=None,
    demo_meta: dict | None = None,
) -> dict:
    enhanced: dict = {
        "id": project_id,
        "name": name,
        "phenotype": phenotype or "",
        "created_at": created_at,
    }

    project_doc = projects.get_projects(data_user_id, project_id)
    gwas_file_id = project_doc.get("gwas_file_id") if project_doc else None
    if gwas_file_id:
        try:
            file_metadata = files.get_file_metadata(data_user_id, gwas_file_id)
            if file_metadata:
                enhanced["gwas_file"] = file_metadata.get("download_url")
                enhanced["gwas_records_count"] = file_metadata.get("record_count")
        except Exception as file_e:
            logger.warning(f"Could not load GWAS metadata for demo project {project_id}: {file_e}")

    try:
        analysis_state = projects.load_analysis_state(data_user_id, project_id)
        raw = analysis_state.get("status") if analysis_state else None
        enhanced["status"] = normalize_status_responses(raw)
        enhanced["running_task"] = project_running_task(analysis_state)
    except Exception as state_e:
        logger.warning(f"Could not load analysis state for project {project_id}: {state_e}")
        enhanced["status"] = normalize_status_responses("Completed")
        enhanced["running_task"] = project_running_task(
            {"status": "Completed", "message": "Analysis completed successfully."}
        )

    if population is not None:
        enhanced["population"] = get_population_label(population)
    elif project_doc:
        enhanced["population"] = get_population_label(project_doc.get("population"))
    if ref_genome is not None:
        enhanced["ref_genome"] = ref_genome
    elif project_doc:
        enhanced["ref_genome"] = project_doc.get("ref_genome")

    total_credible_sets = 0
    total_variants = 0
    try:
        credible_sets_raw = analysis.get_credible_sets_for_project(data_user_id, project_id)
        if credible_sets_raw:
            total_credible_sets = len(credible_sets_raw)
            total_variants = sum(cs.get("variants_count", 0) for cs in credible_sets_raw)
    except Exception as cs_e:
        logger.warning(f"Could not load credible sets for {project_id}: {cs_e}")

    enhanced["total_credible_sets_count"] = total_credible_sets
    enhanced["total_variants_count"] = total_variants

    hypothesis_count = 0
    try:
        all_hyp = hypotheses.get_hypotheses(data_user_id)
        if isinstance(all_hyp, list):
            hypothesis_count = sum(1 for h in all_hyp if h.get("project_id") == project_id)
        elif all_hyp and all_hyp.get("project_id") == project_id:
            hypothesis_count = 1
    except Exception as hyp_e:
        logger.warning(f"Could not count hypotheses for {project_id}: {hyp_e}")

    enhanced["hypothesis_count"] = hypothesis_count

    if demo_meta:
        enhanced.update(demo_meta)

    return enhanced


def build_demo_template_summaries(
    *,
    current_user_id: str,
    demo_templates: DemoTemplateHandler,
    projects: ProjectHandler,
    analysis: AnalysisHandler,
    hypotheses: HypothesisHandler,
    files: FileHandler,
    existing_project_ids: set[str],
) -> list[dict]:
    summaries: list[dict] = []
    for template in demo_templates.list_templates(active_only=True):
        project_id = template["template_project_id"]
        data_user_id = get_template_owner_user_id(template)
        fork_meta = _fork_metadata(demo_templates, current_user_id, project_id)
        demo_meta = {
            "is_demo": True,
            "source_template_slug": template["slug"],
            **fork_meta,
        }

        if project_id in existing_project_ids:
            continue

        project_doc = projects.get_projects(data_user_id, project_id)
        summaries.append(
            build_project_summary(
                project_id=project_id,
                name=template["display_name"],
                phenotype=template.get("phenotype") or (project_doc or {}).get("phenotype", ""),
                created_at=(project_doc or {}).get("created_at"),
                data_user_id=data_user_id,
                projects=projects,
                analysis=analysis,
                hypotheses=hypotheses,
                files=files,
                demo_meta=demo_meta,
            )
        )
    return summaries


def apply_demo_flags_to_owned_project(
    enhanced: dict,
    *,
    current_user_id: str,
    demo_templates: DemoTemplateHandler,
) -> dict:
    template = demo_templates.get_template_by_project_id(enhanced["id"])
    if not template:
        enhanced.setdefault("is_demo", False)
        return enhanced

    fork_meta = _fork_metadata(demo_templates, current_user_id, enhanced["id"])
    enhanced.update(
        {
            "is_demo": True,
            "name": template["display_name"],
            "source_template_slug": template["slug"],
            **fork_meta,
        }
    )
    return enhanced


def resolve_fork_project_id(
    *,
    demo_templates: DemoTemplateHandler,
    projects: ProjectHandler,
    current_user_id: str,
    project_id: str,
    template: dict,
) -> tuple[str, bool]:
    """Return project_id to use for writes and whether a fork was created."""
    template_owner_id = get_template_owner_user_id(template)
    template_project_id = template["template_project_id"]

    existing_fork = demo_templates.get_user_fork(current_user_id, template_project_id)
    if existing_fork:
        return existing_fork, False

    forked_id = projects.fork_project_from_template(
        template_owner_id,
        template_project_id,
        current_user_id,
        new_name=f"{template['display_name']} (from sample)",
        template_slug=template["slug"],
    )
    demo_templates.save_user_fork(
        current_user_id, template_project_id, forked_id, template["slug"]
    )
    return forked_id, True
