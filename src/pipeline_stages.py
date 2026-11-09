"""Analysis pipeline stages, in order, and the rules for resuming a failed run from one.

needs: artifacts recorded in analysis_state.artifacts that a run starting here requires.
cleanup: stored outputs deleted before rerunning this stage (or an earlier one).
params: request overrides consumed by this stage.
"""

PIPELINE_STAGES = [
    {"key": "harmonization", "state_stages": {"File_upload", "Harmonization"},
     "needs": [], "resumable": False},
    {"key": "filtering", "state_stages": {"Filtering"},
     "needs": ["harmonized_file"], "params": ["p_value_threshold"],
     "skipped_on_opentargets": True, "resumable": True},
    {"key": "cojo", "state_stages": {"Cojo"},
     "needs": ["harmonized_file", "significant_variants"],
     "skipped_on_opentargets": True, "resumable": False},
    {"key": "fine_mapping", "state_stages": {"Fine_mapping", "OpenTargets_credible_sets"},
     "needs": ["harmonized_file", "significant_variants", "cojo_results"],
     "cleanup": ["credible_sets", "analysis_results"],
     "skipped_on_opentargets": True, "resumable": False},
    {"key": "ldsc", "state_stages": {"LDSC_Analysis", "LDSC_Tissue_Analysis"},
     "needs": ["harmonized_file", "credible_sets"], "resumable": True},
]

STAGE_INDEX = {stage["key"]: index for index, stage in enumerate(PIPELINE_STAGES)}

STAGE_LABELS = {
    "harmonization": "harmonization",
    "filtering": "filtering",
    "cojo": "COJO",
    "fine_mapping": "fine-mapping",
    "ldsc": "LDSC tissue analysis",
}


def stage_key(state_stage):
    """Registry key of a stored analysis_state stage name, or None."""
    return next(
        (stage["key"] for stage in PIPELINE_STAGES if state_stage in stage["state_stages"]),
        None,
    )


def runs_stage(key, resume_from):
    """Whether a run resumed from resume_from (None for a full run) executes the stage."""
    return resume_from is None or STAGE_INDEX[key] >= STAGE_INDEX[resume_from]


def stages_from(key):
    return PIPELINE_STAGES[STAGE_INDEX[key]:]


def cleanup_targets(key):
    """Stored outputs to delete before rerunning from the stage."""
    return [target for stage in stages_from(key) for target in stage.get("cleanup", [])]


def rerun_params(key):
    """Request overrides allowed when rerunning from the stage."""
    return {param for stage in stages_from(key) for param in stage.get("params", [])}


def _filtering_cannot_pass(state):
    detail = state.get("error_detail") or {}
    return (
        detail.get("error_type") == "no_significant_variants"
        and not detail.get("variants_below_1e5")
    )


def _can_start_at(stage, state):
    artifacts = state.get("artifacts") or {}
    if not stage["resumable"] or not all(artifacts.get(need) for need in stage["needs"]):
        return False
    if stage.get("skipped_on_opentargets") and artifacts.get("credible_sets_source") == "opentargets":
        return False
    return not (stage["key"] == "filtering" and _filtering_cannot_pass(state))


def resumable_from(state):
    """Stages a failed run can be restarted from: the latest startable one up to the failure."""
    if (state or {}).get("status") != "Failed":
        return []
    failed = stage_key(state.get("stage")) or state.get("resumed_from")
    if failed not in STAGE_INDEX:
        return []
    for stage in reversed(PIPELINE_STAGES[:STAGE_INDEX[failed] + 1]):
        if _can_start_at(stage, state):
            return [stage["key"]]
    return []


def harmonized_object_key(user_id, project_id, file_name):
    return f"harmonized/{user_id}/{project_id}/{file_name}"
