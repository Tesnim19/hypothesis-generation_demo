from fastapi import APIRouter

from . import (
    internal,
    enrich,
    hypothesis,
    chat,
    projects,
    analysis_pipeline,
    phenotypes,
    credible_sets,
    gwas_files,
    user_files,
)

router = APIRouter()
router.include_router(internal.router)
router.include_router(enrich.router)
router.include_router(hypothesis.router)
router.include_router(chat.router)
router.include_router(projects.router)
router.include_router(analysis_pipeline.router)
router.include_router(phenotypes.router)
router.include_router(credible_sets.router)
router.include_router(gwas_files.router)
router.include_router(user_files.router)
