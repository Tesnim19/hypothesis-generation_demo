from prefect.deployments import run_deployment
from flows import enrichment_flow

def invoke_enrichment_deployment(current_user_id, phenotype, variant, hypothesis_id, project_id, deps=None):
    """
    Invoke enrichment - run locally if dependencies available, otherwise use deployment.
    """
    if deps is not None:
        # Run locally with passed dependencies (preferred method)
        try:
            result = enrichment_flow(
                current_user_id=current_user_id,
                phenotype=phenotype,
                variant=variant,
                hypothesis_id=hypothesis_id,
                project_id=project_id,
                deps=deps
            )
            return result
        except Exception as e:
            # If local execution fails, fall back to deployment
            print(f"Local enrichment failed: {e}, falling back to deployment")
    
    # Fall back to deployment (for backward compatibility)
    run_deployment(
        name="enrichment-flow/enrichment-flow-deployment",
        parameters={
            "current_user_id": current_user_id, 
            "phenotype": phenotype, 
            "variant": variant,
            "hypothesis_id": hypothesis_id,
            "project_id": project_id
        },
        timeout=0
    )