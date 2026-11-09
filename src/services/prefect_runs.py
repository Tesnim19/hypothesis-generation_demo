import os

import requests
from loguru import logger


def cancel_flow_run(flow_run_id):
    """Ask Prefect to cancel a flow run so its worker tears it down; best effort, never raises."""
    prefect_url = os.getenv("PREFECT_API_URL", "http://prefect-service:4200/api")
    try:
        resp = requests.post(
            f"{prefect_url}/flow_runs/{flow_run_id}/set_state",
            json={"state": {"type": "CANCELLING"}},
            timeout=3,
        )
        if resp.status_code in (200, 201):
            logger.info(f"Requested cancellation of Prefect flow run {flow_run_id}")
        else:
            logger.warning(f"Could not cancel flow run {flow_run_id}: {resp.status_code}")
    except Exception as cancel_e:
        logger.warning(f"Failed to cancel Prefect flow run {flow_run_id}: {cancel_e}")
