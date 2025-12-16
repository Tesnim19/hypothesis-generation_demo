from config import create_dependencies, Config
from loguru import logger

def dask_setup(worker):
    if hasattr(worker, "deps"):
        logger.info("worker deps already been set up")
        return

    # every worker needs to intialize db instances since we cant serialize them
    logger.info("Worker starting, setting up dependencies...")
    logger.info("[DASK PRELOAD] Worker starting; initializing dependencies")
    config = Config.from_env()
    deps = create_dependencies(config)

    worker.deps = deps
    logger.info("Worker dependencies set up.")