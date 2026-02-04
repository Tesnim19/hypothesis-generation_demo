from config import create_dependencies, Config
from loguru import logger

def dask_setup(worker):
    if hasattr(worker, "deps"):
        logger.info("worker deps already been set up")
        return

    # every worker needs to intialize db instances since we cant serialize them
    logger.info("Worker starting, setting up dependencies...")
    logger.info("[DASK PRELOAD] Worker starting; initializing dependencies")
    try:
        config = Config.from_env()
        deps = create_dependencies(config)
    except Exception as e:
        worker.deps = None
        worker.deps_error = str(e)
        logger.exception("Failed to initialize worker dependencies")
        raise  
    else:
        worker.deps = deps
        logger.info("Worker dependencies set up.") 