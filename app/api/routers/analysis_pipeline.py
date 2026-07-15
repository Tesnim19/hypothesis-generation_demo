from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from werkzeug.utils import secure_filename

from app.api.dependencies import get_current_user_id, get_deps
from app.core.utils import (
    allowed_file,
    compute_file_md5,
    get_shared_temp_dir,
)
from app.workers.tasks.project import count_gwas_records
from app.workers.workflows.run_deployment import invoke_analysis_pipeline_deployment
from .utils import _download_to_path_sync

router = APIRouter()


@router.post("/analysis-pipeline", status_code=202)
async def post_analysis_pipeline(
    request: Request,
    current_user_id: str = Depends(get_current_user_id),
    gwas_file: UploadFile | None = File(None),
):
    try:
        form = await request.form()

        project_name: str | None = form.get("project_name")
        phenotype: str | None = form.get("phenotype")
        ref_genome: str = form.get("ref_genome", "GRCh38")
        population: str = form.get("population", "EUR")
        max_workers: int = int(form.get("max_workers", 3))
        is_uploaded: bool = form.get("is_uploaded", "false").lower() == "true"

        maf_threshold: float = float(form.get("maf_threshold", 0.01))
        seed: int = int(form.get("seed", 42))
        window: int = int(form.get("window", 2000))
        L: int = int(form.get("L", -1))
        coverage: float = float(form.get("coverage", 0.95))
        min_abs_corr: float = float(form.get("min_abs_corr", 0.5))
        batch_size: int = int(form.get("batch_size", 5))
        sample_size: int = int(form.get("sample_size", 10000))

        projects = get_deps()["projects"]
        files = get_deps()["files"]
        analysis = get_deps()["analysis"]
        gene_expression = get_deps().get("gene_expression")
        config = get_deps()["config"]
        storage = get_deps().get("storage")
        gwas_library = get_deps().get("gwas_library")

        if not project_name:
            raise HTTPException(status_code=400, detail="project_name is required")
        if not phenotype:
            raise HTTPException(status_code=400, detail="phenotype is required")

        if is_uploaded and gwas_file and not allowed_file(gwas_file.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Supported: .tsv, .txt, .csv, .gz, .bgz",
            )
        if ref_genome not in ("GRCh37", "GRCh38"):
            raise HTTPException(
                status_code=400, detail="Reference genome must be GRCh37 or GRCh38"
            )
        if population not in ("EUR", "AFR", "AMR", "EAS", "SAS"):
            raise HTTPException(
                status_code=400,
                detail="Population must be one of: EUR, AFR, AMR, EAS, SAS",
            )
        if not (1 <= max_workers <= 16):
            raise HTTPException(
                status_code=400, detail="Max workers must be between 1-16"
            )
        if not (0.001 <= maf_threshold <= 0.5):
            raise HTTPException(
                status_code=400, detail="MAF threshold must be between 0.001-0.5"
            )
        if not (1 <= seed <= 999999):
            raise HTTPException(
                status_code=400, detail="Seed must be between 1-999999"
            )
        if window > 10000:
            raise HTTPException(
                status_code=400,
                detail="Fine-mapping window shouldn't be greater than 10000 kb",
            )
        if L != -1 and not (1 <= L <= 50):
            raise HTTPException(
                status_code=400, detail="L must be -1 (auto) or between 1-50"
            )
        if not (0.5 <= coverage <= 0.999):
            raise HTTPException(
                status_code=400, detail="Coverage must be between 0.5-0.999"
            )
        if not (0.5 <= min_abs_corr <= 1.0):
            raise HTTPException(
                status_code=400,
                detail="Min absolute correlation must be between 0.5-1.0",
            )
        if not (1 <= batch_size <= 20):
            raise HTTPException(
                status_code=400, detail="Batch size must be between 1-20"
            )

        start_time = datetime.now()
        file_metadata_id = None
        file_path: str | None = None
        filename: str | None = None
        file_size: int = 0
        gwas_records_count: int = 0
        original_filename: str | None = None

        if not is_uploaded:
            file_id_param: str | None = form.get("gwas_file")
            if not file_id_param:
                raise HTTPException(
                    status_code=400,
                    detail="gwas_file parameter is required when is_uploaded=false",
                )

            logger.info(f"[API] Auto-detecting source for file ID: {file_id_param}")

            file_meta = None
            try:
                file_meta = files.get_file_metadata(current_user_id, file_id_param)
            except Exception as exc:
                logger.info(f"[API] Not a valid user file ID, checking library: {exc}")

            if file_meta:
                storage_key = file_meta.get("storage_key")
                filename = file_meta.get("filename")
                file_size = file_meta.get("file_size", 0)
                gwas_records_count = file_meta.get("record_count", 0)

                if not storage_key:
                    raise HTTPException(
                        status_code=400,
                        detail="File does not have storage information.",
                    )
                if not storage:
                    raise HTTPException(
                        status_code=500, detail="Storage service not available"
                    )

                temp_dir = get_shared_temp_dir(
                    user_id=current_user_id, prefix="user_file"
                )
                temp_file_path = os.path.join(temp_dir, filename)
                if not storage.download_file(storage_key, temp_file_path):
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to retrieve file from storage",
                    )

                file_path = temp_file_path
                file_metadata_id = file_id_param
                original_filename = filename
            else:
                gwas_entry = gwas_library.get_gwas_entry(file_id=file_id_param) if gwas_library else None
                if not gwas_entry:
                    raise HTTPException(
                        status_code=404,
                        detail=f"File not found in user library or system library: {file_id_param}",
                    )

                filename = gwas_entry.get("filename", file_id_param)
                original_filename = filename
                minio_path_lib = f"gwas_cache/{filename}"

                if storage and gwas_entry.get("downloaded") and gwas_entry.get("minio_path"):
                    minio_path = gwas_entry["minio_path"]
                    if storage.exists(minio_path):
                        temp_dir = get_shared_temp_dir(
                            user_id=current_user_id, prefix="gwas_cache"
                        )
                        candidate = os.path.join(temp_dir, filename)
                        if storage.download_file(minio_path, candidate):
                            file_path = candidate
                            file_size = os.path.getsize(file_path)
                    else:
                        gwas_library.update_gwas_entry(
                            file_id_param, {"downloaded": False, "minio_path": None}
                        )

                if not file_path:
                    download_url = (
                        gwas_entry.get("aws_url")
                        or (
                            re.search(r"(https?://[^\s]+)", gwas_entry["wget_command"]).group(1)
                            if gwas_entry.get("wget_command")
                            else None
                        )
                        or gwas_entry.get("dropbox_url")
                    )

                    if not download_url:
                        raise HTTPException(
                            status_code=404,
                            detail=f"No download URL available for {file_id_param}",
                        )

                    temp_dir = get_shared_temp_dir(
                        user_id=current_user_id, prefix="gwas_download"
                    )
                    dl_path = os.path.join(temp_dir, filename)
                    loop = asyncio.get_running_loop()
                    file_size = await loop.run_in_executor(
                        None, _download_to_path_sync, download_url, dl_path
                    )
                    file_path = dl_path

                    if storage:
                        if storage.upload_file(file_path, minio_path_lib):
                            gwas_library.mark_as_downloaded(
                                file_id_param, minio_path_lib, file_size
                            )

                if not file_path:
                    raw_data_path = os.path.join(config.data_dir, "raw")
                    for ext in (".tsv", ".tsv.gz", ".tsv.bgz", ".txt", ".txt.gz", ".csv", ".csv.gz"):
                        candidate = os.path.join(raw_data_path, f"{file_id_param}{ext}")
                        if os.path.exists(candidate):
                            file_path = candidate
                            filename = f"{file_id_param}{ext}"
                            file_size = os.path.getsize(file_path)
                            break

                if not file_path:
                    raise HTTPException(
                        status_code=404,
                        detail=f"GWAS file not found: {file_id_param}",
                    )

                gwas_records_count = count_gwas_records(file_path)

        else:
            # Uploaded file
            if not gwas_file or gwas_file.filename == "":
                raise HTTPException(status_code=400, detail="No GWAS file uploaded")
            if not allowed_file(gwas_file.filename):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file format. Supported: .tsv, .txt, .csv, .gz, .bgz",
                )

            original_filename = gwas_file.filename
            filename = secure_filename(gwas_file.filename)
            file_id_new = str(uuid4())
            temp_dir = get_shared_temp_dir(user_id=current_user_id, prefix="upload")
            temp_file_path = os.path.join(temp_dir, filename)

            with open(temp_file_path, "wb") as fh:
                while chunk := await gwas_file.read(1024 * 1024):
                    fh.write(chunk)

            file_size = os.path.getsize(temp_file_path)
            md5_hash = compute_file_md5(temp_file_path)

            existing_file = None
            if md5_hash and storage:
                existing_file = files.find_file_by_md5(current_user_id, md5_hash)

            if existing_file:
                storage_key = existing_file.get("storage_key")
                if storage_key and storage.exists(storage_key):
                    storage.download_file(storage_key, temp_file_path)
                    file_path = temp_file_path
                    file_metadata_id = existing_file.get("_id")
                    gwas_records_count = existing_file.get(
                        "record_count", count_gwas_records(file_path)
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to retrieve existing file from storage",
                    )
            else:
                gwas_records_count = count_gwas_records(temp_file_path)

                if storage:
                    object_key = f"uploads/{current_user_id}/{file_id_new}/{filename}"
                    if not storage.upload_file(temp_file_path, object_key):
                        raise HTTPException(
                            status_code=500, detail="File upload failed"
                        )
                    file_path = temp_file_path
                    md5_hash_param: str | None = md5_hash
                else:
                    user_upload_dir = os.path.join(
                        "data", "uploads", str(current_user_id)
                    )
                    os.makedirs(user_upload_dir, exist_ok=True)
                    file_path = os.path.join(
                        user_upload_dir, f"{file_id_new}_{filename}"
                    )
                    with open(file_path, "wb") as fh:
                        fh.write(content)
                    file_size = os.path.getsize(file_path)
                    gwas_records_count = count_gwas_records(file_path)
                    object_key = None
                    md5_hash_param = None

        if not file_metadata_id:
            source = "gwas_library" if not is_uploaded else "user_upload"
            storage_key_arg = object_key if is_uploaded and storage else None
            md5_arg = md5_hash_param if is_uploaded else None

            file_metadata_id = files.create_file_metadata(
                user_id=current_user_id,
                filename=filename,
                original_filename=original_filename,
                file_path=file_path,
                file_type="gwas",
                file_size=file_size,
                record_count=gwas_records_count,
                download_url=f"/download/{str(uuid4())}",
                md5_hash=md5_arg,
                storage_key=storage_key_arg,
                source=source,
            )

        analysis_parameters = {
            "maf_threshold": maf_threshold,
            "seed": seed,
            "window": window,
            "L": L,
            "coverage": coverage,
            "min_abs_corr": min_abs_corr,
            "batch_size": batch_size,
            "max_workers": max_workers,
        }

        project_id = projects.create_project(
            user_id=current_user_id,
            name=project_name,
            gwas_file_id=file_metadata_id,
            phenotype=phenotype,
            population=population,
            ref_genome=ref_genome,
            analysis_parameters=analysis_parameters,
        )

        metadata_dir = os.path.join("data", "metadata", str(current_user_id))
        os.makedirs(metadata_dir, exist_ok=True)
        metadata = {
            "file_id": file_metadata_id,
            "user_id": current_user_id,
            "filename": filename,
            "original_filename": original_filename,
            "file_path": file_path,
            "file_type": "gwas",
            "upload_date": str(datetime.now()),
            "file_size": file_size,
            "project_id": project_id,
        }
        with open(os.path.join(metadata_dir, f"{file_metadata_id}.json"), "w") as fh:
            json.dump(metadata, fh)

        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"[API] Completed: {filename} in {total_time:.1f}s")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: invoke_analysis_pipeline_deployment(
                user_id=current_user_id,
                project_id=project_id,
                gwas_file_path=file_path,
                ref_genome=ref_genome,
                population=population,
                batch_size=batch_size,
                max_workers=max_workers,
                maf_threshold=maf_threshold,
                seed=seed,
                window=window,
                L=L,
                coverage=coverage,
                min_abs_corr=min_abs_corr,
                sample_size=sample_size,
            ),
        )

        return {
            "status": "started",
            "project_id": project_id,
            "file_id": file_metadata_id,
            "message": "Analysis pipeline started successfully",
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[API] Error starting analysis pipeline: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Error starting analysis pipeline: {exc}",
        )
