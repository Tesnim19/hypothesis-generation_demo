import os
import requests as _http


def _download_to_path_sync(url: str, path: str) -> int:
    """Run in thread pool: stream download from url to path. Returns file size."""
    with _http.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return os.path.getsize(path)
