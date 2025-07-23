"""Utility functions for saving, loading and viewing files in S3.

These helpers are designed for use in Jupyter notebooks running on
Amazon SageMaker. They provide small wrappers for uploading artefacts,
removing temporary directories and creating presigned links to view the
results in a browser.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import boto3
import shutil
import traceback

__all__ = [
    "push_and_purge",
    "pull_load_purge",
    "generate_presigned_url",
    "display_html_link",
]


def push_and_purge(
    out_dir: str | Path,
    file_name: str,
    bucket: str,
    team_prefix: str,
    user_space: str,
    file_structure: str,
) -> None:
    """Upload a file to S3 and remove the temporary directory.

    Parameters
    ----------
    out_dir : Path or str
        Directory containing ``file_name``.
    file_name : str
        Name of the file to upload.
    bucket : str
        S3 bucket name.
    team_prefix : str
        Common prefix used by the team, e.g. ``"sagemaker_notebooks/"``.
    user_space : str
        User specific folder inside ``team_prefix``.
    file_structure : str
        Folder structure under ``user_space`` where the file will be stored.
    """
    out_dir = Path(out_dir)
    local = out_dir / file_name

    prefix = f"{team_prefix.rstrip('/')}/{user_space.strip('/')}/{file_structure.strip('/')}/"
    s3_key = f"{prefix}{file_name}"
    s3 = boto3.client("s3")

    try:
        s3.upload_file(str(local), bucket, s3_key)
        print(f"Uploaded {local} to s3://{bucket}/{s3_key}")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
        print(f"Removed temporary directory {out_dir}")


def generate_presigned_url(
    bucket: str,
    key: str,
    expires_in: int = 3600,
    s3_client: Optional[boto3.client] = None,
) -> str:
    """Create a presigned URL for an object in S3."""
    s3 = s3_client or boto3.client("s3")
    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )
    return url


def display_html_link(bucket: str, key: str, expires_in: int = 3600):
    """Return a clickable HTML link for an object stored in S3."""
    from IPython.display import HTML

    url = generate_presigned_url(bucket=bucket, key=key, expires_in=expires_in)
    return HTML(f'<a href="{url}" target="_blank">Open report in new tab</a>')


def pull_load_purge(
    s3_uri: str,
    loader_fn: Callable[[str | Path], Any],
    tmp_dir: str | Path = "input_tmp",
) -> Any:
    """Download ``s3_uri`` and load it with ``loader_fn``.

    The file is downloaded to ``tmp_dir`` and removed afterwards. The
    returned value is whatever ``loader_fn`` returns.
    """
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got {s3_uri!r}")

    bucket, key = parsed.netloc, parsed.path.lstrip("/")

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(exist_ok=True)
    local = tmp_dir / Path(key).name

    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket, key, str(local))
        print(f"Downloaded {s3_uri} to {local}")
        obj = loader_fn(local)
        print("Object loaded.")
        return obj
    except Exception:
        traceback.print_exc()
        raise
    finally:
        try:
            local.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except OSError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Removed temporary directory {tmp_dir}")

