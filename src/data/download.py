"""Download MovieLens dataset and extract raw CSV files."""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import requests

from src.common.config import load_params


def download_raw(url: str, raw_dir: str | Path) -> None:
    """Download MovieLens zip and extract ratings.csv and movies.csv directly to raw_dir."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    ratings_file = raw_path / "ratings.csv"
    movies_file = raw_path / "movies.csv"

    if ratings_file.exists() and movies_file.exists():
        print(f"Raw data files already exist in {raw_path}. Skipping download.")
        return

    print(f"Downloading dataset from {url}...")
    default_cert = "/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").exists() else True
    verify_cert = os.getenv("SSL_CERT_FILE", default_cert)
    response = requests.get(url, timeout=60, verify=verify_cert)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        for member in zf.namelist():
            filename = Path(member).name
            if filename in ("ratings.csv", "movies.csv"):
                dest = raw_path / filename
                with zf.open(member) as source, open(dest, "wb") as target:
                    target.write(source.read())
                print(f"Extracted {filename} to {dest}")

    if not ratings_file.exists() or not movies_file.exists():
        raise FileNotFoundError(
            f"Extraction failed: ratings.csv or movies.csv missing in {raw_path}"
        )


def main() -> None:
    """Main entrypoint for dataset download stage."""
    params = load_params()
    data_params = params.get("data", {})
    url = data_params["url"]
    raw_dir = data_params["raw_dir"]
    download_raw(url, raw_dir)


if __name__ == "__main__":
    main()
