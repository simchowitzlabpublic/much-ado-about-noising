"""Download PushT data and upload to HuggingFace.

This script downloads the PushT dataset from the official Diffusion Policy repository
and uploads it to HuggingFace for easy access.

Author: Chaoyi Pan
Date: 2025-10-15
"""

import zipfile
from pathlib import Path

import requests
from huggingface_hub import HfApi
from loguru import logger
from tqdm import tqdm


def download_pusht_dataset(output_dir: str = "data/pusht"):
    """Download PushT dataset from Diffusion Policy repository.

    Args:
        output_dir: Directory to save the downloaded dataset

    Returns:
        Path to the downloaded zarr file
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # URL for PushT dataset
    url = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"

    logger.info(f"Downloading PushT dataset from {url}")

    # Download with progress bar
    response = requests.get(url, stream=True, timeout=300)
    total_size = int(response.headers.get("content-length", 0))

    zip_path = output_path / "pusht.zip"

    with (
        open(zip_path, "wb") as f,
        tqdm(
            desc="Downloading",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info(f"Downloaded to {zip_path}")

    # Extract the zip file
    logger.info("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_path)

    # Remove the zip file to save space
    zip_path.unlink()

    # Find the zarr file
    zarr_files = list(output_path.rglob("*.zarr"))
    if not zarr_files:
        raise FileNotFoundError("No .zarr file found in extracted dataset")

    zarr_path = zarr_files[0]
    logger.info(f"Dataset extracted to {zarr_path}")

    return str(zarr_path)


def upload_to_huggingface(
    zarr_path: str,
    repo_id: str = "ChaoyiPan/mip-dataset",
    path_in_repo: str = "pusht/pusht_cchi_v7_replay.zarr.zip",
):
    """Upload PushT dataset to HuggingFace as a zip file.

    Args:
        zarr_path: Local path to the zarr dataset
        repo_id: HuggingFace repository ID
        path_in_repo: Path within the repository to upload to (should end with .zip)
    """
    logger.info(f"Creating zip file from {zarr_path}")

    # Create a zip file from the zarr directory
    zarr_path_obj = Path(zarr_path)
    zip_path = zarr_path_obj.parent / f"{zarr_path_obj.name}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in tqdm(list(zarr_path_obj.rglob("*")), desc="Compressing"):
            if file_path.is_file():
                # Store relative path within the zarr directory
                arcname = file_path.relative_to(zarr_path_obj.parent)
                zipf.write(file_path, arcname)

    logger.info(f"Created zip file at {zip_path}")
    logger.info(f"Uploading dataset to {repo_id}/{path_in_repo}")

    api = HfApi()

    # Upload the zip file
    api.upload_file(
        path_or_fileobj=str(zip_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )

    logger.info(f"Successfully uploaded to {repo_id}/{path_in_repo}")

    # Clean up the zip file
    zip_path.unlink()
    logger.info("Cleaned up temporary zip file")


def main():
    """Main function to download and upload PushT dataset."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and upload PushT dataset to HuggingFace"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/pusht",
        help="Directory to save downloaded dataset",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="ChaoyiPan/mip-dataset",
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--path_in_repo",
        type=str,
        default="pusht/pusht_cchi_v7_replay.zarr.zip",
        help="Path within the repository (should end with .zip)",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download step (use existing local dataset)",
    )
    parser.add_argument(
        "--skip_upload",
        action="store_true",
        help="Skip upload step (only download)",
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default=None,
        help="Path to existing zarr file (if skip_download is True)",
    )

    args = parser.parse_args()

    # Download dataset
    if args.skip_download:
        if args.zarr_path is None:
            raise ValueError("Must provide --zarr_path when using --skip_download")
        zarr_path = args.zarr_path
        logger.info(f"Using existing dataset at {zarr_path}")
    else:
        zarr_path = download_pusht_dataset(args.output_dir)

    # Upload to HuggingFace
    if not args.skip_upload:
        upload_to_huggingface(zarr_path, args.repo_id, args.path_in_repo)
    else:
        logger.info("Skipping upload step")

    logger.info("Done!")


if __name__ == "__main__":
    main()
