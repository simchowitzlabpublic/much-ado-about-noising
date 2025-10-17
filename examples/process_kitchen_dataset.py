"""Download Kitchen data and upload to HuggingFace.

This script downloads the Kitchen dataset from the Diffusion Policy repository
and uploads it to HuggingFace for easy access.

Author: Chaoyi Pan
Date: 2025-10-17
"""

import zipfile
from pathlib import Path

import requests
from huggingface_hub import HfApi
from loguru import logger
from tqdm import tqdm


def download_kitchen_dataset(output_dir: str = "data/kitchen"):
    """Download Kitchen dataset from Diffusion Policy repository.

    Args:
        output_dir: Directory to save the downloaded dataset

    Returns:
        Path to the downloaded dataset directory
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # URL for Kitchen dataset
    url = "https://diffusion-policy.cs.columbia.edu/data/training/kitchen.zip"

    logger.info(f"Downloading Kitchen dataset from {url}")

    # Download with progress bar
    response = requests.get(url, stream=True, timeout=300)
    total_size = int(response.headers.get("content-length", 0))

    zip_path = output_path / "kitchen.zip"

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

    # Find the extracted dataset directory
    # The kitchen dataset should contain observations_seq.npy, actions_seq.npy, existence_mask.npy
    dataset_dirs = []
    for item in output_path.iterdir():
        if item.is_dir():
            # Check if this directory contains the required files
            has_obs = (item / "observations_seq.npy").exists()
            has_act = (item / "actions_seq.npy").exists()
            has_mask = (item / "existence_mask.npy").exists()
            if has_obs and has_act and has_mask:
                dataset_dirs.append(item)

    if not dataset_dirs:
        raise FileNotFoundError(
            "No valid kitchen dataset directory found with required .npy files"
        )

    dataset_path = dataset_dirs[0]
    logger.info(f"Dataset extracted to {dataset_path}")

    return str(dataset_path)


def upload_to_huggingface(
    dataset_path: str,
    repo_id: str = "ChaoyiPan/mip-dataset",
    path_in_repo: str = "kitchen/kitchen_demos_multitask.zip",
):
    """Upload Kitchen dataset to HuggingFace as a zip file.

    Args:
        dataset_path: Local path to the dataset directory
        repo_id: HuggingFace repository ID
        path_in_repo: Path within the repository to upload to (should end with .zip)
    """
    logger.info(f"Creating zip file from {dataset_path}")

    # Create a zip file from the dataset directory
    dataset_path_obj = Path(dataset_path)
    zip_path = dataset_path_obj.parent / f"{dataset_path_obj.name}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in tqdm(list(dataset_path_obj.rglob("*")), desc="Compressing"):
            if file_path.is_file():
                # Store relative path within the dataset directory
                arcname = file_path.relative_to(dataset_path_obj.parent)
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
    """Main function to download and upload Kitchen dataset."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and upload Kitchen dataset to HuggingFace"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/kitchen",
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
        default="kitchen/kitchen_demos_multitask.zip",
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
        "--dataset_path",
        type=str,
        default=None,
        help="Path to existing dataset directory (if skip_download is True)",
    )

    args = parser.parse_args()

    # Download dataset
    if args.skip_download:
        if args.dataset_path is None:
            raise ValueError("Must provide --dataset_path when using --skip_download")
        dataset_path = args.dataset_path
        logger.info(f"Using existing dataset at {dataset_path}")
    else:
        dataset_path = download_kitchen_dataset(args.output_dir)

    # Upload to HuggingFace
    if not args.skip_upload:
        upload_to_huggingface(dataset_path, args.repo_id, args.path_in_repo)
    else:
        logger.info("Skipping upload step")

    logger.info("Done!")


if __name__ == "__main__":
    main()
