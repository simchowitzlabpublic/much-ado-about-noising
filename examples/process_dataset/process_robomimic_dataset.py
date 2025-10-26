"""Process and upload ALL robomimic datasets (state and image) with full demos to HuggingFace."""

import tyro
import loguru
from process_single_robomimic_dataset import (
    download_original_dataset,
    upload_to_hub,
    validate_dataset,
)
from huggingface_hub import HfApi

# Repository configuration
PROCESSED_REPO_ID = "ChaoyiPan/mip-dataset"

# Define all tasks and their configurations
TASKS = [
    # Task name, sources to process, has_mh (whether it has machine-generated data)
    ("lift", ["ph", "mh"], True),
    ("can", ["ph", "mh"], True),
    ("square", ["ph", "mh"], True),
    ("transport", ["ph", "mh"], True),
    ("tool_hang", ["ph"], False),  # tool_hang only has ph
]


def process_and_upload_state_dataset(task: str, source: str) -> bool:
    """Download and upload state-based (low_dim) dataset."""
    try:
        # Download low_dim dataset from original repo
        loguru.logger.info(f"Downloading low_dim dataset for {task}/{source}...")
        local_path = download_original_dataset(
            task=task, source=source, dataset_type="low_dim"
        )

        # Validate dataset
        info = validate_dataset(local_path)
        loguru.logger.info(f"State dataset validated: {info['n_demos']} demos")

        # Upload to our repo
        loguru.logger.info(f"Uploading state dataset to {PROCESSED_REPO_ID}...")
        repo_path = upload_to_hub(
            local_path=local_path,
            task=task,
            source=source,
            dataset_type="low_dim",
            repo_id=PROCESSED_REPO_ID,
        )
        return True

    except Exception as e:
        loguru.logger.error(f"Failed to process state dataset {task}/{source}: {e}")
        return False


def process_and_upload_image_dataset(task: str, source: str) -> bool:
    """Process and upload image dataset with ALL demos."""
    try:
        # Use the get_or_create function which has the fixed processing
        from process_single_robomimic_dataset import get_or_create_image_dataset

        loguru.logger.info(f"Processing image dataset for {task}/{source}...")
        image_path = get_or_create_image_dataset(
            task=task,
            source=source,
            n_demos=-1,  # Process ALL demos
            force_recreate=True,  # Force to ensure we get the fixed version
        )

        # Validate processed dataset
        info = validate_dataset(image_path)
        loguru.logger.info(f"Image dataset created: {info['n_demos']} demos")
        print(f"  Image keys: {info.get('image_keys', [])}")

        # The get_or_create function already handles caching and uploading
        loguru.logger.info("Dataset processed, cached, and uploaded successfully")
        return True

    except Exception as e:
        loguru.logger.error(f"Failed to process image dataset {task}/{source}: {e}")
        return False


def main(
    tasks: list[str] = ["lift", "can", "square", "transport", "tool_hang"],
    sources: list[str] = ["ph", "mh"],
    modalities: list[str] = ["state", "image"],
):
    """Process and upload all datasets."""
    loguru.logger.info("Processing and uploading all datasets.")

    api = HfApi()
    api.create_repo(repo_id=PROCESSED_REPO_ID, repo_type="dataset", exist_ok=True)
    loguru.logger.info(f"Repository ready: {PROCESSED_REPO_ID}")

    # Track results
    results = {
        "state_success": [],
        "state_failed": [],
        "image_success": [],
        "image_failed": [],
    }

    # Process all tasks
    for task in tasks:
        for source in sources:
            for modality in modalities:
                if modality == "state":
                    if process_and_upload_state_dataset(task, source):
                        results["state_success"].append(f"{task}/{source}")
                    else:
                        results["state_failed"].append(f"{task}/{source}")

                elif modality == "image":
                    if process_and_upload_image_dataset(task, source):
                        results["image_success"].append(f"{task}/{source}")
                    else:
                        results["image_failed"].append(f"{task}/{source}")

    # Print summary
    loguru.logger.info("Processing summary:")
    loguru.logger.info(
        f"State datasets (low_dim): {len(results['state_success'])} successful, {len(results['state_failed'])} failed"
    )
    for item in results["state_success"]:
        loguru.logger.info(f"    - {item}")
    for item in results["state_failed"]:
        loguru.logger.info(f"    - {item}")

    loguru.logger.info(
        f"Image datasets: {len(results['image_success'])} successful, {len(results['image_failed'])} failed"
    )
    for item in results["image_success"]:
        loguru.logger.info(f"    - {item}")
    for item in results["image_failed"]:
        loguru.logger.info(f"    - {item}")

    total_success = len(results["state_success"]) + len(results["image_success"])
    total_failed = len(results["state_failed"]) + len(results["image_failed"])

    loguru.logger.info(f"Total: {total_success} successful, {total_failed} failed")
    loguru.logger.info(
        f"View datasets at: https://huggingface.co/datasets/{PROCESSED_REPO_ID}"
    )


if __name__ == "__main__":
    tyro.cli(main)
