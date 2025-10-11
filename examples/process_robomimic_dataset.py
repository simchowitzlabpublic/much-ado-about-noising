"""Process and upload ALL robomimic datasets (state and image) with full demos to HuggingFace."""

from pathlib import Path

from huggingface_hub import HfApi
from process_single_robomimic_dataset import (
    download_original_dataset,
    process_to_image_dataset,
    upload_to_hub,
    validate_dataset,
)

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
    print(f"\n{'=' * 80}")
    print(f"Processing STATE dataset: {task}/{source}")
    print(f"{'=' * 80}")

    try:
        # Download low_dim dataset from original repo
        print(f"Downloading low_dim dataset for {task}/{source}...")
        local_path = download_original_dataset(
            task=task, source=source, dataset_type="low_dim"
        )

        # Validate dataset
        info = validate_dataset(local_path)
        print(f"✓ State dataset validated: {info['n_demos']} demos")

        # Upload to our repo
        print(f"Uploading state dataset to {PROCESSED_REPO_ID}...")
        repo_path = upload_to_hub(
            local_path=local_path,
            task=task,
            source=source,
            dataset_type="low_dim",
            repo_id=PROCESSED_REPO_ID,
        )
        print(f"✓ Uploaded to: {repo_path}")
        return True

    except Exception as e:
        print(f"✗ Failed to process state dataset {task}/{source}: {e}")
        return False


def process_and_upload_image_dataset(task: str, source: str) -> bool:
    """Process and upload image dataset with ALL demos."""
    print(f"\n{'=' * 80}")
    print(f"Processing IMAGE dataset: {task}/{source} (ALL DEMOS)")
    print(f"{'=' * 80}")

    try:
        # Use the get_or_create function which has the fixed processing
        from process_single_robomimic_dataset import get_or_create_image_dataset

        print(f"Processing image dataset for {task}/{source}...")
        image_path = get_or_create_image_dataset(
            task=task,
            source=source,
            n_demos=-1,  # Process ALL demos
            force_recreate=True  # Force to ensure we get the fixed version
        )

        # Validate processed dataset
        info = validate_dataset(image_path)
        print(f"✓ Image dataset created: {info['n_demos']} demos")
        print(f"  Image keys: {info.get('image_keys', [])}")

        # The get_or_create function already handles caching and uploading
        print(f"✓ Dataset processed, cached, and uploaded successfully")
        return True

    except Exception as e:
        print(f"✗ Failed to process image dataset {task}/{source}: {e}")
        return False


def main():
    """Process and upload all datasets."""
    print("=" * 80)
    print("PROCESSING AND UPLOADING ALL ROBOMIMIC DATASETS")
    print("Dataset Types: STATE (low_dim) and IMAGE")
    print("Demo Count: FULL DATASETS (no limits)")
    print(f"Repository: {PROCESSED_REPO_ID}")
    print("=" * 80)

    # Check authentication
    api = HfApi()
    try:
        whoami = api.whoami()
        print(f"✓ Authenticated as: {whoami['name']}")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        print("Please run: huggingface-cli login")
        return

    # Create or verify repo exists
    try:
        api.create_repo(repo_id=PROCESSED_REPO_ID, repo_type="dataset", exist_ok=True)
        print(f"✓ Repository ready: {PROCESSED_REPO_ID}")
    except Exception as e:
        print(f"⚠️ Repository check: {e}")

    # Track results
    results = {
        "state_success": [],
        "state_failed": [],
        "image_success": [],
        "image_failed": [],
    }

    # Process all tasks
    for task, sources, _ in TASKS:
        for source in sources:
            # Process state dataset
            if process_and_upload_state_dataset(task, source):
                results["state_success"].append(f"{task}/{source}")
            else:
                results["state_failed"].append(f"{task}/{source}")

            # Process image dataset
            if process_and_upload_image_dataset(task, source):
                results["image_success"].append(f"{task}/{source}")
            else:
                results["image_failed"].append(f"{task}/{source}")

    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)

    print("\n📊 STATE DATASETS (low_dim):")
    print(f"  ✓ Successful: {len(results['state_success'])}")
    for item in results["state_success"]:
        print(f"    - {item}")
    print(f"  ✗ Failed: {len(results['state_failed'])}")
    for item in results["state_failed"]:
        print(f"    - {item}")

    print("\n🖼️ IMAGE DATASETS:")
    print(f"  ✓ Successful: {len(results['image_success'])}")
    for item in results["image_success"]:
        print(f"    - {item}")
    print(f"  ✗ Failed: {len(results['image_failed'])}")
    for item in results["image_failed"]:
        print(f"    - {item}")

    total_success = len(results["state_success"]) + len(results["image_success"])
    total_failed = len(results["state_failed"]) + len(results["image_failed"])

    print("\n" + "=" * 80)
    print(f"TOTAL: {total_success} successful, {total_failed} failed")
    print(f"🌐 View datasets at: https://huggingface.co/datasets/{PROCESSED_REPO_ID}")
    print("=" * 80)


if __name__ == "__main__":
    main()
