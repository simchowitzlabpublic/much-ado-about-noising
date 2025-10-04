#!/usr/bin/env python
"""Upload processed datasets to HuggingFace Hub."""

from huggingface_hub import HfApi, upload_file
import os

# Repository configuration
REPO_ID = "ChaoyiPan/mip-dataset"

# Files to upload - full datasets without n_demos limitation
files_to_upload = [
    {
        "local_path": "/home/ANT.AMAZON.COM/iscoyizj/.cache/huggingface/hub/datasets--amandlek--robomimic/snapshots/74fa018461f479cd9fd15b924a16103012096203/v1.5/lift/ph/image_v15.hdf5",
        "repo_path": "robomimic/lift/ph/image.hdf5"
    },
    {
        "local_path": "/home/ANT.AMAZON.COM/iscoyizj/.cache/huggingface/hub/datasets--amandlek--robomimic/snapshots/74fa018461f479cd9fd15b924a16103012096203/v1.5/lift/ph/low_dim_v15.hdf5",
        "repo_path": "robomimic/lift/ph/low_dim.hdf5"
    }
]

def main():
    api = HfApi()

    # Create repository if it doesn't exist
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
        print(f"✓ Repository {REPO_ID} is ready")
    except Exception as e:
        print(f"Repository creation note: {e}")

    # Upload each file
    for file_info in files_to_upload:
        local_path = file_info["local_path"]
        repo_path = file_info["repo_path"]

        if not os.path.exists(local_path):
            print(f"⚠️ Skipping {repo_path}: Local file not found")
            continue

        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"Uploading {repo_path} ({file_size_mb:.1f} MB)...")

        try:
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="dataset",
            )
            print(f"✓ Uploaded {repo_path}")
        except Exception as e:
            print(f"✗ Failed to upload {repo_path}: {e}")

    print(f"\n✓ Dataset upload complete!")
    print(f"  Repository: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()