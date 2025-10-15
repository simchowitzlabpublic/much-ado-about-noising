# Instruction for porting a new task

## 1. Download the dataset and prepare dataset loader

We use huggingface to manage the dataset. To define a dataloader, it is recommended to first upload your dataset to huggingface dataset repo: `ChaoyiPan/mip-dataset`, where you can refer to [process_robomimic_dataset.py](../examples/process_single_robomimic_dataset.py) for how to upload your dataset to huggingface. Consider creating a dataset uploader in `examples/` for convenience. (e.g. for pusht, you can first download the dataset from `lerobot/pusht` and then upload)

Then define a new dataset loader under folder `mip/datasets/`, required function are: `make_dataset`, For dataset class, make sure `__get_item__` is implemented. (for dataset usage, you got to refer to the original dataset repo, e.g. for `lerobot/pusht`, you can refer to )
