import os
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

dataset = load_dataset("nielsr/CelebA-faces")
print("dataset loaded")

dir = Path(".") / "data" / "celeba" / "img_align_celeba"

if not dir.exists():
    os.makedirs(dir)
    # Iterate over the dataset and save each image
    for idx, item in tqdm(enumerate(dataset["train"]), total=len(dataset["train"]), desc="Saving individual images..."):
        # The image is already a JpegImageFile, so we can directly save it
        item["image"].save(Path(".") / "data" / "celeba" / "img_align_celeba" / f"{idx:06}.jpg")

    print("All images have been saved.")
