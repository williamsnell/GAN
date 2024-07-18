import wandb
import pdb
import sys
from typing import Dict, List
from collections import defaultdict

def get_available_images(project: str, run: str) -> Dict[str, List[str]]:
    images_available = defaultdict(list)
    api = wandb.Api()
    event = api.run(f"{project}/{run}")

    for file in event.files():
        if file.name.endswith(".png"):
            number = file.name.replace("media/images/images_", "").split("_")[0]

            images_available[number] += [f"{project}/{run}/{file.name}"]

    print(images_available)

    return images_available

def pull_images(images: List[str], out_path: str):
    api = wandb.Api()
    "day5-gan/899kfp49/media/images/images_99912_72d30fdde2ee39797133.png"
    print(images)
    project, run, *_ = images[0].split("/")
    event = api.run(f"{project}/{run}")

    for file in images:
        print(f"Downloading {file}")
        # make sure the path is from the right run
        assert file.find(f"{project}/{run}") != -1
            
        filename = file.replace(f"{project}/{run}/", "")
        event.file(filename).download(root=out_path)

if __name__ == '__main__':
    if len(sys.argv) > 4 or len(sys.argv) < 3:
        raise ValueError("Usage: pull_images.py [project] [run] [optional:outpath]")
    project = sys.argv[1]
    run = sys.argv[2]

    if len(sys.argv) > 3:
        out = sys.argv[3]
    else:
        out = "."

    files = get_available_images(project, run)
    
    pull_images([file for group in files.values() for file in group], "./test")
