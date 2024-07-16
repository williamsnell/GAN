import wandb
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError("Usage: pull_images.py [project] [run]")
    project = sys.argv[1]
    run = sys.argv[2]


    api = wandb.Api()
    run = api.run(f"{project}/{run}")

    for file in run.files():
        if(str(file).find(".png") != -1):
            file.download()

