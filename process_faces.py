from PIL import Image, ImageFont, ImageDraw
import pickle
import torch
import einops
import os
import numpy as np
import cv2
import wandb
from query_wandb import get_available_images, pull_images
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class VideoSettings:
    name: str
    duration: int = 30 # seconds
    framerate: int = 30 # frames per second
    pad_vertical: int = 10 # pixels
    height: int = 900 # pixels
    width: int = 640 # pixels
    text_height: int = 60 # pixels
    hold_at_end: int = 5 # seconds

# pre-process by getting all available images
# for image in Path("./gan_full/images/faces").glob("*.png"):
#     image_steps += [int(image.stem.strip("images_").split("_")[0])]

# # map sequence number (in the discriminator/generator layer exports)
# # to the face images
# map = {i: number for i, number in enumerate(sorted(image_steps)[::4])}

# get the fonts
font_path = os.path.join(np.__path__[0], "qt", "fonts", "DejaVuSans.ttf")
titleFont = ImageFont.truetype(font_path, size=20)
subheadFont = ImageFont.truetype(font_path, size=15)

def write_text(content, width, height, xpos, ypos, font=subheadFont):
    text = Image.new("RGB", (width, height), color='white')
    draw = ImageDraw.Draw(text)
    draw.text((xpos, ypos), content, fill="black", font=font)

    return text

def composite_frame(gen, discrim, faces, image_number, settings: VideoSettings):
    # pad horizontally
    inputs = [faces, gen, discrim]

    for i, raw in enumerate(inputs):
        padded = np.ones((raw.shape[0], settings.width, 3)) 
        assert raw.shape[1] < settings.width, f"{raw.shape} input too wide"
        offset = (settings.width - raw.shape[1]) // 2
        padded[:, offset:raw.shape[1] + offset, ...] = raw

        inputs[i] = (padded * 255).astype(np.uint8)

    # Image is shape (width, height, channel)
    image = np.ones((settings.height, settings.width, 3), dtype=np.uint8) * 255 # Initialize as white

    inputs = [
        np.asarray(write_text(f"Images seen: {image_number}", settings.width, 50, 220, 30)),
        np.asarray(write_text("Generated Faces", settings.width, settings.text_height, 220, 30, titleFont)),
        inputs[0],
        np.asarray(write_text("Generator Last Layer", settings.width, settings.text_height, 200, 30, titleFont)),
        inputs[1],
        np.asarray(write_text("Discriminator First Layer", settings.width, settings.text_height, 200, 30, titleFont)),
        inputs[2],
            ]

    # Do all the compositing

    position = settings.pad_vertical

    for input in inputs:
        new_position = input.shape[0] + position
        image[position:new_position, ...] = input
        position = new_position + settings.pad_vertical
 

    # # Composite on the text
    # text = Image.new("RGB", (settings.width, settings.text_height), color='white')
    # draw = ImageDraw.Draw(text)
    # draw.text((240, 30), f"Images seen: {image_number}", fill="black", font=subheadFont)
    # draw.text((240, 120), f"Generated Images", fill="black", font=titleFont)

    return image


def collect_stats(checkpoint, stats):
    model = torch.load(checkpoint, map_location=torch.device('cpu'))

    generator = model['generator_state_dict']
    out_kernel_weight = generator['weight']

    stats['generator_max'] = max(out_kernel_weight.max(), stats['generator_max'])
    stats['generator_min'] = min(out_kernel_weight.min(), stats['generator_min'])

    discrim = model['discriminator_state_dict']
    in_kernel_weight = discrim['weight']
    
    stats['discrim_max'] = max(in_kernel_weight.max(), stats['discrim_max'])
    stats['discrim_min'] = min(in_kernel_weight.min(), stats['discrim_min'])

    
def make_video(settings: VideoSettings, path, glob):
    total_frames = settings.duration * settings.framerate
    hold_frames = settings.hold_at_end * settings.framerate
    #  frame_number = np.concatenate((np.floor(np.cumsum(np.logspace(-1, 0.3, total_frames - hold_frames))), np.array([max(map.keys()) - 1] * hold_frames)))
    
    stats = defaultdict(int)
        
    files = [c for c in sorted(Path("~/src/ai/gan/export/export/steps/").expanduser().glob(f"discrim [{layers},*.pt"),
                                                      key=lambda x: int(str(x).split("]")[-1].split("_")[0]))]

    total_files = len(files)

    frame_to_checkpoint = {}

    # get stats for normalizing images later
    for i, checkpoint in enumerate(files):
        collect_stats(checkpoint, stats)
        frame_to_checkpoint[i] = checkpoint 
     

    # print(stats)
    print(total_files)

    frame_number = np.concatenate((np.floor(np.linspace(0.9**(1./3), (total_files - 1)**(1./3), total_frames - hold_frames)**3), np.array([int(total_files - 1)] * hold_frames)))

    video = cv2.VideoWriter(settings.name, cv2.VideoWriter_fourcc(*'avc1'),
                            settings.framerate, (settings.width, settings.height))


    for i in frame_number:
        checkpoint = frame_to_checkpoint[i]
        image_number = int(str(checkpoint.stem).split("]")[-1].split("_")[0]) - 100_000_000
        
        model = torch.load(checkpoint, map_location=torch.device('cpu'))
        generator = model['generator_state_dict']
        out_kernel_weight = generator['weight']

        discriminator = model['discriminator_state_dict']
        in_kernel_weight = discriminator['weight']

        faces = model['images']

        # rescale
        upscale_factor = 6
        gen = einops.repeat(out_kernel_weight, "f c h w -> f c (h h2) (w w2)", h2=upscale_factor, w2=upscale_factor)
        discrim = einops.repeat(in_kernel_weight, "f c h w -> f c (h h2) (w w2)", h2=upscale_factor, w2=upscale_factor)

        # pad everything to give it a border between features
        gen = torch.nn.functional.pad(gen, (1, 1, 1, 1), "constant", gen.min())
        discrim = torch.nn.functional.pad(discrim, (1, 1, 1, 1), "constant", discrim.min())
        faces = torch.nn.functional.pad(faces, (1, 1, 1, 1), "constant", 1)

        # rearrange everything into image shape
        gen = einops.rearrange(gen, "(feature1 feature2) channel height width -> (feature1 width) (feature2 height) channel", feature1=8) 
        discrim = einops.repeat(discrim, "(feature1 feature2) channel height width -> (feature1 width) (feature2 height) channel", feature2=8) 

        # only keep the first 4 faces
        face_upscale_factor = 2
        faces = einops.repeat(faces, "(b1 b2) c h w -> b1 (h h2) (b2 w w2) c", b1=2, h2=face_upscale_factor, w2=face_upscale_factor)[0]


        # normalize everything
        # gen = (gen - stats['generator_min']) / (stats['generator_max'] - stats['generator_min'])
        gen = (gen - gen.min()) / (gen.max() - gen.min())
        # discrim = (discrim - stats['discriminator_min']) / (stats['discriminator_max'] - stats['discriminator_min'])
        discrim = (discrim - discrim.min()) / (discrim.max() - discrim.min())
        faces = (faces + 1) / 2
        # fix the rgb colors
        faces = torch.flip(faces, dims=(2,))

        frame = composite_frame(gen, discrim, faces, image_number, settings)
        
        video.write(frame)



    # # for n, (i, number) in zip(range(100), map.items()):
    # for i in frame_number:
    #     number = map[i]
    #     # load files
    #     pull_images(images, out_path="./faces")
    #     faces = [cv2.imread(str(path)) for path in Path("./gan_full/images/faces").glob(f"images_{number}_*.png")]
    #     discrim = cv2.imread(str(Path(f"{discrim_path}/{100_000_000 + int(i)}.png")))
    #     gen = cv2.imread(str(Path(f"{gen_path}/{100_000_000 + int(i)}.png")))

    #     frame = composite_frame(gen, discrim, faces, number, settings)
    #     video.write(frame)


    cv2.destroyAllWindows()
    video.release()
  
def cached_get_available_faces(project, run):
    cache_path = f"./cache/{project}{run}face_images_available.pkl"
    try:
        cached = Path(cache_path).read_bytes()
        face_images_available = pickle.loads(cached)

    except:
        face_images_available = get_available_images(project, run)

        Path(cache_path).write_bytes(pickle.dumps(face_images_available))
    return face_images_available


if __name__ == '__main__':
    for layers in [8, 32, 64]:
        settings = VideoSettings(f"first_layer_{layers}.mp4", duration=20, hold_at_end=5)

        make_video(settings, Path("~/src/ai/gan/export/export/steps").expanduser(), f"discrim [{layers},*.pt")
