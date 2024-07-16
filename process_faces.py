from PIL import Image
import numpy as np
import cv2
from pathlib import Path

video_name = "gan_full.mp4"

image_steps = []

for image in Path("./gan_full/images/faces").glob("*.png"):
    print(str(image.stem))

    image_steps += [int(image.stem.strip("images_").split("_")[0])]

# map sequence number (in the discriminator/generator layer exports)
# to the face images
map = {i: number for i, number in enumerate(sorted(image_steps)[::4])}

print(map)

framerate = 25 # frames per second

duration_per_image = np.logspace(0.08, -1, len(map))
frames_per_image = np.round(duration_per_image * framerate) 

height = 480 + 480 + 40 + 128
width = max((128 + 10) * 4, 640)

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), framerate, (width, height))

face_pad = 10
gen_pad = 50
discrim_pad = -50
face_size = 128
face_offset = (width - face_pad*4 - face_size * 4)//2
face_start = 100

duration = 30 # seconds
hold_at_end = 5 # seconds
total_frames = duration * framerate
hold_frames = hold_at_end * framerate
frame_number = np.concatenate((np.floor(np.cumsum(np.logspace(-1, 0.3, total_frames - hold_frames))), np.array([max(map.keys()) - 1] * hold_frames)))

# for n, (i, number) in zip(range(100), map.items()):
for i in frame_number:
    number = map[i]
    # load files
    faces = [cv2.imread(path) for path in Path("./gan_full/images/faces").glob(f"images_{number}_*.png")]
    discrim = cv2.imread(Path(f"./gan_full/images/discriminator/{100_000_000 + int(i)}.png"))
    gen = cv2.imread(Path(f"./gan_full/images/generator/{100_000_000 + int(i)}.png"))


    # crop gen
    gen = np.concatenate((gen[:40], gen[100:]), axis=0)
    # crop discrim
    discrim = np.concatenate((discrim[:40], discrim[100:]), axis=0)


    # Image is shape (width, height, channel)
    image = np.ones((height, width, 3), dtype=np.uint8) * 255 # Initialize as white

    # Composite in the layers
    face_end = face_start + face_size

    for i, face in enumerate(faces):
        face = cv2.resize(face, dsize=(face_size, face_size), interpolation=cv2.INTER_NEAREST)
        # upscale
        image[face_start:face_end, face_offset + i * (face_size + face_pad) : face_offset + i * face_pad + (i+1)*face_size] = face



    gen_start = face_end + gen_pad
    gen_end = gen_start + gen.shape[0]
    image[gen_start:gen_end] = gen

    discrim_start = gen_end + discrim_pad
    discrim_end = discrim_start + discrim.shape[0]
    image[discrim_start:discrim_end] = discrim

    # Composite on the text
    image = cv2.putText(image, f"Images seen: {number}", (130, 40), cv2.FONT_HERSHEY_COMPLEX, 
                fontScale=1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)


    video.write(image)


cv2.destroyAllWindows()
video.release()
