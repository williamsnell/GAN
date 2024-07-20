from gan import DCGAN
from tqdm import tqdm
from pathlib import Path
import torch
import numpy as np
import einops
import matplotlib.pyplot as plt
import cv2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# From https://github.com/davidmickisch/torch-rot/blob/main/torch_rot/rotations.py
def rotate_vector(
    theta: torch.Tensor, 
    n_1 : torch.Tensor, 
    n_2 : torch.Tensor, 
    vec : torch.Tensor) -> torch.Tensor:
    """
    This method returns a rotated vector which rotates @vec in the 2 dimensional plane spanned by 
    @n1 and @n2 by an angle @theta. The vectors @n1 and @n2 have to be orthogonal.
    Inspired by 
    https://analyticphysics.com/Higher%20Dimensions/Rotations%20in%20Higher%20Dimensions.htm
    :param @n1: first vector spanning 2-d rotation plane, needs to be orthogonal to @n2
    :param @n2: second vector spanning 2-d rotation plane, needs to be orthogonal to @n1
    :param @theta: rotation angle
    :param @vec: vector to be rotated
    :returns : rotation matrix
    """
    assert len(n_1) == len(n_2)
    assert len(n_1) == len(vec)
    assert (n_1.dot(n_2).abs() < 1e-4)
    
    n1_dot_vec = n_1.dot(vec)
    n2_dot_vec = n_2.dot(vec)
    
    return (vec +
        (n_2 * n1_dot_vec - n_1 * n2_dot_vec) * torch.sin(theta) +
        (n_1 * n1_dot_vec + n_2 * n2_dot_vec) * (torch.cos(theta) - 1)
    )

# spherical interpolation, from https://github.com/soumith/dcgan.torch/issues/14
def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high


if __name__ == '__main__':
    model = torch.load(Path("~/src/ai/gan/export/export/64_epoch_2_model.pt").expanduser(), map_location=device)
    gan = DCGAN()
    gan.load_state_dict(model)

    num_repeats = 100

    upscale_factor = 6
    framerate = 60
    rotation_duration = 0.5 # seconds
    video = cv2.VideoWriter("lerp_latent_space.mp4", cv2.VideoWriter_fourcc(*'avc1'),
                            framerate, (64 * upscale_factor, 64 * upscale_factor))

    init_vec = torch.randn(100)

    for i in tqdm(range(num_repeats)):
        # 
        # # start with random vectors
        # s1 = torch.randn(100)
        # s2 = torch.randn(100)
        # # orthogonalize them both using gram-schmidt
        # u1 = s1 / torch.linalg.vector_norm(s1)
        # s2_hat = s2.dot(u1) * u1
        # eps2 = s2 - s2_hat
        # u2 = eps2 / torch.linalg.vector_norm(eps2)
        # # rescale them to the original expected magnitude (10)
        # rotation_plane_vec_1 = u1
        # rotation_plane_vec_2 = u2
    
        new_vec = None

        with torch.inference_mode():
            # for angle in torch.linspace(0, np.pi, framerate * rotation_duration):
            #   new_vec = rotate_vector(angle, rotation_plane_vec_1, rotation_plane_vec_2, init_vec)

            vec2 = torch.randn(100)
            for fraction in np.linspace(0, 1, int(framerate * rotation_duration)):
                # new_vec = slerp(fraction, init_vec, vec2)
                new_vec = init_vec + fraction * (vec2 - init_vec)
                image = gan.netG(torch.reshape(new_vec, (1, 100)))
                normalized = (einops.rearrange(image, "b c h w -> (b h) w c") + 1) / 2
                frame = (normalized.numpy() * 255).astype(np.uint8)
                frame = np.flip(cv2.resize(frame, (64 * upscale_factor, 64 * upscale_factor), cv2.INTER_CUBIC), 2)

                video.write(frame)

            print(torch.linalg.vector_norm(new_vec))
        init_vec = new_vec


    cv2.destroyAllWindows()
    video.release()
 
