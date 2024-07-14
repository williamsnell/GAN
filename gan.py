import matplotlib.pyplot as plt
import sys
import torch as t
from torch import nn, optim
import einops
from einops.layers.torch import Rearrange
from tqdm import tqdm
from datasets import load_dataset
from dataclasses import dataclass, field
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from typing import Optional, Tuple, List, Literal, Union
import plotly.express as px
import torchinfo
import time
import wandb
from PIL import Image
import pandas as pd
from pathlib import Path
from collections import OrderedDict

device = t.device("cuda" if t.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: List[int] = [128, 256, 512],
    ):
        '''
        Implements the generator architecture from the DCGAN paper (the diagram at the top
        of page 4). We assume the size of the activations doubles at each layer (so image
        size has to be divisible by 2 ** len(hidden_channels)).

        Args:
            latent_dim_size:
                the size of the latent dimension, i.e. the input to the generator
            img_size:
                the size of the image, i.e. the output of the generator
            img_channels:
                the number of channels in the image (3 for RGB, 1 for grayscale)
            hidden_channels:
                the number of channels in the hidden layers of the generator (starting from
                the smallest / closest to the generated images, and working backwards to the
                latent vector).
        '''
        n_layers = len(hidden_channels)
        assert img_size % (2 ** n_layers) == 0, "activation size must double at each layer"

        super().__init__()
        spatial_resolutions = [img_size // 2**(i + 1) for i in range(n_layers)]

        self.project_and_reshape = nn.Sequential(
            nn.Linear(latent_dim_size, hidden_channels[-1] * spatial_resolutions[-1]**2, bias=False),
            einops.layers.torch.Rearrange("b (c h w) -> b c h w", c=hidden_channels[-1], h=spatial_resolutions[-1], w=spatial_resolutions[-1]),
            nn.BatchNorm2d(hidden_channels[-1]),
            nn.ReLU()
        )

        hidden_layers = [component
                         for in_channels, out_channels in zip(hidden_channels[::-1], hidden_channels[::-1][1:])
                         for component in [
                            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU()
                        ]]



        self.hidden_layers = nn.Sequential(OrderedDict(**{
            "hidden_layers": nn.Sequential(*hidden_layers),
            "output_kernels": nn.ConvTranspose2d(hidden_channels[0], img_channels,
                                kernel_size=4, stride=2, padding=1),
            "activation": nn.Tanh()})
        )



    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.project_and_reshape(x)
        x = self.hidden_layers(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: List[int] = [128, 256, 512],
    ):
        '''
        Implements the discriminator architecture from the DCGAN paper (the mirror image of
        the diagram at the top of page 4). We assume the size of the activations doubles at
        each layer (so image size has to be divisible by 2 ** len(hidden_channels)).

        Args:
            img_size:
                the size of the image, i.e. the input of the discriminator
            img_channels:
                the number of channels in the image (3 for RGB, 1 for grayscale)
            hidden_channels:
                the number of channels in the hidden layers of the discriminator (starting from
                the smallest / closest to the input image, and working forwards to the probability
                output).
        '''
        n_layers = len(hidden_channels)
        assert img_size % (2 ** n_layers) == 0, "activation size must double at each layer"

        self.kernel_size = 4
        self.stride = 2
        self.padding = 1

        super().__init__()

        hidden_layers = [component
             for in_channels, out_channels in zip(hidden_channels, hidden_channels[1:])
             for component in [
                nn.Conv2d(in_channels, out_channels, self.kernel_size,
                          self.stride, self.padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            ]]

        self.hidden_layers = nn.Sequential(
            OrderedDict(**{
                "input_kernel": nn.Conv2d(img_channels, hidden_channels[0], self.kernel_size,
                        self.stride, self.padding, bias=False),
                "activation": nn.LeakyReLU(),
                "hidden_layers": nn.Sequential(*hidden_layers)
            }))

        output_size = (img_size // (2 ** n_layers))**2 * hidden_channels[-1]

        self.project_and_reshape = nn.Sequential(
            einops.layers.torch.Rearrange("b c h w -> b (c h w)"),
            nn.Linear(output_size, 1),
            nn.Sigmoid()
        )


    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.hidden_layers(x)
        x = self.project_and_reshape(x)
        return x


class DCGAN(nn.Module):
    netD: Discriminator
    netG: Generator

    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: List[int] = [128, 256, 512],
    ):
        super().__init__()
        self.netD = Discriminator(img_size, img_channels, hidden_channels)
        self.netG = Generator(latent_dim_size, img_size, img_channels,
                              hidden_channels)


model = DCGAN().to(device)
x = t.randn(3, 100).to(device)
statsG = torchinfo.summary(model.netG, input_data=x)
statsD = torchinfo.summary(model.netD, input_data=model.netG(x))
print(statsG, statsD)

def initialize_weights(model: nn.Module) -> None:
    '''
    Initializes weights according to the DCGAN paper (details at the end of
    page 3), by modifying the weights of the model in place.
    '''
    for name, module in model.named_modules():
      if isinstance(module, nn.BatchNorm2d):
        # init N(1, 0.02), bias = 0
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
      elif any([isinstance(module, layer) for layer in [nn.Linear, nn.Conv2d, nn.ConvTranspose2d]]):
        # init N(0, 0.02)
        nn.init.normal_(module.weight.data, 0.0, 0.02)

def get_dataset(dataset: Literal["MNIST", "CELEB"], train: bool = True) -> Dataset:
    assert dataset in ["MNIST", "CELEB", "BOTH"]

    if dataset == "CELEB":
        image_size = 64
        assert train, "CelebA dataset only has a training set"
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = datasets.ImageFolder(
            root =  Path(".")/ "data" / "celeba",
            transform = transform
        )

    elif dataset == "MNIST":
        img_size = 28
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST(
            root = Path(".") / "data",
            transform = transform,
            download = True,
        )

    return trainset

def display_data(x: t.Tensor, nrows: int, title: str):
    '''Displays a batch of data, using plotly.'''
    # Reshape into the right shape for plotting (make it 2D if image is monochrome)
    y = einops.rearrange(x, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=nrows).squeeze()
    # Normalize, in the 0-1 range
    y = (y - y.min()) / (y.max() - y.min())
    # Display data
    plt.imshow(
        y,
    )
    plt.show()


@dataclass
class DCGANArgs():
    '''
    Class for the arguments to the DCGAN (training and architecture).
    Note, we use field(defaultfactory(...)) when our default value is a mutable object.
    '''
    latent_dim_size: int = 100
    hidden_channels: List[int] = field(default_factory=lambda: [128, 256, 512])
    dataset: Literal["MNIST", "CELEB"] = "CELEB"
    batch_size: int = 64
    epochs: int = 3
    lr: float = 0.0002
    betas: Tuple[float, float] = (0.5, 0.999)
    seconds_between_eval: int = 20
    wandb_project: Optional[str] = 'day5-gan'
    wandb_name: Optional[str] = None


class DCGANTrainer:
    def __init__(self, args: DCGANArgs):
        self.args = args
        self.criterion = nn.BCELoss()

        self.trainset = get_dataset(self.args.dataset)
        self.trainloader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)

        batch, img_channels, img_height, img_width = next(iter(self.trainloader))[0].shape
        assert img_height == img_width

        self.model = DCGAN(
            args.latent_dim_size,
            img_height,
            img_channels,
            args.hidden_channels,
        ).to(device).train()

        self.optG = t.optim.Adam(self.model.netG.parameters(), lr=args.lr, betas=args.betas)
        self.optD = t.optim.Adam(self.model.netD.parameters(), lr=args.lr, betas=args.betas, maximize=True)
        
        [initialize_weights(model) for model in self.model.modules()]
        
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
        wandb.watch((self.model.netG.hidden_layers[-2], self.model.netD.hidden_layers[0]), log_freq=20)


    def training_step_discriminator(self, img_real: t.Tensor, img_fake: t.Tensor) -> t.Tensor:
        '''
        Generates a real and fake image, and performs a gradient step on the discriminator
        to maximize log(D(x)) + log(1-D(G(z))).
        '''
        self.optD.zero_grad()

        real = self.model.netD(img_real)
        fake = self.model.netD(img_fake)

        loss = (t.log(real) + t.log(1 - fake)).mean()
        loss.backward()

        self.optD.step()

        return loss.detach()

    def training_step_generator(self, img_fake: t.Tensor) -> t.Tensor:
        '''
        Performs a gradient step on the generator to maximize log(D(G(z))).
        '''
        self.optG.zero_grad()
        self.optD.zero_grad() # do we need to do this?

        loss = -t.log(self.model.netD(img_fake)).sum()

        loss.backward()

        self.optG.step()

        return loss.detach()


    @t.inference_mode()
    def evaluate(self) -> None:
        '''
        Performs evaluation by generating 8 instances of random noise and passing them through
        the generator.
        '''
        noise = t.randn(8, self.args.latent_dim_size).to(device)
        return self.model.netG(noise)


    def train(self) -> None:
        '''
        Performs a full training run, logging to wandb.
        '''
        self.step = 0
        last_log_time = time.time()

        for epoch in range(self.args.epochs):

            progress_bar = tqdm(self.trainloader, total=len(self.trainloader))

            for (img_real, label) in progress_bar:

                # Generate random noise & fake image
                noise = t.randn(self.args.batch_size, self.args.latent_dim_size).to(device)
                img_real = img_real.to(device)
                img_fake = self.model.netG(noise)

                # Training steps
                lossD = self.training_step_discriminator(img_real, img_fake.detach())
                lossG = self.training_step_generator(img_fake)

                # Log data
                wandb.log(dict(lossD=lossD, lossG=lossG), step=self.step)

                # Update progress bar
                self.step += img_real.shape[0]
                progress_bar.set_description(f"{epoch=}, lossD={lossD:.4f}, lossG={lossG:.4f}, examples_seen={self.step}")

                # Evaluate model on the same batch of random data
                if time.time() - last_log_time > self.args.seconds_between_eval:
                    last_log_time = time.time()
                    images = [wandb.Image(a) for a in img_fake.detach()]
                    wandb.log({"images": images[:4]}, 
                              step=self.step)
                    t.save({
                        'epoch': epoch,
                        'generator_state_dict': self.model.netG.hidden_layers[-2].state_dict(),
                        'discriminator_state_dict': self.model.netD.hidden_layers[0].state_dict(),
                        },
                        f"{self.step}_model.pt")

        wandb.finish()


