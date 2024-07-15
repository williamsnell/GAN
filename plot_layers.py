import torch
from pathlib import Path
import matplotlib.pyplot as plt

for checkpoint in Path(".").glob("*.pt"):

    model = torch.load(checkpoint)

    generator = model['generator_state_dict']

    out_kernel_weight = generator['hidden_layers.output_kernels.weight']
    out_kernel_bias = generator['hidden_layers.output_kernels.bias']

    [plt.imshow(torch.nn.functional.tanh(weight + bias)) for weight, bias in zip(out_kernel_weight, out_kernel_bias)]
    plt.show()
