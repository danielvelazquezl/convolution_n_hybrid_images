#!/usr/bin/python3

"""
PyTorch tutorial on constructing neural networks:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def my_conv2d_pytorch(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Aplica los filtros dados a la imagen de entrada.

    Parámetros:
        image: Tensor de shape (1, d1, h1, w1)
        kernel: Tensor de shape (N, d1/groups, k, k) que deben ser aplicados a la 
        imagen
    Retorna:
        filtered_image: Tensor de shape (1, d2, h2, w2) donde
           d2 = N
           h2 = (h1 - k + 2 * padding) / stride + 1
           w2 = (w1 - k + 2 * padding) / stride + 1

    TIPS:
    - Debesa usar la operación de convolución de torch.nn.functional.
    - En PyTorch, d1 es `in_channels`, d2 es `out_channels`
    - Asegurate de aplicar el padding a la imagen de forma apropiada (es un parámetro 
    para la función de convolución que vas a usar acá)
    - Puedes asumir que el número de grupos es igual al número de canales de entrada.
    - Puedesa asumir que solo usaremos filtros cuadrados para esta función
    """

    # return F.conv2d(image, kernel, padding=kernel.shape[2] // 2, groups=image.shape[1], stride=1)
    return F.conv2d(image, kernel, padding=1, groups=image.shape[1], stride=2)
