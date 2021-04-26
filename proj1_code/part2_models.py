#!/usr/bin/python3

"""
PyTorch tutorial on constructing neural networks:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from proj1_code.part1 import create_Gaussian_kernel_2D


class HybridImageModel(nn.Module):
    def __init__(self):
        """
        Inicializa una instancia de la clase HybridImageModel.
        """
        super(HybridImageModel, self).__init__()

    def get_kernel(self, cutoff_frequency: int) -> torch.Tensor:
        """
        Retorna un kernel Gaussiano que utiliza el valor de cutoff específicado.


        PyTorch requiere que el kernel tenga una forma particular para poder
        aplicarlo a una imagen. Específicamente, el kernel debe estar en la forma
        (c, 1, k, k) donde c es el número de canales en la imagen. Empiece por obtener un
        Kernel gaussiano 2D usando su implementación de la Parte 1, que será
        de forma (k, k). Entonces, digamos que tiene una imagen RGB, necesitará
        convertir esto en un tensor de forma (3, 1, k, k) apilando el kernel gaussiano
        3 veces. 

        Parámetros
            cutoff_frequency: int que especifica el valor de cutoff para las frecuencias.
        Returns
            kernel: Tensor de shape (c, 1, k, k) donde c es  el # de canáles

        TIPS:
        - Utilizará la función create_Gaussian_kernel_2D () de part1.py en
          esta función. 
        - Dado que los # de canales pueden diferir en cada imagen del dataset,
          asegúrese de no usar valores fijos para las dimensiones del kernel
          Hay una variable definida en esta clase para proveer información acerca del número de canáles.
	- Puedes usar np.reshape() para cambiar las dimensiones de un arreglo numpy
	- Puedes usar np.tile() para repetir un arreglo numpy en axes específicados.
	- Puedes usar torch.Tensor() para convertir arreglos numpy a Tensores de Torch.
        """

        ############################
        ### TODO: EL CÓDIGO EMPIEZA ACÁ ###

        raise NotImplementedError(
            "La función `get_kernel` debe ser implementada in `part2_models.py`"
        )

        ### EL CÓDIGO TERMINA ACÁ ####
        ############################

        return kernel

    def low_pass(self, x: torch.Tensor, kernel: torch.Tensor):
        """
        Applies low pass filter to the input image. Aplica un filtro low-pass a la imagen. 
        Parámetros:
            x: Tensor de shape (b, c, m, n) donde b es el batch size
            kernel: filtro low-pass a ser aplicado a la imagen
        Retorna:
            filtered_image: Tensor de shape (b, c, m, n)

        Tips:
        - Debe utilizar el operador de convolución 2d de torch.nn.functional. 
	- Asegúrese de rellenar (padding) la imagen de forma adecuada (pad es un parámetro de la
          función de convolución que debe utilizar aquí). 
	- Pase self.n_channels como el valor del parámetro "groups" de
          la función de convolución. Esto representa el # de canales a los que
          se aplicará el filtro. 
        """

        ############################
        ### TODO: EL CÓDIGO EMPIEZA AQUÍ ###

        raise NotImplementedError(
            "La función `low_pass` debe ser implementada in `part2_models.py`"
        )

        ### EL CÓDIGO TERMINA AQUÍ ####
        ############################

        return filtered_image

    def forward(
        self, image1: torch.Tensor, image2: torch.Tensor, cutoff_frequency: torch.Tensor
    ):
        """
	Toma dos imágenes y crea una image híbrida. Retorna las frecuencias bajas 
	de la primera imagen, las frecuencias altas de la segunda y la imagen híbrida en sí.

        Parámetros:
            image1: Tensor de shape (b, c, m, n)
            image2: Tensor de shape (b, c, m, n)
            cutoff_frequency: Tensor de shape (b)
        Retorna:
            low_frequencies: Tensor de shape (b, c, m, n)
            high_frequencies: Tensor de shape (b, c, m, n)
            hybrid_image: Tensor de shape (b, c, m, n)

        TIPS:
	- Vas a la función get_kernel() y la función low_pass() en esta función.
	- Similar a la parte 1, puedes obtener las frecuencias altas de una imagen restándole sus frecuencias bajas.
	- No olvides de asegurarte de aplicar 'clippping' a la imagen. Valores deben estar entre 0 y 1. Puedes usar torch.clamp()
	- Si quieres usar imágenes con dimensiones diferentes, primero debes redimensionarles en la clase HubridImagesDataset usando tochvision.transforms
        """
        self.n_channels = image1.shape[1]

        ############################
        ### TODO: EL CÓDIGO EMPIEZA ACÁ ###

        raise NotImplementedError(
            "La función `forward` debe ser implementada en `part2_models.py`"
        )

        ### EL CÓDIGO TERMINA ACÁ ####
        ############################

        return low_frequencies, high_frequencies, hybrid_image
