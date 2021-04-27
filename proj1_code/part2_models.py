#!/usr/bin/python3

"""
PyTorch tutorial on constructing neural networks:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as transforms

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
        (c, 1, k, k) donde c es el número de canales en la imagen. Empiece por obtener 
        un Kernel gaussiano 2D usando su implementación de la Parte 1, que será de 
        forma (k, k). Entonces, digamos que tiene una imagen RGB, necesitará convertir 
        esto en un tensor de forma (3, 1, k, k) apilando el kernel gaussiano 3 veces. 

        Parámetros
            cutoff_frequency: int que especifica el valor de cutoff para las 
            frecuencias.
        Returns
            kernel: Tensor de shape (c, 1, k, k) donde c es  el # de canáles

        TIPS:
        - Utilizará la función create_Gaussian_kernel_2D () de part1.py en esta 
        función. 
        - Dado que los # de canales pueden diferir en cada imagen del dataset, 
        asegúrese de no usar valores fijos para las dimensiones del kernel. Hay una 
        variable definida en esta clase para proveer información acerca del número de 
        canáles.
	    - Puedes usar np.reshape() para cambiar las dimensiones de un arreglo numpy
	    - Puedes usar np.tile() para repetir un arreglo numpy en axes específicados.
	    - Puedes usar torch.Tensor() para convertir arreglos numpy a Tensores de Torch.
        """
        
        kernel_size = cutoff_frequency * 4 + 1
        kernel2d_with_channels = np.zeros((self.n_channels, 1, kernel_size, kernel_size))
        for i in range(self.n_channels):
            kernel2d_with_channels[i,:,:] = create_Gaussian_kernel_2D(cutoff_frequency)

        return torch.Tensor(kernel2d_with_channels)

    
    def low_pass(self, x: torch.Tensor, kernel: torch.Tensor):
        """
        Applies low pass filter to the input image.
        Parámetros:
            x: Tensor de shape (b, c, m, n) donde b es el batch size
            kernel: filtro low-pass a ser aplicado a la imagen
        Retorna:
            filtered_image: Tensor de shape (b, c, m, n)

        Tips:
        - Debe utilizar el operador de convolución 2d de torch.nn.functional. 
	    - Asegúrese de rellenar (padding) la imagen de forma adecuada (pad es un 
        parámetro de la función de convolución que debe utilizar aquí). 
	    - Pase self.n_channels como el valor del parámetro "groups" de la función de 
        convolución. Esto representa el # de canales a los que se aplicará el filtro. 
        """

        return F.conv2d(x, kernel, padding=kernel.shape[2] // 2, groups=self.n_channels)

    
    def forward(
        self, image1: torch.Tensor, image2: torch.Tensor, cutoff_frequency: torch.Tensor
    ):
        """
        Toma dos imágenes y crea una image híbrida. Retorna las frecuencias bajas de la 
        primera imagen, las frecuencias altas de la segunda y la imagen híbrida en sí.

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
        - Similar a la parte 1, puedes obtener las frecuencias altas de una imagen 
        restándole sus frecuencias bajas.
        - No olvides de asegurarte de aplicar 'clippping' a la imagen. Valores deben 
        estar entre 0 y 1. Puedes usar torch.clamp()
        - Si quieres usar imágenes con dimensiones diferentes, primero debes 
        redimensionarles en la clase HubridImagesDataset usando tochvision.transforms
        """
        self.n_channels = image1.shape[1]

        kernel = self.get_kernel(int(cutoff_frequency))
        low_frequencies = self.low_pass(image1, kernel)
        high_frequencies = image2 - self.low_pass(image2, kernel)
        hybrid_image = low_frequencies + high_frequencies

        return low_frequencies, high_frequencies, torch.clamp(hybrid_image, 0, 1)
