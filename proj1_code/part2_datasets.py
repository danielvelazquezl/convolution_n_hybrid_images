#!/usr/bin/python3

"""
PyTorch tutorial on data loading & processing:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import os
from typing import List, Tuple

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms


def make_dataset(path: str) -> Tuple[List[str], List[str]]:
    """
    Crear un dataset de parejas de imágenes que se encuentran en un directorio.
    
    El dataset debe tener 2 conjuntos. Un conjunto contiene imágenes sobre las que se va a pasar el low-pass filter
    y el otro conjunto va a ser el de las imágenes sobre las que se va a pasar el high-pass filter.

    Parámetros:
        path: string especificando el path a las imágenes
    Retorna:
        images_a: lista de strings conteniendo los paths a lás imágenes del conjunto A ordenados de forma lexicográfica.
        images_b: lista de strings conteniendo los paths a lás imágenes del conjunto B ordenados de forma lexicográfica.
    """

    ############################
    ### TODO: EL CÓDIGO EMPIEZA ACÁ ###

    raise NotImplementedError(
        "La función `make_dataset` debe ser implementada en `part2_datasets.py`
    )

    ### EL CÓDIGO TERMINA ACÁ ####
    ############################

    return images_a, images_b


def get_cutoff_frequencies(path: str) -> List[int]:
    """
    Obtiene el valor de cutoff para las frecuencias correspondientes a cada par de imágenes
    
    Los valores de frecuencia cutoff son valores que descubriste de tus experimentos en la parte 1 del trabajo.

    Parámetros:
        path: string conteniendo el path al .txt que contiene los valores de cutoff para las frecuencias
    Retorna:
        cutoff_frequencies: numpy array de integers. El arreglo debe tener la misma longitud que el número de imágenes en el dataset de parejas de imágenes.
    """

    ############################
    ### TODO: EL CÓDIGO EMPIEZA ACÁ ###

    raise NotImplementedError(
        "La función `get_cutoff_frequencies` debe ser implementada en "
        + "`part2_datasets.py`"
    )

    ### EL CÓDIGO TERMINA ACÁ ####
    ############################

    return cutoff_frequencies


class HybridImageDataset(data.Dataset):
    """Hybrid images dataset."""

    def __init__(self, image_dir: str, cf_file: str) -> None:
        """
        HybridImageDataset class constructor.

        Debe reemplazar self.transform con la transformación apropiada de
        torchvision.transforms que convierte una imagen PIL en un tensor de Torch.
        Puedes especificar transformaciones adicionales (por ejemplo, cambio de tamaño de la imagen) si lo desea,
        pero no es necesario para las imágenes que le proporcionamos ya que cada par tiene
        las mismas dimensiones. 

        Parámetros:
            image_dir: string que especifica el directorio que contiene a las imágenes
            cf_file: string que especifica el path del archivo .txt con los valores de cutoff para las frecuencias.
        """
        images_a, images_b = make_dataset(image_dir)
        cutoff_frequencies = get_cutoff_frequencies(cf_file)

        self.transform = None
        ############################
        ### TODO: EL CÓDIGO EMPIEZA ACÁ ###

        raise NotImplementedError(
            "`La función self.transform` debe ser implementada en `part2_datasets.py`"
        )

        #### EL CÓDIGO TERMINA ACÁ ####
        ############################

        self.images_a = images_a
        self.images_b = images_b
        self.cutoff_frequencies = cutoff_frequencies

    def __len__(self) -> int:
	""" Retorna el número de pares de imágenes que hay en el dataset."""

        ############################
        ### TODO: EL CÓDIGO EMPIEZA ACÁ ###

        raise NotImplementedError(
            "La función `__len__` debe implementarse en `part2_datasets.py`"
        )

        ### EL CÓDIGO TERMINA ACÁ ####
        ############################

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Retorna el par de imágenes y los valores de cutoff correspondientes en la posición `idx`.

	Dado que self.images_a y self.images_b contienen rutas a las imágenes,
        Debería leer las imágenes aquí y normalizar los píxeles para que estén entre 0
        y 1. Asegúrese de transponer las dimensiones de modo que image_a y
        image_b tengan la forma (c, m, n) en lugar de (m, n, c), y
        conviértalos en tensores de Torch. 

        Parámetros:
            idx: Integer que especifica el índice desde donde los datos deben devolverse.
        Retorna:
            image_a: Tensor de shape (c, m, n)
            image_b: Tensor de shape (c, m, n)
            cutoff_frequency: valor integer que especifica el valor cutoff que corresponde al par de imágenes (image_a, image_b)

        TIPS:
        - Usa la librería PIL para leer imágenes
        - Usa self.transform para convertir la imagen PIL a un Tensor de Torch.
        """

        ############################
        ### TODO: EL CÓDIGO EMPIEZA ACÁ ###

        raise NotImplementedError(
            "La función `__getitem__ debe ser implementada en `part2_datasets.py`"
        )

        ### EL CÓDIGO TERMINA ACÁ ####
        ############################

        return image_a, image_b, cutoff_frequency
