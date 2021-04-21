#!/usr/bin/python3

import copy
import os
import numpy as np
import PIL
import torch
import torchvision
from PIL import Image

from typing import Any, Callable, List, Tuple


def verify(function: Callable) -> str:
    """Will indicate with a print statement whether assertions passed or failed
    within function argument call.
    Indica si la aserción pasó o falló dentro de la función que se le pasa como argumento

    Parámetros:
        function: Función en python

    Retorna:
	string en rojo o verde que indica éxito o fallo
    """
    try:
        function()
        return '\x1b[32m"Correct"\x1b[0m'
    except AssertionError:
        return '\x1b[31m"Wrong"\x1b[0m'


def PIL_resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Parámetros:
        img: Array que representa a una imagen
        size: Tupla representando el nuevo tamaño (width, height)

    Retorna:
        img que es una imagen PIL
    """
    img = numpy_arr_to_PIL_image(img, scale_to_255=True)
    img = img.resize(size)
    img = PIL_image_to_numpy_arr(img)
    return img


def PIL_image_to_numpy_arr(img: Image, downscale_by_255: bool = True) -> np.ndarray:
    """
    Parámetros:
        img: PIL Image
        downscale_by_255: si dividir o no valores uint8 por 255 para normalizarlos a valores en el rango[0,1]

    Retorna:
        img
    """
    img = np.asarray(img)
    img = img.astype(np.float32)
    if downscale_by_255:
        img /= 255
    return img


def vis_image_scales_numpy(image: np.ndarray) -> np.ndarray:
    """
    Esta función mostrará una imagen a diferentes escalas (factores de zoom). La
    La imagen original aparecerá en el extremo izquierdo y luego la imagen
    iterativamente se reducirá 2x en cada imagen a la derecha.

    Esta es una forma particularmente efectiva de simular el efecto de perspectiva, ya que
    pareciera ver una imagen a diferentes distancias. Por lo tanto, lo usamos para visualizar
    imágenes híbridas, que representan una combinación de dos imágenes, como se describe
    en el artículo de SIGGRAPH 2006 "Imágenes híbridas" de Oliva, Torralba, Schyns. 

    Parámetros:
        image: Array de shape (H, W, C)

    Retorna:
        img_scales: Array de shape (M, K, C) representando imagenes apiladas horizontalemte
		, que se van haciendo mas pequeñas de izquierda derecha
            K = W + int(1/2 W + 1/4 W + 1/8 W + 1/16 W) + (5 * 4)
    """
    original_height = image.shape[0]
    original_width = image.shape[1]
    num_colors = 1 if image.ndim == 2 else 3
    img_scales = np.copy(image)
    cur_image = np.copy(image)

    scales = 5
    scale_factor = 0.5
    padding = 5

    new_h = original_height
    new_w = original_width

    for scale in range(2, scales + 1):
        # add padding
        img_scales = np.hstack(
            (
                img_scales,
                np.ones((original_height, padding, num_colors), dtype=np.float32),
            )
        )

        new_h = int(scale_factor * new_h)
        new_w = int(scale_factor * new_w)
        # downsample image iteratively
        cur_image = PIL_resize(cur_image, size=(new_w, new_h))

        # pad the top to append to the output
        h_pad = original_height - cur_image.shape[0]
        pad = np.ones((h_pad, cur_image.shape[1], num_colors), dtype=np.float32)
        tmp = np.vstack((pad, cur_image))
        img_scales = np.hstack((img_scales, tmp))

    return img_scales


def im2single(im: np.ndarray) -> np.ndarray:
    """
    Parámetros:
        img: uint8 array de shape (m,n,c) o (m,n) en un rango [0,255]

    Retorna:
        im: arreglo de floats o dobles del mismo shape en el rango [0,1]
    """
    im = im.astype(np.float32) / 255
    return im


def single2im(im: np.ndarray) -> np.ndarray:
    """
    Parámetros:
        im: arreglo float o doble cuyo shape es (m,n,c) o (m,n) en el rango [0,1]

    Retorna:
        im: arreglo uint8 del mismo shape que la imagen en el rango [0,255]
    """
    im *= 255
    im = im.astype(np.uint8)
    return im


def numpy_arr_to_PIL_image(img: np.ndarray, scale_to_255: False) -> PIL.Image:
    """
    Parámetros:
        img: en rango [0,1]

    Retorna:
        img en rango [0,255]
    """
    if scale_to_255:
        img *= 255
    return PIL.Image.fromarray(np.uint8(img))


def load_image(path: str) -> np.ndarray:
    """
    Parámetros:
        path: string que especifica el path a una imagen

    Retorna:
        float_img_rgb: arreglo de floats o dobles cuyo shape es (m,n,c) o (m,n)
           y se encuentra en el rango [0,1], representando una imagen RGB
    """
    img = PIL.Image.open(path)
    img = np.asarray(img, dtype=float)
    float_img_rgb = im2single(img)
    return float_img_rgb


def save_image(path: str, im: np.ndarray) -> None:
    """
    Parámetros:
        path: string que representa la ubicación donde se desea guardar una imagen
        img: arreglo numpy
    """
    folder_path = os.path.split(path)[0]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    img = copy.deepcopy(im)
    img = single2im(img)
    pil_img = numpy_arr_to_PIL_image(img, scale_to_255=False)
    pil_img.save(path)


def write_objects_to_file(fpath: str, obj_list: List[Any]) -> None:
    """
    Si la lista contiene datos de tipo float o int, los convierte en strings separados 
    por carriage return.

    Parámetros:
        fpath: string que representa un path a una imagen
        obj_list: Lista de strings, floats o integers a ser escritos en un archivo, uno por línea.
    """
    obj_list = [str(obj) + "\n" for obj in obj_list]
    with open(fpath, "w") as f:
        f.writelines(obj_list)


def rgb2gray(img: np.ndarray) -> np.ndarray:
    """
    Usa los coeficientes usandos por OpenCV, encontrados aquí:
    https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html

    Parámetros:
	Arreglo Numpy de shape (M,N,3) representando una imagen RGB and formato HWC

    Retorna:
        arreglo Numpy de shape (M,N) representando una imagen en escala de grises
    """
    # Grayscale coefficients
    c = [0.299, 0.587, 0.114]
    return img[:, :, 0] * c[0] + img[:, :, 1] * c[1] + img[:, :, 2] * c[2]
