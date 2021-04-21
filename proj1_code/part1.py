#!/usr/bin/python3

from typing import Tuple

import numpy as np
import math

def univariate_gaussian(x, mean, variance):
    """
    Params:
        x: index of array
        mean
        variance
    Return:
        value for [x] position of kernel
    """
    return (1 / (np.sqrt(2 * np.pi) * variance)) * np.exp(-(x - mean)**2 / (2 * variance**2))

def create_Gaussian_kernel_1D(ksize: int, sigma: int) -> np.ndarray:
    """
    Cree un kernel gaussiano 1D utilizando el tamaño de filtro y la desviación estándar especificados.
    
    El kernel debe tener:
    - shape (k,1)
    - mean = floor (ksize / 2)
    - los valores deben sumar 1
    
    Parámetros:
        ksize: longitud del kernel
        sigma: desviación estandar para la distribución Gaussiana
    
    Retorna:
        kernel: vector columna normalizado de 1D cuyo shape es (k, 1)
    
    TIPS:
    - Puedes calcular el valor del vector usando la función de densidad probabilística (pdf) Gaussiana [la formula] 
        en la cual el vector es una línea y el valor mas alto del vector se encuentra 
    - El objetivo es discretizar los valores que se extraen de dicha fórmula dentro de un vector de 1 dimensión. 
    """

    mean = math.floor(ksize / 2) # posicion central en el kernel
    kernel = np.zeros((ksize, 1))
    with np.nditer(kernel, op_flags=['readwrite'], flags=['f_index']) as it:
        for x in it:
            x[...] = univariate_gaussian(it.index, mean, sigma)

    return kernel / kernel.sum()

def create_Gaussian_kernel_2D(cutoff_frequency: int) -> np.ndarray:
    """
    Cree un kernel gaussiano 2D utilizando el tamaño de filtro, la desviación estándar y 
    el valor cutoff especificados.

    El kernel debe tener:
    - shape (k, k) donde k = cutoff_frequency * 4 + 1
    - mean = floor(k / 2)
    - standard deviation = cutoff_frequency
    - los valores deben sumar 1

    Args:
        cutoff_frequency: integer que controla que tantos valores de baja frecuencia dejar 
        en la imagen. Este parámetro controla K y la desviación estandar.
    Retorna:
        kernel: numpy nd-array de shape (k, k)

    TIPS:
    - Puedes usar create_Gaussian_kernel_1D() para completar esta función. 
    - La idea es que el resultado el producto de un vector columna (mx1) por un vector fila (1xm) 
      retorne una matriz de mxm. 
    - La otra alternativa es directamente usar la fórmula de la función de distribución de densidad Gaussiana multivariante. Evalúan la función para cada posición del filtro cuyo centro es (0,0)
    - La idea nuevamente es distretizar los valores gaussianos en una matriz.
    """

    ############################
    ### TODO: EL CÓDIGO EMPIEZA ACÁ ###

    raise NotImplementedError(
        "La función `create_Gaussian_kernel_2D` necesita ser implementada en `part1.py`"
    )

    ### EL CÓDIGO TERMINA ACÁ ####
    ############################

    return kernel


def my_conv2d_numpy(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """ Aplicar un filtro 2d a cada canal de una imagen. Retornar la imagen filtrada.
    Parámetros:
        image: arreglo con shape (m, n, c)
        filter: arreglo con shape (k, j)
    Retorna:
        filtered_image: arreglo con shape (m, n, c). El tamaño de la imagen debe ser preservada.

    Notas:
    - You may not use any libraries that do the work for you. Using numpy to
      work with matrices is fine and encouraged. Using OpenCV or similar to do
      the filtering for you is not allowed.
    - No es válido usar librerías que ya hagan la convolución o apliquen filtros. 
      Usar numpy para manejar matrices y realizar operaciones con las mismas es lo que se recomienda. 
      Usar opencv por ejemplo con su método de filtrado no va a ser considerado válido.
    - Primero pueden probar implementando esto de manera naive usando bucles quizás. Esto puede llevar
      un buen tiempo de ejecución. La función que implementes debe tener un máximo de tiempo para
      retornar el resultado. Más de 5 minutos por ej es demasiado.
    - Para el padding de la imagen, en este caso vamos a usar solamente zero-padding. Manualmente se tiene que especificar cuanto padding es requerido.
    - "Stride" debe setearse a 1.
    - Puedes implementar "cross-correlation" o "convolución" y el resultado va a ser el mismo porque solamente vamos a usar filtros simétricos.
    """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1


    ############################
    ### TODO: EL CÓDIGO EMPIEZA ACÁ ###

    raise NotImplementedError(
        "La función `my_conv2d_numpy` debe ser implementada en `part1.py`"
    )

    ### EL CÓDIGO TERMINA ACÁ ####
    ############################


    return filtered_image


def create_hybrid_image(
    image1: np.ndarray, image2: np.ndarray, filter: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes two images and a low-pass filter and creates a hybrid image. Returns
    the low frequency content of image1, the high frequency content of image 2,
    and the hybrid image.
    Dadas 2 imágenes y un filtro low-pass, crea una imagen híbrida. 
    Retornar las frecuencias bajas de la imagen1 y las frecuencias altas de la imagen 2, 
    y la imágen híbrida.

    Parámetros:
        image1: array de dimensión (m, n, c)
        image2: array de dimensión (m, n, c)
        filter: array de dimensión (x, y)
    Returnar:
        low_frequencies: array de shape (m, n, c)
        high_frequencies: array de shape (m, n, c)
        hybrid_image: array de shape (m, n, c)

    TIPS:
    - Se hace uso de my_conv2d_numpy.
    - Se extrae las altas frecuencias de una imagen restándole sus bajas frecuencias. Piensa como 
      puedes hacer esto en términos matemáticos y de filtros.
    - No olvides que los valores de los pixeles deben estar entre 0  y 1. Esto se conoce como 'clipping'. Numpy ofrece una función para hacer clipping, es básicamente lo que vimos en clase. 
      Todo valor que sobrepase un threshold máximo o mínimo es cambiado por dicho valor máximo o mínimo.
    - Las imágenes tienen que tener la misma dimensión. Si quieres usar imágenes de dimensiones distintas tienes que redimensionarlas primero en la notebook.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]
    assert filter.shape[0] <= image1.shape[0]
    assert filter.shape[1] <= image1.shape[1]
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    ############################
    ### TODO: EL CÓDIGO EMPIEZA ACÁ ###

    raise NotImplementedError(
        "La función `create_hybrid_image` debe ser implementada en `part1.py`"
    )

    ### EL CÓDIGO TERMINA ACÁ ####
    ############################

    return low_frequencies, high_frequencies, hybrid_image
