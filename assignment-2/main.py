import ctypes

import cv2
import requests
import numpy as np
from numpy.ctypeslib import ndpointer

from utils import timeit

IMAGE_URL = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fu.kanobu.ru%2Farticles%2Fpics%2F9bef0327-7917-4c08-872a-e550e7da9d68.jpg&f=1&nofb=1"
SO_FILE = "./main.so"
MAINLIB = ctypes.CDLL(SO_FILE)

def get_photo(url):
    req = requests.get(url)
    arr = np.asarray(bytearray(req.content), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img

@timeit
def simple_sum_of_pixels(array):
    return array.sum()

@timeit
def cuda_sum_of_pixels(array):
    MAINLIB.task1.restype = ctypes.c_float
    MAINLIB.task1.argtypes = [ndpointer(dtype=np.float32), ctypes.c_int]
    array_sum = MAINLIB.task1(array.astype(np.float32), array.shape[0])
    return array_sum

def task1(image):
    simple_sum_of_pixels(image[0, :, :].flatten())
    cuda_sum_of_pixels(image[0, :, :].flatten())

@timeit
def simple_min_of_pixels(array):
    return np.min(array)

@timeit
def cuda_min_of_pixels(array):
    MAINLIB.task2.restype = ctypes.c_float
    MAINLIB.task2.argtypes = [ndpointer(dtype=np.float32), ctypes.c_int]
    array_sum = MAINLIB.task2(array.astype(np.float32), array.shape[0])
    return array_sum

def task2(image):
    simple_min_of_pixels(image[0, :, :].flatten())
    cuda_min_of_pixels(image[0, :, :].flatten())

def task3(image):
    pass

def main():
    image = get_photo(IMAGE_URL).T
    print(image.shape)
    task1(image)
    task2(image)

if __name__ == "__main__":
    main()