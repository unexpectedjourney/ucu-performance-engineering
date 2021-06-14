import ctypes

import cv2
import requests
import numpy as np
from numpy.ctypeslib import ndpointer

from utils import timeit

IMAGE_URL = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fu.kanobu.ru%2Farticles%2Fpics%2F9bef0327-7917-4c08-872a-e550e7da9d68.jpg&f=1&nofb=1"
FILTER_WIDTH = 32
SO_FILE = "./main.so"
MAINLIB = ctypes.CDLL(SO_FILE)


def get_photo(url):
    req = requests.get(url)
    arr = np.asarray(bytearray(req.content), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img


@timeit
def simple_sum_of_pixels(array):
    sum_value = 0
    for i in range(array.shape[0]):
        sum_value += array[i]
    return sum_value


@timeit
def cuda_sum_of_pixels(array):
    MAINLIB.task1.restype = ctypes.c_float
    MAINLIB.task1.argtypes = [ndpointer(dtype=np.float32), ctypes.c_int]
    array_sum = MAINLIB.task1(array.astype(np.float32), array.shape[0])
    return array_sum


def task1(image):
    print("Task1")
    simple_sum_of_pixels(image[0, :, :].flatten())
    cuda_sum_of_pixels(image[0, :, :].flatten())


@timeit
def simple_min_of_pixels(array):
    min_value = 1e9
    for i in range(array.shape[0]):
        if array[i] < min_value:
            min_value = array[i]
    return min_value


@timeit
def cuda_min_of_pixels(array):
    MAINLIB.task2.restype = ctypes.c_float
    MAINLIB.task2.argtypes = [ndpointer(dtype=np.float32), ctypes.c_int]
    array_sum = MAINLIB.task2(array.astype(np.float32), array.shape[0])
    return array_sum


def task2(image):
    print("Task2")
    simple_min_of_pixels(image[0, :, :].flatten())
    cuda_min_of_pixels(image[0, :, :].flatten())


@timeit
def simple_convolution_of_pixels(image, image_filter):
    for i in range(3):
        array = image[i, :, :].flatten()
        result_array = []
        for j in range(array.shape[0]):
            value = 0
            start_point = j - FILTER_WIDTH // 2
            for k in range(FILTER_WIDTH):
                current_position = start_point + k
                if current_position < 0 or current_position >= array.shape[0]:
                    continue
                value = value + array[current_position] * image_filter[k]
            result_array.append(value)


@timeit
def cuda_convolution_of_pixels(image, image_filter):
    for i in range(3):
        MAINLIB.task3.restype = ctypes.c_float
        MAINLIB.task3.argtypes = [
            ndpointer(dtype=np.float32),
            ndpointer(dtype=np.float32),
            ndpointer(dtype=np.float32),
            ctypes.c_int,
            ctypes.c_int
        ]
        array = image[i, :, :].flatten().astype(np.float32)
        result_array = np.full_like(array, 0)
        MAINLIB.task3(
            array,
            result_array,
            image_filter.astype(np.float32),
            array.shape[0],
            image_filter.shape[0]
        )


def generate_filter(size):
    return np.array([i % 2 for i in range(size)])


def task3(image):
    print("Task3")
    m = 32
    image_filter = generate_filter(m)
    simple_convolution_of_pixels(image, image_filter)
    cuda_convolution_of_pixels(image, image_filter)


def main():
    image = get_photo(IMAGE_URL).T
    task1(image)
    task2(image)
    task3(image)


if __name__ == "__main__":
    main()
