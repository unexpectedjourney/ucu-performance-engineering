# Assignment-2
This assigment contains the solution of 5 tasks(3 main and 2 for integration) to apply cuda kernels.

## Dependencies
This application uses __nvcc__ and __python3__, so please install these dependencies before building this project.
Also, you can install some python packages via command:
```python3
pip install -r requirements.txt
```

## Task
Реалізувати із використанням NVIDIA CUDA та CPU і порівняти швидкодію для такої задачі:
1) Сума всіх пікселів у зображенні по одному з каналів RGB кольорового зображення.
2) Пошук мінімального значення в одному з каналів RGB кольорового зображення із застосуванням підходу Reduction.
3) Розрахунок  згортки (конволюції) для кожного з RGB каналів зображення.
   Кількість згорток, параметри ядра кожної з згорткок задаються один раз в
   коді програми для всіх зображень що оброблюються.
   Матеріали стосовно конволюції:
   https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
   https://www.sciencedirect.com/topics/computer-science/convolution-filter
   Вітається, але не обов'язково застосовувати результати згортки за допомогою OpenCV
   в якості тестів.
   Можна реалізовувати високорівневими засобами Python + CUDA.     
   
Оцінювання:  
   1)Реалізація завдання №1 -  до 20 балів
   2)Реалізація завдання №2 -  до 20 балів  
   3)Реалізація завдання №3 -  до 35 балів  
   4)Виконання роботи на низькому рівні (С/C++ CUDA library) - до 15 балів  
   5)Інтеграція рішення на С/C++ у вигляді бібліотечних викликів з Python - до 10 балів.  

## Build and execute
To build and execute this project, use:
```bash
make all
```
To just build this project, use:
```bash
make build
```
It will create two binary data:
- main.o - executable file of the kernels
- main.so - dynamic library, which is used in the python code.  

To execute this project, use:
```bash
make run
```
It will execute cuda kernels directly and then via python3.
## Results
```
Task1
simple_sum_of_pixels: 0.1854095458984375
cuda_sum_of_pixels: 0.0859982967376709
Task2
simple_min_of_pixels: 0.10199856758117676
cuda_min_of_pixels: 0.0043294429779052734
Task3
simple_convolution_of_pixels: 154.74093866348267
cuda_convolution_of_pixels: 0.023753881454467773
```
Hence, as you can see, cuda kernels speed up the application execution.