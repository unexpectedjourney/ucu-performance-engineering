# Assignment-3
This assigment contains the solution of 3 tasks to apply multithreading.

## Dependencies
This application uses external library __opencv2__ and __pkg-config__, so please install it to build this application.

## Task
Реалізувати із використанням Multithreading on CPU і порівняти швидкодію із single thread для такої задачі:

1) Сума всіх пікселів у зображенні по одному з каналів RGB кольорового зображення.
2) Пошук мінімального значення в одному з каналів RGB кольорового зображення.
3) Розрахунок  згортки (конволюції) для кожного з RGB каналів зображення.
   Кількість згорток, параметри ядра кожної з згорток задаються один раз в  коді програми для всіх зображень що оброблюються.
   Вітається, але не обов'язково застосовувати результати згортки за допомогою OpenCV в якості тестів.

Оцінювання:
1) до 30 балів
2) до 30 балів
3) до 40 балів

## Build and execute
To build and execute this project, use:
```bash
make all
```
To just build this project, use:
```bash
make build
```
To execute this project, use:
```bash
make run
```

## Results
```
Task1:
Simple results:	         0.01 	
Result:                  178266820
Multithread results:	 2.60 	
Result:                  178266820
Task2:
Simple results:	         0.01 	
Result:                  0
Multithread results:	 0.03 	
Result:                  0
Task3:
Simple results:	         9.26 	
Multithread results:	 3.76 
```
Hence, as you can see, multithreading speeds up the application execution on complex tasks.