# Assignment-1
This assigment contains the solution of 3 tasks to apply vectorization.

## Dependencies
This application uses external library __openblas-pthread__, so please provide the proper path in __MAKEFILE__ to it.

## Task
Реалізувати із використанням векторизації (Intrinsics https://software.intel.com/sites/landingpage/IntrinsicsGuide/) задачі:
1) A*B + C*D, де A,B,C,D - вектори однакової довжини (масиви).
2) Множення матриць. Для спрощення обрати квадратны матрицы, яки можуть бути розміщені в пам'яті. Порівняти швидкість вашої реалізації із швидкістю BLAS (наприклад: https://www.konda.eu/c-openblas-matrix-multiplication/)
3) Пошук підрядка в рядку. Знайти перше співпадіння. str1 = 'XYZAVKLRPZA ', str2 = 'ZA', FirstOccurence = 2.
   Оцінювання:
1) 30 балів
2) 30 балів
3) 40 балів

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
Task 1 basic:	     2.66 	
Task 1 vectorized:	 0.90 	
Task 2 basic:	     7.16 	
Task 2 cblas:	     0.74 	
Task 2 vectorized:	 3.44 	
Task 3 basic:	     0.11 	
Task 3 vectorized:	 0.02
```
Hence, as you can see, vectorization speeds up the application execution.