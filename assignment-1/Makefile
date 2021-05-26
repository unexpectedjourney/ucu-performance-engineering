all:
	make build
	make run

build:
	#gcc -O3 -march=native -g3 -ftree-vectorize -fopt-info-vec -fopt-info-vec-missed -fno-omit-frame-pointer -lstdc++ -L /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so -l openblas -lpthread -o main ./main.cpp
	gcc -march=native -O1 -o main ./main.cpp
	#gcc -O3 -g3 -ftree-vectorize -fopt-info-vec -fopt-info-vec-missed -fno-omit-frame-pointer -lstdc++ -I /usr/lib/x86_64-linux-gnu/openblas-pthread/ -L/usr/lib/x86_64-linux-gnu/openblas-pthread/ -lopenblas -lpthread -o task2 ./task2.c
	#gcc -O3 -march=native -g3 -ftree-vectorize -fopt-info-vec -fopt-info-vec-missed -fno-omit-frame-pointer -lstdc++ -L /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so -l openblas -lpthread -o task2 ./task2.c

run:
	./main

clean:
	rm -rf ./main
