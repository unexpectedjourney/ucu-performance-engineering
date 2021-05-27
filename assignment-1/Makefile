all:
	make build
	make run

build:
	gcc -march=native -O1 -o main ./main.cpp -I /usr/lib/x86_64-linux-gnu/openblas-pthread/ -L/usr/lib/x86_64-linux-gnu/openblas-pthread/ -lopenblas -lpthread

run:
	./main

clean:
	rm -rf ./main
