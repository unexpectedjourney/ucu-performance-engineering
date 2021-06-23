all:
	make build
	make run

build:
	#gcc -o main ./main.cpp -I /home/antonbabenko/opencv/include -L/home/antonbabenko/opencv/include -lopencv
	g++ -O3 -g3 -std=c++17 -o main ./main.cpp -lpthread -pthread

run:
	./main

clean:
	rm -rf ./main