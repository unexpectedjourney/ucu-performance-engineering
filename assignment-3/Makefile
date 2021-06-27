OPENCV = `pkg-config --cflags --libs opencv4`
LIBS = $(OPENCV)

all:
	make build
	make run

build:
	g++ $(LIBS) -O3 -g3 -std=c++17 -o main ./main.cpp -lpthread -pthread

run:
	./main

clean:
	rm -rf ./main