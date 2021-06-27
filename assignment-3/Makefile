OPENCV = `pkg-config --cflags --libs opencv4`
LIBS = $(OPENCV)

all:
	make build
	make run

build:
	g++ $(LIBS) -O3 -g3 -std=c++17 -o main ./main.cpp -lpthread -pthread
#-I/usr/local/Cellar/opencv/4.5.2_4/include/opencv4
#-L/usr/local/Cellar/opencv/4.5.2_4/lib/

run:
	./main

clean:
	rm -rf ./main