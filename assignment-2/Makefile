all:
	make build
	make run

build:
	nvcc -o main.so main.cu

run:
	./main.so

clean:
	rm -rf ./main.so