all:
	make build
	make run

build:
	nvcc -o main.o main.cu
	nvcc --shared -o main.so main.cu --compiler-options '-fPIC'

run:
	./main.o
	python3 main.py

clean:
	rm -rf ./main.so ./main.o