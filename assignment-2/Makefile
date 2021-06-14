all:
	make build
	make run

build:
	nvcc -o main.o main.cu
	#nvcc --ptxas-options=-v --compiler-options '-fPIC' -o main.so --shared main.cu
	nvcc --shared -o main.so main.cu --compiler-options '-fPIC'

run:
	./main.o

clean:
	rm -rf ./main.so ./main.o