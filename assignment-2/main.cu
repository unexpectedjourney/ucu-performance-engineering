#include <stdio.h>

const int BLOCK_SIZE = 32;

extern "C" float task1(float * array, int n);

extern "C" float task2(float * array, int n);

extern "C" void task3(float * array, float * result_array, float * filter, int n, int m);

__global__ void fill_1_block(float *array) {
  int idx = threadIdx.x;
  array[idx] = 1;
}

__global__ void fill_0_block(float *array) {
  int idx = threadIdx.x;
  array[idx] = 0;
}

float * fill_random_block(float *array, int n) {
  for (int i = 0; i < n; ++i) {
    array[i] = (float)rand()/(float)(RAND_MAX/1024);
  }
  return array;
}

float * generate_filter(float *array, int n) {
  for (int i = 0; i < n; ++i) {
    array[i] = (i % 2);
  }
  return array;
}

__global__ void sum_array(float * array, float * result_array, int n) {
  int idx = threadIdx.x;
  if (idx < n) {
    atomicAdd(result_array, array[idx]);
  }
}


float task1(float * array, int n) {
  float * device_array, * device_result_array;
  float * result_array = (float*)malloc(n*sizeof(float));

  cudaMalloc(&device_array, n*sizeof(float));
  cudaMalloc(&device_result_array, n*sizeof(float));

  cudaMemcpy(device_array, array, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_result_array, result_array, n*sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 1024;
  int gridSize = (int)ceil((float)n/blockSize);

  fill_0_block<<<gridSize, blockSize>>>(device_result_array);

  sum_array<<<gridSize, blockSize>>>(device_array, device_result_array, n);
  cudaMemcpy(result_array, device_result_array, n*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(device_array);
  cudaFree(device_result_array);

  float result = result_array[0];
  free(result_array);

  return result;
}

void execute_task1() {
  int n = 1 << 20;
  printf("%d\n", n);
  float * array = (float*)malloc(n*sizeof(float));
  array = fill_random_block(array, n);
  float result = task1(array, n);
  printf("Sum of the array: %f\n", result);

  free(array);
}

__global__ void get_min_array(float *array, float *min_results) {
  extern __shared__ float mintile[BLOCK_SIZE];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  mintile[tid] = array[i];
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (mintile[tid + s] < mintile[tid]) {
        mintile[tid] = mintile[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    min_results[blockIdx.x] = mintile[0];
  }
}

__global__ void get_final_min_array(float * min_results) {
  __shared__ float mintile[BLOCK_SIZE];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  mintile[tid] = min_results[i];
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (mintile[tid + s] < mintile[tid]) {
        mintile[tid] = mintile[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    min_results[blockIdx.x] = mintile[0];
  }
}

float task2(float * array, int n) {
  float * device_array, * device_result_array;
  float * result_array = (float*)malloc(n*sizeof(float));

  cudaMalloc(&device_array, n*sizeof(float));
  cudaMalloc(&device_result_array, n*sizeof(float));

  cudaMemcpy(device_array, array, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_result_array, result_array, n*sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = BLOCK_SIZE;
  int gridSize = (int)ceil((float)n/blockSize);

  fill_0_block<<<gridSize, blockSize>>>(device_result_array);

  get_min_array<<<gridSize, blockSize>>>(device_array, device_result_array);
  get_final_min_array<<<1, blockSize>>>(device_result_array);

  cudaMemcpy(result_array, device_result_array, n*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(device_array);
  cudaFree(device_result_array);
  float min_value = result_array[0];
  free(result_array);
  return min_value;
}

void execute_task2() {
  int n = 1 << 10;
  printf("%d\n", n);
  float * array = (float*)malloc(n*sizeof(float));
  array = fill_random_block(array, n);
  float result = task2(array, n);
  printf("Min of the array: %f\n", result);
  free(array);
}

__global__ void convolute(float * array, float * filter, float * result_array, int array_size, int filter_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float value = 0;

  int start_point =  idx - (filter_size / 2);
  for (int i = 0; i < filter_size; ++i) {
    int current_position = start_point + i;
    if (current_position < 0 || current_position >= array_size) {
      continue;
    }
    value = value + array[current_position] * filter[i];
  }
  result_array[idx] = value;
}

void task3(float * array, float * result_array, float * filter, int n, int m) {
  float * device_array, * device_result_array, * device_filter;

  cudaMalloc(&device_array, n*sizeof(float));
  cudaMalloc(&device_result_array, n*sizeof(float));
  cudaMalloc(&device_filter, m*sizeof(float));

  cudaMemcpy(device_array, array, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_result_array, result_array, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_filter, filter, m*sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = BLOCK_SIZE;
  int gridSize = (int)ceil((float)n/blockSize);

  fill_0_block<<<gridSize, blockSize>>>(device_result_array);

  convolute<<<gridSize, blockSize>>>(device_array, device_filter, device_result_array, n, m);

  cudaMemcpy(result_array, device_result_array, n*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(device_array);
  cudaFree(device_result_array);
  cudaFree(device_filter);
}

void execute_task3() {
  int n = 1 << 10;
  int m = 32;
  printf("%d\n", n);
  float * array = (float*)malloc(n*sizeof(float));
  float * result_array = (float*)malloc(n*sizeof(float));
  float * filter = (float*)malloc(m*sizeof(float));

  array = fill_random_block(array, n);
  filter = generate_filter(filter, m);

  task3(array, result_array, filter, n, m);

  free(array);
  free(result_array);
  free(filter);
}

int main() {
  execute_task1();
  execute_task2();
  execute_task3();
  return 0;
}
