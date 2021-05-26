#include <cstdlib>
#include <cstdio>
#include <sys/times.h>
#include <ctime>
#include <xmmintrin.h>

#define LEN 33000
#define NTIMES 100000

const float sec_const = 1000000.0;

float a[LEN] __attribute__((aligned(16)));
float b[LEN] __attribute__((aligned(16)));
float c[LEN] __attribute__((aligned(16)));
float d[LEN] __attribute__((aligned(16)));

float sum[LEN] __attribute__((aligned(16)));

int nothing_1(float _a[LEN], float _b[LEN], float _c[LEN], float _d[LEN], float _sum[LEN]){
  return (0);
}

void count_time(void (*f)(), char *func_name) {
  clock_t start_t = clock();
  for (int n = 0; n < NTIMES; ++n) {
    f();
  }
  clock_t end_t = clock();
  clock_t clock_delta = end_t - start_t;
  double clock_delta_sec = (double) (clock_delta / sec_const);
  printf("%s:\t %.2f \t\n", func_name, clock_delta_sec);
}

void init_vectors() {
  for (int i = 0; i < LEN; ++i) {
    a[i] = (rand() % 50) / 50;
    b[i] = (rand() % 50) / 50;
    c[i] = (rand() % 50) / 50;
    d[i] = (rand() % 50) / 50;
  }
}

void basic_sum() {
    for (int i = 0; i < LEN; ++i) {
      sum[i] = a[i] * b[i] + c[i] * d[i];
    }
    nothing_1(a, b, c, d, sum);
}


void vectorized_sum() {
  __m128 rA, rB, rC, rD, rab, rcd, rabcd;
  for (int i = 0; i < LEN; i += 4) {
    rA = _mm_load_ps(&a[i]);
    rB = _mm_load_ps(&b[i]);
    rC = _mm_load_ps(&c[i]);
    rD = _mm_load_ps(&d[i]);

    rab = _mm_mul_ps(rA, rB);
    rcd = _mm_mul_ps(rC, rD);

    rabcd = _mm_add_ps(rab, rcd);

    _mm_store_ps(&sum[i], rabcd);
  }
  nothing_1(a, b, c, d, sum);
}


int main() {
  // Task 1
  init_vectors();
  count_time(&basic_sum, "Task 1 basic");
  count_time(&vectorized_sum, "Task 1 vectorized");
  return 0;
}
