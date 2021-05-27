#include <cstdlib>
#include <cstdio>
#include <sys/times.h>
#include <ctime>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <cblas-openblas.h>

#define LEN 33000
#define MLEN 50
#define NTIMES 100000

const float sec_const = 1000000.0;

float a[LEN] __attribute__((aligned(16)));
float b[LEN] __attribute__((aligned(16)));
float c[LEN] __attribute__((aligned(16)));
float d[LEN] __attribute__((aligned(16)));
float sum[LEN] __attribute__((aligned(16)));

double m_a[MLEN][MLEN] __attribute__((aligned(16)));
double m_b[MLEN][MLEN] __attribute__((aligned(16)));
double result_vector[MLEN] __attribute__((aligned(16)));
double result[MLEN][MLEN] __attribute__((aligned(16)));

int nothing_1(float _a[LEN], float _b[LEN], float _c[LEN], float _d[LEN], float _sum[LEN]) {
  return (0);
}

int nothing_2(double _m_a[MLEN][MLEN], double _m_b[MLEN][MLEN], double _result[MLEN][MLEN]) {
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
    a[i] = (float)rand()/(float)(RAND_MAX) * 10;
    b[i] = (float)rand()/(float)(RAND_MAX) * 10;
    c[i] = (float)rand()/(float)(RAND_MAX) * 10;
    d[i] = (float)rand()/(float)(RAND_MAX) * 10;
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

void init_matrix() {
  for (int i = 0; i < MLEN; ++i) {
    for (int j = 0; j < MLEN; ++j) {
      m_a[i][j] = (double)rand()/(double)(RAND_MAX) * 10;
      m_b[i][j] = (double)rand()/(double)(RAND_MAX) * 10;
      result[i][j] = 0.;
    }
  }
}

void transpose_b() {
  for (int i = 0; i < MLEN; ++i) {
    for (int j = i; j < MLEN; ++j) {
      double temp = m_b[i][j];
      m_b[i][j] = m_b[j][i];
      m_b[j][i] = temp;
    }
  }
}

void basic_multiplication() {
  for (int i = 0; i < MLEN; ++i) {
    for (int j = 0; j < MLEN; ++j) {
      result[i][j] = 0;
      for (int k = 0; k < MLEN; ++k) {
        result[i][j] += (m_a[i][k] * m_a[k][j]);
      }
    }
  }
  nothing_2(m_a, m_b, result);
}

void vectorized_multiplication() {
  for (int i = 0; i < MLEN; ++i) {
    for (int j = 0; j < MLEN; ++j) {
      __m128d rA, rB, rab;
      __m128d rR = _mm_setzero_pd();

      for (int k = 0; k < MLEN; k += 2) {
        rA = _mm_load_pd(&m_a[i][k]);
        rB = _mm_load_pd(&m_b[j][k]);
        rab = _mm_mul_pd(rA, rB);
        rR = _mm_add_pd(rab, rR);
      }

      rR = _mm_hadd_pd(rR, rR);
      rR = _mm_hadd_pd(rR, rR);
      rR = _mm_hadd_pd(rR, rR);
      result[i][j] = rR[0];
    }
  }
  nothing_2(m_a, m_b, result);
}

void cblas_multiplication() {
  cblas_dgemm(
      CblasRowMajor,
      CblasNoTrans, CblasTrans,
      MLEN, MLEN, MLEN, 1,
      &m_a[0][0], MLEN,
      &m_a[0][0], MLEN,
      MLEN,
      &result[0][0], MLEN);
}

int main() {
  // Task 1
  init_vectors();
  count_time(&basic_sum, "Task 1 basic");
  count_time(&vectorized_sum, "Task 1 vectorized");

  // Task 2
  init_matrix();
  count_time(&basic_multiplication, "Task 2 basic");
  transpose_b();
  count_time(&cblas_multiplication, "Task 2 cblas");
  count_time(&vectorized_multiplication, "Task 2 vectorized");
  return 0;
}
