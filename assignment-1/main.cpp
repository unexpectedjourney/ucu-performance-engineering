#include <cstdlib>
#include <cstdio>
#include <sys/times.h>
#include <ctime>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <cblas-openblas.h>

#include <string>
#include <cstring>

#define LEN 33000
#define MLEN 50
#define STRLEN 901
#define SUBSTRLEN 3
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

char * first_str;
char * second_str;

const char charset[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

int nothing_1(float _a[LEN], float _b[LEN], float _c[LEN], float _d[LEN], float _sum[LEN]) {
  return (0);
}

int nothing_2(double _m_a[MLEN][MLEN], double _m_b[MLEN][MLEN], double _result[MLEN][MLEN]) {
  return (0);
}

int nothing_3(char * _first_str, char * _second_str) {
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


static char *random_string(char *str, int size) {
  str = (char*)malloc(size + 1);
  if (size > 0) {
    --size;
    for (size_t n = 0; n < size; n++) {
      int key = rand() % (int) (sizeof charset - 1);
      str[n] = charset[key];
    }
    str[size] = '\0';
  }
  return str;
}


void init_strings() {
  first_str = random_string(first_str, STRLEN);
  second_str = random_string(second_str, SUBSTRLEN);
}

void basic_substing_find() {
  for (int i = 0; i < strlen(first_str); ++i) {
    bool is_found = true;
    if (first_str[i] != second_str[0]) {
      continue;
    }
    for (int j = 1; j < strlen(second_str); ++j) {
      if (first_str[i + j] != second_str[j]) {
        is_found = false;
        break;
      }
    }
    if (is_found) {
      break;
    }
  }
  nothing_3(first_str, second_str);
}

void vectorized_substing_find() {
  int second_len = strlen(second_str);

  const __m128i first_letter = _mm_set1_epi8(second_str[0]);
  const __m128i last_letter = _mm_set1_epi8(second_str[second_len - 1]);

  for (int i = 0; i < STRLEN; i += 2) {
    bool is_found = false;

    const __m128i block_first_letter = _mm_load_si128(
        reinterpret_cast<const __m128i*>(&first_str[i]));
    const __m128i block_last_letter = _mm_load_si128(
        reinterpret_cast<const __m128i*>(&first_str[i + second_len - 1]));

    const __m128i first_equality = _mm_cmpeq_epi8(first_letter, block_first_letter);
    const __m128i second_equality = _mm_cmpeq_epi8(last_letter, block_last_letter);

    int mask = _mm_movemask_epi8(_mm_and_si128(first_equality, second_equality));

    while (mask != 0) {
      int bitpos = __builtin_ctzl(mask);
      if (memcmp(&first_str[i + bitpos + 1], &second_str[1], second_len - 2) == 0) {
        is_found = true;
        break;
      }
      mask &= (mask - 1);
    }
    if (is_found) {
      break;
    }
  }
  nothing_3(first_str, second_str);
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

  // Task 3
  init_strings();
  count_time(&basic_substing_find, "Task 3 basic");
  count_time(&vectorized_substing_find, "Task 3 vectorized");
  return 0;
}
