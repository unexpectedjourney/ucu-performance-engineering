#include <iostream>
#include <vector>
#include <cstdlib>
#include <thread>
#include <atomic>

#define N 1000000

const float sec_const = 1000000.0;

void count_time(void (*f)(), char *func_name) {
  clock_t start_t = clock();
  f();
  clock_t end_t = clock();
  clock_t clock_delta = end_t - start_t;
  double clock_delta_sec = (double) (clock_delta / sec_const);
  printf("%s:\t %.2f \t\n", func_name, clock_delta_sec);
}

void random_array(std::vector<int> &array) {
  for (int i = 0; i < array.size(); ++i) {
    array[i] = rand() % 100;
  }
}

int basic_sum(std::vector<int> &array) {
  int result = 0;
  for (int i = 0; i < array.size(); ++i) {
    result += array[i];
  }
  return result;
}

void sum_slice(std::atomic<int> &result, std::vector<int> &array, int start_pos, int end_pos) {
  for (int i = start_pos; i < end_pos; ++i) {
    result.fetch_add(array[i]);
  }
}

void multithread_sum(std::vector<int> &array, std::atomic<int> &result) {
  int thread_amount = 16;
  int block_size = N / thread_amount;

  std::vector<std::thread> threads;
  for (int i = 0; i < thread_amount; ++i) {
    int start_pos = i * block_size;
    int end_pos = (i + 1) * block_size;
    threads.push_back(std::thread(sum_slice, std::ref(result), std::ref(array), start_pos, end_pos));
  }
  for (int i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}

void task1() {
  std::vector<int> array(N, 0);
  random_array(array);
  int simple_result = basic_sum(array);
  printf("%d\n", simple_result);

  std::atomic<int> m_result{0};
  multithread_sum(array, m_result);
  printf("%d\n", int(m_result));
}

int basic_min(std::vector<int> &array) {
  int min_value = 1e9;
  for (int i = 0; i < array.size(); ++i) {
    if (min_value > array[i]) {
      min_value = array[i];
    }
  }
  return min_value;
}

void min_slice(std::atomic<int> &result, std::vector<int> &array, int start_pos, int end_pos) {
  for (int i = start_pos; i < end_pos; ++i) {
    if (int(result) > array[i]) {
      result.store(array[i]);
    }
  }
}

void multithread_min(std::vector<int> &array, std::atomic<int> &result) {
  int thread_amount = 16;
  int block_size = N / thread_amount;

  std::vector<std::thread> threads;
  for (int i = 0; i < thread_amount; ++i) {
    int start_pos = i * block_size;
    int end_pos = (i + 1) * block_size;
    threads.push_back(std::thread(min_slice, std::ref(result), std::ref(array), start_pos, end_pos));
  }
  for (int i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}

void task2() {
  std::vector<int> array(N, 0);
  random_array(array);
  int simple_result = basic_min(array);
  printf("%d\n", simple_result);

  std::atomic<int> m_result{0};
  multithread_min(array, m_result);
  printf("%d\n", int(m_result));
}

void generate_filter(std::vector<int> &array) {
  for (int i = 0; i < array.size(); ++i) {
    array[i] = i % 2;
  }
}

void basic_conv(std::vector<int> &array, std::vector<int> filter, std::vector<int> &result_array) {
  for (int i = 0; i < result_array.size(); ++i) {
    int value = 0;
    int start_point =  i - (filter.size() / 2);
    for (int j = 0; j < filter.size(); ++j) {
      int current_position = start_point + j;
      if (current_position < 0 || current_position >= result_array.size()) {
        continue;
      }
      value = value + array[current_position] * filter[j];
    }
    result_array[i] = value;
  }
}

void conv_slice(std::vector<int> &array, std::vector<int> &filter, std::vector<int> &result_array, int start_pos, int end_pos) {
  for (int i = start_pos; i < end_pos; ++i) {
    int value = 0;
    int start_point =  i - (filter.size() / 2);
    for (int j = 0; j < filter.size(); ++j) {
      int current_position = start_point + j;
      if (current_position < 0 || current_position >= result_array.size()) {
        continue;
      }
      value = value + array[current_position] * filter[j];
    }
    result_array[i] = value;
  }
}

void multithread_conv(std::vector<int> &array, std::vector<int> &filter, std::vector<int> &result_array) {
  int thread_amount = 16;
  int block_size = N / thread_amount;

  std::vector<std::thread> threads;
  for (int i = 0; i < thread_amount; ++i) {
    int start_pos = i * block_size;
    int end_pos = (i + 1) * block_size;
    threads.push_back(std::thread(conv_slice, std::ref(array), std::ref(filter), std::ref(result_array), start_pos, end_pos));
  }
  for (int i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}

void task3() {
  std::vector<int> array(N, 0);
  random_array(array);
  std::vector<int> filter(32, 0);
  generate_filter(filter);
  std::vector<int> result_array(N, 0);

  basic_conv(array, filter, result_array);
  for (int i = 0; i < result_array.size(); ++i) {
    printf("%d\t", result_array[i]);
  }

  std::vector<int> m_result_array(N, 0);
  multithread_conv(array, filter, m_result_array);
  for (int i = 0; i < m_result_array.size(); ++i) {
    printf("%d\t", m_result_array[i]);
  }
}


int main() {
  task1();
  task2();
  task3();
  return 0;
}
