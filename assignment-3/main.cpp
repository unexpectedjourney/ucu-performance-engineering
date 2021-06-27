#include <iostream>
#include <vector>
#include <cstdlib>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>

#define TIMES 100

void random_array(std::vector<int> &array) {
  for (int i = 0; i < array.size(); ++i) {
    array[i] = rand() % 100;
  }
}

cv::Mat load_image() {
  cv::Mat image = cv::imread("./image.jpeg");
  if (image.empty()) {
    std::cout << "322";
  }
  return image;
}

std::vector<int> get_channel_from_mat(cv::Mat &image, int channel) {
  std::vector<int> array{0};
  for (int i = 0; i < image.cols; ++i) {
    for (int j = 0; j < image.rows; ++j) {
      cv::Vec3b intensity = image.at<cv::Vec3b>(j, i);
      uchar col = intensity.val[channel];
      array.push_back(col);
    }
  }
  return array;
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
  int block_size = array.size() / thread_amount;

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
  printf("Task1:\n");
  cv::Mat image = load_image();
  std::vector<int> array = get_channel_from_mat(image, 0);
  std::chrono::time_point<std::chrono::system_clock> start_t = std::chrono::system_clock::now();
  int simple_result = 0;
  for (int t = 0; t < TIMES; ++t) {
    simple_result = basic_sum(array);
  }
  std::chrono::time_point<std::chrono::system_clock> end_t = std::chrono::system_clock::now();
  std::chrono::duration<double> clock_delta = end_t - start_t;

  printf("Simple results:\t %.2f \t\n", clock_delta.count());
  printf("Result: %d\n", simple_result);

  std::atomic<int> m_result{0};
  start_t = std::chrono::system_clock::now();
  for (int t = 0; t < TIMES; ++t) {
    m_result.store(0);
    multithread_sum(array, m_result);
  }
  end_t = std::chrono::system_clock::now();
  clock_delta = end_t - start_t;

  printf("Multithread results:\t %.2f \t\n", clock_delta.count());
  printf("Result: %d\n", int(m_result));
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
  int block_size = array.size() / thread_amount;

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
  printf("Task2:\n");
  cv::Mat image = load_image();
  std::vector<int> array = get_channel_from_mat(image, 0);
  std::chrono::time_point<std::chrono::system_clock> start_t = std::chrono::system_clock::now();
  int simple_result = 0;
  for (int t = 0; t < TIMES; ++t) {
    simple_result = basic_min(array);
  }
  std::chrono::time_point<std::chrono::system_clock> end_t = std::chrono::system_clock::now();
  std::chrono::duration<double> clock_delta = end_t - start_t;

  printf("Simple results:\t %.2f \t\n", clock_delta.count());
  printf("Result: %d\n", simple_result);

  std::atomic<int> m_result{0};
  start_t = std::chrono::system_clock::now();
  for (int t = 0; t < TIMES; ++t) {
    m_result.store(1000000);
    multithread_min(array, m_result);
  }
  end_t = std::chrono::system_clock::now();
  clock_delta = end_t - start_t;

  printf("Multithread results:\t %.2f \t\n", clock_delta.count());
  printf("Result: %d\n", int(m_result));
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
  end_pos = fmin(array.size(), end_pos);
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
  int block_size = array.size() / thread_amount;

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
  printf("Task3:\n");
  cv::Mat image = load_image();
  std::vector<int> filter(32, 0);
  generate_filter(filter);

  std::chrono::time_point<std::chrono::system_clock> start_t = std::chrono::system_clock::now();
  for (int t = 0; t < TIMES; ++t) {
    for (int i = 0; i < 3; ++i) {
      std::vector<int> array = get_channel_from_mat(image, i);
      std::vector<int> result_array(array.size(), 0);
      basic_conv(array, filter, result_array);
    }
  }
  std::chrono::time_point<std::chrono::system_clock> end_t = std::chrono::system_clock::now();
  std::chrono::duration<double> clock_delta = end_t - start_t;

  printf("Simple results:\t %.2f \t\n", clock_delta.count());

  start_t = std::chrono::system_clock::now();
  for (int t = 0; t < TIMES; ++t) {
    for (int i = 0; i < 3; ++i) {
      std::vector<int> array = get_channel_from_mat(image, i);
      std::vector<int> m_result_array(array.size(), 0);
      multithread_conv(array, filter, m_result_array);
    }
  }
  end_t = std::chrono::system_clock::now();
  clock_delta = end_t - start_t;

  printf("Multithread results:\t %.2f \t\n", clock_delta.count());
}


int main() {
  task1();
  task2();
  task3();
  return 0;
}
