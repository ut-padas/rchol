#include "timer.hpp"

void Timer::start() {
  t0 = std::chrono::high_resolution_clock::now();
}

void Timer::stop() {
  t1 = std::chrono::high_resolution_clock::now();
}

float Timer::elapsed() {
  auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0);
  return dt.count()/1.e3;
}

