#ifndef timer_hpp
#define timer_hpp

#include <chrono>


class Timer {
    
  public:
    void start();
    void stop();
    float elapsed();

  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> t0, t1;
};

#endif
