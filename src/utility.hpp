#pragma once

#include <set>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace nn {
namespace utility {


template <typename ForwardIterator, typename Function>
void adjacent_pairs(ForwardIterator first, ForwardIterator last, Function f)
{
  if (first != last) {
    ForwardIterator trailer = first;
    ++first;
    for (; first != last; ++first, ++trailer)
      f(*trailer, *first);
  }
}


class Timer
{
public:
  Timer() : running(false), elapsed_time(0) {}

  void Start()
  {
    start_time = std::chrono::high_resolution_clock::now();
    running = true;
  }
  double Stop()
  {
    end_time = std::chrono::high_resolution_clock::now();
    total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    running = false;
    return (elapsed_time = std::chrono::duration<double>(total_time).count());
  }

  double GetElapsedTime() const
  {
    if (running) {
      auto now = std::chrono::high_resolution_clock::now();
      auto time_so_far = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time);
      return (std::chrono::duration<double>(total_time).count());
    } else {
      return elapsed_time;
    }
  }

  std::string GetElapsedTimeAsString() const
  {
    if (running) {
      auto now = std::chrono::high_resolution_clock::now();
      auto time_so_far = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time);
      return GetElapsedTimeAsStringImpl(time_so_far);
    } else {
      return GetElapsedTimeAsStringImpl(total_time);
    }
  }

private:
  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;
  std::chrono::microseconds total_time;
  bool running;
  double elapsed_time;

  std::string GetElapsedTimeAsStringImpl(std::chrono::microseconds time_duration) const
  {
    auto hh = std::chrono::duration_cast<std::chrono::hours>(time_duration);
    auto mm = std::chrono::duration_cast<std::chrono::minutes>(time_duration);
    auto ss = std::chrono::duration_cast<std::chrono::seconds>(time_duration);

    int64_t h = hh.count();
    int64_t m = mm.count() - 60 * h;
    int64_t s = ss.count() - 60 * mm.count();
    int64_t l = time_duration.count() - 1000000 * ss.count();
    l = (l + 500)/1000;

    std::ostringstream out;

    out << h << ":" << std::setw(2) << std::setfill('0') << m
             << ":" << std::setw(2) << std::setfill('0') << s
             << "." << std::setw(3) << std::setfill('0') << l;

    return out.str();
  }
};






class Observer
{
public:
  virtual ~Observer() {}

  virtual void UpdateBatch() = 0;
  virtual void UpdateEpoch() = 0;
};


class Observable
{
public:
  void Attach(Observer* obs) { observers.insert(obs); }
  void Detatch(Observer* obs) { observers.erase(obs); }
  
  void NotifyBatch() { for (auto& obs : observers) { obs->UpdateBatch(); } }
  void NotifyEpoch() { for (auto& obs : observers) { obs->UpdateEpoch(); } }

private:
  std::set<Observer*> observers;
};

}

}
