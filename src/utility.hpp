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
  void Start()
  {
    start_time = std::chrono::high_resolution_clock::now();
  }
  double Stop()
  {
    end_time = std::chrono::high_resolution_clock::now();
    total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return (elapsed_time = std::chrono::duration<double>(total_time).count());
  }
  double GetElapsedTime() const { return elapsed_time; }

  std::string GetElapsedTimeAsString() const
  {
    auto hh = std::chrono::duration_cast<std::chrono::hours>(total_time);
    auto mm = std::chrono::duration_cast<std::chrono::minutes>(total_time);
    auto ss = std::chrono::duration_cast<std::chrono::seconds>(total_time);
    //auto ll = std::chrono::duration_cast<std::chrono::milliseconds>(total_time);

    int h = hh.count();
    int m = mm.count() - 60 * h;
    int s = ss.count() - 60 * mm.count();
    int l = total_time.count() - 1000000 * ss.count();

    std::ostringstream out;

    out << h << ":" << std::setw(2) << std::setfill('0') << m
             << ":" << std::setw(2) << std::setfill('0') << s
             << "." << std::setw(6) << std::setfill('0') << l;

    return out.str();
  }

private:
  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;
  std::chrono::microseconds total_time;
  double elapsed_time;
};




class Observer
{
public:
  virtual ~Observer() {}

  virtual void Update() = 0;
};


class Observable
{
public:
  void Attach(Observer* obs) { observers.insert(obs); }
  void Detatch(Observer* obs) { observers.erase(obs); }
  
  void Notify()
  {
    for (auto& obs : observers) {
      obs->Update();
    }
  }

private:
  std::set<Observer*> observers;
};

}

}
