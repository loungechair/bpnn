#pragma once

#include <set>

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
  Timer(const std::string& name);
  void Start();
  double Stop();
  double GetTime();

private:
  
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
  void Attach(Observer* obs)
  {
    observers.insert(obs);
  }
  void Detatch(Observer* obs)
  {
    observers.erase(obs);
  }
  
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
