#pragma once

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


}
}
