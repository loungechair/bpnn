#pragma once

#include <iostream>
#include <string>
#include <exception>


namespace nn
{

#define CHECK_TEMPLATE_REAL_TYPE(T, ClassName) \
static_assert(std::is_floating_point<T>::value, "" #ClassName "<" #T "> : " #T " must be double or float");

inline void NonFatalError(const std::string& msg)
{
  std::cerr << "Warning: " << msg << std::endl;
}

class Exception : std::runtime_error
{
public:
  Exception(const std::string& msg) : std::runtime_error(msg)
  {}

  std::string Message() const { return std::string(what()); }
};

} // namespace nn