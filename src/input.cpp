#include "input.hpp"

#include <algorithm>


#include <cstddef>
#include <cassert>



namespace nn
{
namespace input
{


IntegerCategoryEncoder::IntegerCategoryEncoder(int min_value_use,
                                               int max_value_use,
                                               double on_value_use,
                                               double off_value_use)
  : min_value(min_value_use),
    max_value(max_value_use),
    num_categories(max_value - min_value + 1),
    on_value(on_value_use),
    off_value(off_value_use),
    empty_pattern(num_categories, off_value)
{
}


std::vector<double>
IntegerCategoryEncoder::EncodeField(const void* field_ptr)
{
  auto out = empty_pattern;
  int x = *(int *)field_ptr - min_value;
  out[x] = on_value;
  
  return out;
}


void
IntegerCategoryEncoder::DecodeField(std::vector<double>::const_iterator& p, const void* field_ptr)
{
  int idx = std::distance(p, std::max_element(p, p + num_categories));
  *(int *)field_ptr = idx + min_value;
  p += num_categories;
}




IntegerToBinaryEncoder::IntegerToBinaryEncoder(int min_value_use,
                                               int max_value_use,
                                               double on_value_use,
                                               double off_value_use)
  : min_value(min_value_use),
    max_value(max_value_use),
    num_values(max_value - min_value + 1),
    on_value(on_value_use),
    off_value(off_value_use),
    bits(num_bits(num_values)),
    empty_pattern(bits, off_value)
{
}

std::vector<double>
IntegerToBinaryEncoder::EncodeField(const void* field_ptr)
{
  int value = *(int *)field_ptr - min_value;
  int idx = 0;
  auto out = empty_pattern;
  while (value) {
    if (value % 2) {
      out[idx] = on_value;
    }
    value >>= 1;
    ++idx;
  }
  
  return out;
}

void
IntegerToBinaryEncoder::DecodeField(std::vector<double>::const_iterator& p, const void* field_ptr)
{
  int value = 0;
  int this_bit_val = 1;
  
  double mid_val = (on_value + off_value)/2;
  
  for (int i = 0; i < bits; ++i, ++p, this_bit_val *= 2) {
    if (*p > mid_val) {
      value += this_bit_val;
    }
  }
  
  *(int*)field_ptr = value + min_value;
}

int
IntegerToBinaryEncoder::num_bits(int x)
{
  // assert(x > 0);
  // return sizeof(int) * 8 - __builtin_clz(x) + 1;
  
  int num_bits = 1;
  int max_val = 2;
  
  while (max_val < x) {
    ++num_bits;
    max_val *= 2;
  }
  
  return num_bits;
}










DoubleScaleEncoder::DoubleScaleEncoder(double a, double b, double c, double d)
  : in_min(a),
    in_max(b),
    out_min(c),
    out_max(d)
{
}



std::vector<double>
DoubleScaleEncoder::EncodeField(const void* field_ptr)
{
  double in_val = *(double *)field_ptr;
  
  double out_val = out_min + (out_max - out_min) * (in_val - in_min)/(in_max - in_min);
  
  return std::vector<double>(1, out_val);
}



void
DoubleScaleEncoder::DecodeField(std::vector<double>::const_iterator& p, const void* field_ptr)
{
  *(double *)field_ptr = in_min + (in_max - in_min) * (*p - out_min)/(out_max - out_min);
  ++p;
}




DoubleNormalizeEncoder::DoubleNormalizeEncoder(double mean_use, double std_dev_use)
  : mean(mean_use),
    std_dev(std_dev_use)
{
}


std::vector<double>
DoubleNormalizeEncoder::EncodeField(const void* field_ptr)
{
  double input_val = *(double *)field_ptr;
  double output_val = (input_val - mean)/std_dev;
  
  return std::vector<double>(1, output_val);
}


void
DoubleNormalizeEncoder::DecodeField(std::vector<double>::const_iterator& p, const void* field_ptr)
{
  *(double *)field_ptr = *p * std_dev + mean;
  ++p;
}


} // namespace
} // namespace
