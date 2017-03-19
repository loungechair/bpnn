#pragma once


#include "matrix.hpp"

#include <memory>
#include <map>
#include <algorithm>
#include <iostream>

#include <cstddef>
#include <cmath>

#include <cassert>

namespace nn
{
namespace input
{

template <typename CategoryType>
class CategoryStatistics
{
public:

  typedef CategoryType type;

  CategoryStatistics() : num_values(0) {}

  void ProcessValue(const CategoryType& category)
  {
    auto p = cat_num.find(category);

    if (p != cat_num.end()) {
      cat_freq[p->second]++;
    } else {
      int id = cat_num.size();
      cat_num.insert(std::make_pair(category, id));
      num_cat.insert(std::make_pair(id, category));
      cat_freq.insert(std::make_pair(id, 1));
    }
    ++num_values;
  }

  int GetNumCategories() const { return cat_num.size(); }

  int GetCategoryFrequency(const CategoryType& category) const
  {
    const auto& p = cat_num.find(category);
    if (p != cat_num.end()) {
      const auto& q = cat_freq.find(p->second);
      return q->second;
    }
    return 0;
  }

  double GetCategoryProbability(const CategoryType& category) const
  {
    return ((double)GetCategoryFrequency(category))/num_values;
  }

  std::vector<CategoryType> GetCategories() const
  {
    std::vector<CategoryType> out;

    for (auto& cat : num_cat) {
      out.push_back(cat.second);
    }

    return out;
  }
  
private:
  int num_values;
  std::map<CategoryType, int> cat_num;
  std::map<int, CategoryType> num_cat;
  std::map<int, int> cat_freq;
};



template <typename DataType>
class ScalarFieldStatistics
{
public:

  typedef DataType type;

  ScalarFieldStatistics()
    : num_values(0),
      min_value(0.0),
      mean(0.0),
      M2(0.0)
  {}

  void ProcessValue(DataType value)
  {
    if (num_values == 0 || value < min_value) {
      min_value = value;
    }
    if (num_values == 0 || value > max_value) {
      max_value = value;
    }

    ++num_values;

    double d1 = value - mean;
    mean += d1/num_values;
    double d2 = value - mean;
    M2 += d1 * d2;
  }

  DataType GetMinimum() const { return min_value; }
  DataType GetMaximum() const { return max_value; }
  double GetMean() const { return mean; }
  double GetVariance() const { return M2/(num_values-1); }
  double GetStandardDeviation() const { return sqrt(GetVariance()); }
  int GetObservations() const { return num_values; }

  std::vector<double> GetResults() const
  {
    return {(double)num_values, min_value, max_value, mean, GetVariance(), GetStandardDeviation()};
  }
  
private:
  int    num_values;
  DataType min_value;
  DataType max_value;
  double mean;
  double M2;
};




template <typename DataType, typename InputType>
void CalculateFieldStatistic(int offset,
                             const std::vector<InputType>& data,
                             ScalarFieldStatistics<DataType>& stat)
{
  for (auto& item : data) {
    char* intype = (char*)(&item);
    DataType x = *(DataType *)(intype + offset);

    stat.ProcessValue(x);
  }
}



template <typename CategoryType, typename InputDataType>
void CalculateFieldStatistic(int offset,
                             const std::vector<InputDataType>& data,
                             CategoryStatistics<CategoryType>& stat)
{
  for (auto& item : data) {
    char* intype = (char*)(&item);
    CategoryType x = *(CategoryType *)(intype + offset);

    stat.ProcessValue(x);
  }
}


#define nn_CALCULATE_FIELD_STATS(data, structType, group, stat) \
  CalculateFieldStatistic(offsetof(structType, group), data, stat)






class FieldEncoder
{
public:
  virtual std::vector<double> EncodeField(const void* field_ptr) = 0;
  virtual void DecodeField(std::vector<double>::const_iterator& p, const void* field_ptr) = 0;
  virtual size_t Length() const = 0;
};



#define nn_ADD_FIELD_ENCODER(encoder, structType, field, field_encoder) \
  encoder.AddFieldEncoder(offsetof(structType, field), field_encoder)


template <typename InputType>
class InputEncoder
{
public:

  void AddFieldEncoder(int offset, std::shared_ptr<FieldEncoder> encoder)
  {
    encoders.insert(std::make_pair(offset, encoder));
  }

  std::vector<double> Encode(const InputType* data) const;
  void Decode(const std::vector<double>& input, InputType* data) const;
  InputType Decode(const std::vector<double>& input) const;

  size_t Length() const {
    size_t length = 0;
    for (auto& p : encoders) {
      length += p.second->Length();
    }
    return length;
  }

private:
  std::map<int, std::shared_ptr<FieldEncoder>> encoders;
};




class DoubleDefaultEncoder : public FieldEncoder
{
public:
  std::vector<double> EncodeField(const void *field_ptr)
  {
    return std::vector<double>(1, *(double *)field_ptr);
  }

  void DecodeField(std::vector<double>::const_iterator& p, const void* field_ptr)
  {
    *(double *)field_ptr = *p;
    ++p;
  }

  size_t Length() const { return 1; }
};


// Has one output per category, turns one output on for a given category.
class IntegerCategoryEncoder : public FieldEncoder
{
public:
  IntegerCategoryEncoder(int min_value_use,
                         int max_value_use,
                         double on_value_use = 1.0,
                         double off_value_use = 0.0);

  std::vector<double> EncodeField(const void* field_ptr);
  void DecodeField(std::vector<double>::const_iterator& p, const void* field_ptr);

  size_t Length() const { return empty_pattern.size(); }

private:
  int min_value;
  int max_value;
  int num_categories;
  double on_value;
  double off_value;
  std::vector<double> empty_pattern;
};


// Converts integer to binary
class IntegerToBinaryEncoder : public FieldEncoder
{
public:
  IntegerToBinaryEncoder(int min_value_use,
                         int max_value_use,
                         double on_value_use = 1.0,
                         double off_value_use = 0.0);
  
  std::vector<double> EncodeField(const void* field_ptr) override;
  void DecodeField(std::vector<double>::const_iterator& p, const void* field_ptr);

  size_t Length() const { return empty_pattern.size(); }

private:
  int max_value;
  int min_value;
  int num_values;
  double on_value;
  double off_value;
  int bits;
  std::vector<double> empty_pattern;


  int num_bits(int x);
};



template <typename CategoryType, typename IntegerEncoderType = IntegerCategoryEncoder>
class CategoryEncoder : public FieldEncoder
{
public:

  CategoryEncoder(double on_value_use = 1.0, double off_value_use = 0.0)
    : on_value(on_value_use),
      off_value(off_value_use)
  {
  }

  CategoryEncoder(const std::vector<CategoryType>& categories, double on_value_use = 1.0, double off_value_use = 0.0)
    : CategoryEncoder{ on_value_use, off_value_use }
  {
    AddCategories(categories);
  }


  std::vector<double> EncodeField(const void* field_ptr) override
  {
    const CategoryType* val = static_cast<const CategoryType*>(field_ptr);
    const auto& p = category_id.find(*val);

    if (p == category_id.end()) {
      throw 1111;
    }
    int id = p->second;
    
    return int_encoder->EncodeField(&id);
  }
  
  void DecodeField(std::vector<double>::const_iterator& p, const void* field_ptr) override
  {
    int value;
    int_encoder->DecodeField(p, &value);

    *(CategoryType *)field_ptr = category_name[value];
  }


  void AddCategory(const CategoryType& category)
  {
    auto p = category_id.find(category);
    if (p == category_id.end()) {
      int id = category_id.size();
      category_id.insert(std::make_pair(category, id));
      category_name.insert(std::make_pair(id, category));
      int_encoder = std::make_shared<IntegerEncoderType>(0,
                                                         category_id.size() - 1,
                                                         on_value,
                                                         off_value);
    }
  }


  void AddCategories(const std::vector<CategoryType>& categories) {
    for (const auto& p : categories) {
      AddCategory(p);
    }
  }

  size_t Length() const { return int_encoder->Length(); }

private:
  double on_value;
  double off_value;
  
  std::map<CategoryType, int> category_id;
  std::map<int, CategoryType> category_name;

  std::shared_ptr<IntegerEncoderType> int_encoder;
};



// Scales from [a, b] to [c, d]
class DoubleScaleEncoder : public FieldEncoder
{
public:
  DoubleScaleEncoder(double a, double b, double c, double d);

  std::vector<double> EncodeField(const void* field_ptr) override;
  void DecodeField(std::vector<double>::const_iterator& p, const void* field_ptr) override;

  size_t Length() const { return 1; }

private:
  double in_min;
  double in_max;
  double out_min;
  double out_max;
};


// Scales x to (x - mu)/sigma
class DoubleNormalizeEncoder : public FieldEncoder
{
public:
  DoubleNormalizeEncoder(double mean_use, double std_dev_use);

  std::vector<double> EncodeField(const void* field_ptr) override;
  void DecodeField(std::vector<double>::const_iterator& p, const void* field_ptr) override;

  size_t Length() const { return 1; }

private:
  double mean;
  double std_dev;
};





template <typename InputType>
std::vector<double>
InputEncoder<InputType>::Encode(const InputType* data) const
{
  std::vector<double> out;
  
  const char *base_ptr = (char *)data;
  
  for (auto& encoder : encoders) {
    const char *field_ptr = base_ptr + encoder.first;
    auto field_out = encoder.second->EncodeField((void *)field_ptr);
    
    out.insert(end(out), begin(field_out), end(field_out));
  }
  
  return out;
}

template <typename InputType>
void
InputEncoder<InputType>::Decode(const std::vector<double>& input, InputType* data) const
{
  const char* base_ptr = (char *)data;
  auto p = input.begin();
  
  for (auto& encoder : encoders) {
    const char *field_ptr = base_ptr + encoder.first;
    
    encoder.second->DecodeField(p, (void *)field_ptr);
  }
}

template <typename InputType>
InputType
InputEncoder<InputType>::Decode(const std::vector<double>& input) const
{
  InputType data;
  const char* base_ptr = (char *)&data;
  auto p = input.begin();

  for (auto& encoder : encoders) {
    const char *field_ptr = base_ptr + encoder.first;

    encoder.second->DecodeField(p, (void *)field_ptr);
  }
  return data;
}



}
}
