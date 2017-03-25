#pragma once

#include "matrix.hpp"
#include "input.hpp"

#include <map>
#include <vector>
#include <list>

namespace nn
{


class Batch
{
public:
  Batch(int batch_size, int input_length, int output_length)
    : max_batch_size(batch_size),
      current_batch_size(0),
      input(batch_size, input_length),
      output(batch_size, output_length)
  {}

  int AddPair(const dblvector& in, const dblvector& out)
  {
    if (current_batch_size >= max_batch_size) {
      throw "Batch Full!";
    }

    input.SetRowValues(current_batch_size, in);
    output.SetRowValues(current_batch_size, out);
    ++current_batch_size;
  }

  const dblmatrix& Input() const { return input; }
  const dblmatrix& Output() const { return output; }

  int MaxBatchSize() const { return max_batch_size; }
  int CurrentBatchSize() const { return current_batch_size; }

private:
  int max_batch_size;
  int current_batch_size;
  dblmatrix input;
  dblmatrix output;
};


namespace td_test
{


class ActivationPattern
{
  struct ActivationRecord
  {
    static const int DEFAULT_START_TIME = -10000;
    static const int DEFAULT_END_TIME   = 10000;
    double start_time;
    double end_time;
    dblvector activation;

    ActivationRecord()
      : start_time(DEFAULT_START_TIME),
        end_time(DEFAULT_END_TIME)
    {}

    ActivationRecord(const dblvector& activation_use)
      : start_time(DEFAULT_START_TIME),
        end_time(DEFAULT_END_TIME),
        activation(activation_use)
    {}

    ActivationRecord(double start_time_use, double end_time_use, const dblvector& activation_use)
      : start_time(start_time_use),
        end_time(end_time_use),
        activation(activation_use)
    {}

    bool operator<(const ActivationRecord& b) const {
      return start_time < b.start_time;
    }
    bool operator==(double t) const
    {
      return (start_time <= t && t <= end_time);
    }
  };

public:
  dblvector GetActivationValue(const std::string& layer_name) const;
  dblvector GetActivationValue(const std::string& layer_name, double time) const;

  void AddActivationValue(const std::string& layer_name, const nn::dblvector& values);
  void AddActivationValue(const std::string& layer_name, double start_time, double stop_time,
                          const nn::dblvector& values);
private:
  std::map<std::string, std::list<ActivationRecord>> data;
};


class TrainingData
{
public:
  bool HasActivation(const std::string& layer_name, double time) const;
  const dblmatrix& GetActivation(const std::string& layer_name, double time) const;

private:
};



template <typename InputType, typename OutputType>
class TrainingDataBuilder
{
public:

  void AddPair(const InputType& input_data, const OutputType& output_data);
  TrainingData GetTrainingData() const;

private:
};

} // namespace td_test


template <typename InputType, typename OutputType>
class TrainingData
{
public:
  TrainingData(size_t batch_size_use, size_t num_batches_use,
               size_t input_length_use, size_t output_length_use,
               const input::InputEncoder<InputType>* input_enc_use,
               const input::InputEncoder<OutputType>* output_enc_use)
    : batch_size(batch_size_use),
      input_length(input_length_use),
      output_length(output_length_use),
      num_batches(num_batches_use),
      num_patterns(0),
      batch_to_add_to(0),
      batches(num_batches, Batch(batch_size, input_length, output_length)),
      input_encoder(input_enc_use),
      output_encoder(output_enc_use)
  {
  }

  const std::vector<Batch>& Batches() const { return batches; }

  void AddPair(const InputType& in, const OutputType& out)
  {
    batches[batch_to_add_to].AddPair(input_encoder->Encode(&in), output_encoder->Encode(&out));
    batch_to_add_to = (batch_to_add_to + 1) % num_batches;
    ++num_patterns;
  }

private:
  size_t batch_size;
  size_t input_length;
  size_t output_length;
  size_t num_batches;
  size_t num_patterns;
  size_t batch_to_add_to;
  std::vector<Batch> batches;
  const input::InputEncoder<InputType>* input_encoder;
  const input::InputEncoder<OutputType>* output_encoder;
};



} // namespace nn