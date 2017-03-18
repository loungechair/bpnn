#pragma once

#include "matrix.hpp"
#include "input.hpp"

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



template <typename InputType, typename OutputType>
class TrainingData
{
public:
  TrainingData(int batch_size_use, int num_batches_use, int input_length_use, int output_length_use,
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

  int AddPair(const InputType& in, const OutputType& out)
  {
    batches[batch_to_add_to].AddPair(input_encoder->Encode(&in), output_encoder->Encode(&out));
    batch_to_add_to = (batch_to_add_to + 1) % num_batches;
    ++num_patterns;
    return num_patterns;
  }

private:
  int batch_size;
  int input_length;
  int output_length;
  int num_batches;
  int num_patterns;
  int batch_to_add_to;
  std::vector<Batch> batches;
  const input::InputEncoder<InputType>* input_encoder;
  const input::InputEncoder<OutputType>* output_encoder;
};


} // namespace nn