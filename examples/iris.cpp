#include "../src/network.hpp"
#include "../src/train.hpp"
#include "../src/input.hpp"

#include "../src/trainingdata.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <ratio>
#include <string>
#include <memory>

struct IrisInput
{
  double sepal_length;
  double sepal_width;
  double petal_length;
  double petal_width;
};

struct IrisOutput
{
  std::string iris_type;
};


void ReadIrisData(std::vector<IrisInput>& input_data, std::vector<IrisOutput>& output_data)
{
  std::ifstream in("E:/Dropbox/MLDatasets/iris.data");
  if (!in) {
    std::cerr << "Couldn't find file!";
    return;
  }

  IrisInput input;
  IrisOutput output;

  while (in.good()) {
    in >> input.sepal_length >> input.sepal_width
      >> input.petal_length >> input.petal_width
      >> output.iris_type;

    input_data.push_back(input);
    output_data.push_back(output);
  }
  in.close();
}

void IrisNetwork()
{
  std::vector<IrisInput> input_data;
  std::vector<IrisOutput> output_data;

  ReadIrisData(input_data, output_data);


  nn::input::CategoryStatistics<std::string> iris_type_stats;
  nn_CALCULATE_FIELD_STATS(output_data, IrisOutput, iris_type, iris_type_stats);

  nn::input::InputEncoder<IrisOutput> output_encoder;
  auto iris_type_encoder = std::make_shared<nn::input::CategoryEncoder<std::string>>(iris_type_stats.GetCategories());
  nn_ADD_FIELD_ENCODER(output_encoder, IrisOutput, iris_type, iris_type_encoder);

  nn::input::ScalarFieldStatistics<double> sepal_length_stats;
  nn_CALCULATE_FIELD_STATS(input_data, IrisInput, sepal_length, sepal_length_stats);
  nn::input::ScalarFieldStatistics<double> sepal_width_stats;
  nn_CALCULATE_FIELD_STATS(input_data, IrisInput, sepal_width, sepal_width_stats);
  nn::input::ScalarFieldStatistics<double> petal_length_stats;
  nn_CALCULATE_FIELD_STATS(input_data, IrisInput, petal_length, petal_length_stats);
  nn::input::ScalarFieldStatistics<double> petal_width_stats;
  nn_CALCULATE_FIELD_STATS(input_data, IrisInput, petal_width, petal_width_stats);

  nn::input::InputEncoder<IrisInput> input_encoder;
  auto sepal_length_encoder = std::make_shared<nn::input::DoubleScaleEncoder>(sepal_length_stats.GetMinimum(), sepal_length_stats.GetMaximum(), -1, 1);
  auto sepal_width_encoder  = std::make_shared<nn::input::DoubleScaleEncoder>(sepal_width_stats.GetMinimum(), sepal_width_stats.GetMaximum(), -1, 1);
  auto petal_length_encoder = std::make_shared<nn::input::DoubleScaleEncoder>(petal_length_stats.GetMinimum(), petal_length_stats.GetMaximum(), -1, 1);
  auto petal_width_encoder  = std::make_shared<nn::input::DoubleScaleEncoder>(petal_width_stats.GetMinimum(), petal_width_stats.GetMaximum(), -1, 1);

  //auto sepal_length_encoder = std::make_shared<nn::input::DoubleDefaultEncoder>();
  //auto sepal_width_encoder = std::make_shared<nn::input::DoubleDefaultEncoder>();
  //auto petal_length_encoder = std::make_shared<nn::input::DoubleDefaultEncoder>();
  //auto petal_width_encoder = std::make_shared<nn::input::DoubleDefaultEncoder>();

  nn_ADD_FIELD_ENCODER(input_encoder, IrisInput, sepal_length, sepal_length_encoder);
  nn_ADD_FIELD_ENCODER(input_encoder, IrisInput, sepal_width, sepal_width_encoder);
  nn_ADD_FIELD_ENCODER(input_encoder, IrisInput, petal_length, petal_length_encoder);
  nn_ADD_FIELD_ENCODER(input_encoder, IrisInput, petal_width, petal_width_encoder);

  const int BATCH_SIZE = 151;
  const int NUM_BATCHES = 1;

  nn::TrainingData<IrisInput, IrisOutput> training_data(BATCH_SIZE, NUM_BATCHES, input_encoder.Length(), output_encoder.Length(), &input_encoder, &output_encoder);

  for (int i = 0; i < input_data.size(); ++i) {
    training_data.AddPair(input_data[i], output_data[i]);
  }

  auto hid_act = std::make_shared<nn::TanhActivation>();
  auto out_act = std::make_shared<nn::SigmoidActivation>(0, 1);

  auto err_function = std::make_shared<nn::CrossEntropyError>();

  //nn::Network network({4, 240, 240, 3}, 151, hid_act, out_act, err_function);
  nn::Network network({ 4, 24, 24, 3 }, BATCH_SIZE, hid_act, out_act, err_function);

  nn::utility::Timer train_timer;

  nn::ErrorStatistics<double> err_stats(10, network);
  nn::ErrorPrinter err_printer(100, network, &train_timer);

  network.Attach(&err_stats);
  network.Attach(&err_printer);

  nn::train::BackpropTrainingParameters params{ 0.001, 0.9, 0, true, 100'000, 0.1 };

  auto tr = std::make_unique<nn::train::BackpropTrainingAlgorithm>(network, params);

  tr->InitializeNetwork();
  tr->SetTrainingData(&training_data.Batches());

  train_timer.Start();
  tr->Train();
  train_timer.Stop();

  for (auto& batch : training_data.Batches()) {
    auto& in = batch.Input();
    auto& targ = batch.Output();

    std::cout << "Batch Size == " << in.Rows() << std::endl;

    auto& output = network.FeedForward(in);

    for (int pattern = 0; pattern < output.Rows(); ++pattern) {
      //auto& in_p = input_encoder.Decode(in.GetRowValues(pattern));
      //auto& out_p = output_encoder.Decode(output.GetRowValues(pattern));

      auto& in_p = in.GetRowValues(pattern);
      auto& out_p = output.GetRowValues(pattern);

      std::cout << "{";
      for (auto& x : in_p) {
        std::cout << std::fixed << std::setprecision(1) << std::setw(8) << x;
      }
      std::cout << "} --> {";
      for (auto& x : out_p) {
        std::cout << std::fixed << std::setprecision(4) << std::setw(8) << x;
      }
      std::cout << "} --> " << output_encoder.Decode(out_p).iris_type << std::endl;
    }
    std::cout << "Total time was " << train_timer.GetElapsedTimeAsString() << std::endl;
  }
}