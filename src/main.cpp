#include "network.hpp"
#include "train.hpp"
#include "input.hpp"

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

int
main(int argc, char *argv[])
{
  std::vector<IrisInput> input_data;
  std::vector<IrisOutput> output_data;
  
  ReadIrisData(input_data, output_data);


  nn::input::CategoryStatistics<std::string> iris_stats;

  nn_CALCULATE_FIELD_STATS(output_data, IrisOutput, iris_type, iris_stats);

  nn::input::InputEncoder<IrisOutput> output_encoder;
  auto iris_type_encoder = std::make_shared<nn::input::CategoryEncoder<std::string>>(iris_stats.GetCategories());
  nn_ADD_FIELD_ENCODER(output_encoder, IrisOutput, iris_type, iris_type_encoder);


  nn::input::TrainingData training_data(151, 4, 3);
  for (int i = 0; i < input_data.size(); ++i) {
    std::vector<double> input {input_data[i].sepal_length, input_data[i].sepal_width,
                               input_data[i].petal_length, input_data[i].petal_width};

    training_data.SetPair(i, input, output_encoder.Encode(&output_data[i]));
  }

  auto hid_act = std::make_shared<nn::TanhActivation>();
  auto out_act = std::make_shared<nn::SigmoidActivation>(0, 1);

  auto err_function = std::make_shared<nn::CrossEntropyError>();

  nn::Network network({4, 240, 240, 3}, 151, hid_act, out_act, err_function);
  //nn::Network network({ 4, 24, 24, 3 }, 151, hid_act, out_act, err_function);

  nn::ErrorStatistics<double> err_stats(10, network);
  nn::ErrorPrinter err_printer(100, network);

  network.Attach(&err_stats);
  network.Attach(&err_printer);

  nn::train::BackpropTrainingParameters params{ 0.0005, 0.9, 0, true, 100'000, 0.1 };
  
  auto tr = std::make_unique<nn::train::BackpropTrainingAlgorithm>(network, params);

  tr->InitializeNetwork();
  tr->SetTrainingData(&training_data);

  nn::utility::Timer train_timer;
  train_timer.Start();
  tr->Train();
  train_timer.Stop();

  std::cout << "Total time was " << train_timer.GetElapsedTimeAsString() << std::endl;
  return 0;

  auto& in = training_data.in;
  auto& targ = training_data.out;

  auto& output = network.FeedForward(in);

  for (int pattern = 0; pattern < output.Rows(); ++pattern) {
    auto& in_p = in.GetRow(pattern);
    auto& out_p = output.GetRow(pattern);

    std::cout << "{";
    for (auto& x : in_p) {
      std::cout << std::fixed << std::setprecision(1) << std::setw(8) << x;
    }
    std::cout << "} --> {";
    for (auto& x : out_p) {
      std::cout << std::fixed << std::setprecision(4) << std::setw(8) << x;
    }
    std::cout << "} --> ";
    IrisOutput iris_out;
    output_encoder.Decode(output.GetRowValues(pattern), &iris_out);
    std::cout << iris_out.iris_type << std::endl;
  }
  std::cout << "Total time was " << train_timer.GetElapsedTimeAsString() << std::endl;
}
