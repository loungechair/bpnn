#include "network.hpp"
#include "train.hpp"
#include "input.hpp"

#include <iostream>
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

  auto iris_categories = iris_stats.GetCategories();

  nn::input::InputEncoder<IrisOutput> output_encoder;
  auto iris_type_encoder = std::make_shared<nn::input::CategoryEncoder<std::string>>();
  iris_type_encoder->AddCategories(iris_categories);
  nn_ADD_FIELD_ENCODER(output_encoder, IrisOutput, iris_type, iris_type_encoder);


  nn::input::TrainingData training_data;
  for (int i = 0; i < input_data.size(); ++i) {
    std::vector<double> input;
    input.push_back(input_data[i].sepal_length);
    input.push_back(input_data[i].sepal_width);
    input.push_back(input_data[i].petal_length);
    input.push_back(input_data[i].petal_width);

    training_data.AddPair(input, output_encoder.Encode(&output_data[i]));
  }


  //nn::input::TrainingData training_data{
  //  {
  //    { 0, 0, 0, 0 },
  //    { 0, 1, 0, 1 },
  //    { 0, 1, 1, 0 },
  //    { 1, 0, 0, 1 }
  //  },
  //  
  //  {
  //    { -1.75, -1.75, -1.75, -1.75 },
  //    { -1.75, 0.992, -1.75, 0.992 },
  //    { -1.75, 0.992, 0.992, -1.75 },
  //    { 1.992, -1.75, -1.75, 0.992 }
  //  }
  //};

  nn::dblmatrix m(10, 10);
  nn::dblvector v{ 1, 2, 3, 4, 5, 4, 3, 2, 1, 0 };
  m.SetRowValues(5, v);
  auto r = m.GetRow(5);
  for (auto& p : r) {
    std::cout << p << " -> ";
  }
  std::cout << std::endl;

  auto hid_act = std::make_shared<nn::SigmoidActivation>(-1, 1);
  auto out_act = std::make_shared<nn::SigmoidActivation>(0, 1);
  //auto out_act = std::make_shared<nn::LinearActivation>();

  nn::Network n({4, 48, 3}, hid_act, out_act);
  
  auto tr = std::make_unique<nn::train::BackpropTrainingAlgorithm>(n, 0.1, std::make_shared<nn::SquaredError>());

  tr->InitializeNetwork();
  tr->SetTrainingData(&training_data);
  //
  auto start_time = std::chrono::high_resolution_clock::now();
  tr->Train();
  auto end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> total_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
  std::cout << "Total time was " << total_time.count() << std::endl;

}
