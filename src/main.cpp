#include "network.hpp"
#include "train.hpp"
#include "input.hpp"

#include <iostream>
#include <chrono>
#include <ratio>



int
main(int argc, char *argv[])
{
  auto hid_act = std::make_shared<nn::SigmoidActivation>(-1, 1);
  //auto out_act = std::make_shared<nn::SigmoidActivation>(0, 1);
  auto out_act = std::make_shared<nn::LinearActivation>();

  nn::Network n({4, 6, 2, 6, 4}, hid_act, out_act);
  
  auto tr = std::make_unique<nn::train::BackpropTrainingAlgorithm>(n, 0.1, std::make_shared<nn::SquaredError>());

  tr->InitializeNetwork();

  auto err_fn = std::make_shared<nn::SquaredError>();

  auto start_time = std::chrono::high_resolution_clock::now();

  tr->Train();

  auto end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> total_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
  std::cout << "Total time was " << total_time.count() << std::endl;

}
