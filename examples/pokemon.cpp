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




struct PokemonInput
{
  double hp;
  double attack;
  double defense;
  double sp_attack;
  double sp_defense;
  double speed;
};


struct PokemonOutput
{
  std::string name;
  std::string type1;
  std::string type2;
};


std::istream& operator >> (std::istream& in, PokemonInput& pokemon)
{
  in >> pokemon.hp
    >> pokemon.attack
    >> pokemon.defense
    >> pokemon.sp_attack
    >> pokemon.sp_defense
    >> pokemon.speed;

  return in;
}

std::istream& operator >> (std::istream& in, PokemonOutput& out)
{
  in >> out.name >> out.type1 >> out.type2;
  return in;
}

std::ostream& operator << (std::ostream& out, PokemonOutput& poke)
{
  out << poke.name << "\t" << poke.type1 << "\t" << poke.type2;
  return out;
}

void
ReadPokemonData(std::vector<PokemonOutput>& outs, std::vector<PokemonInput>& ins)
{
  std::ifstream in("E:/Dropbox/MLDatasets/Pokemon.txt");

  if (!in) {
    std::cerr << "Fail" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string header;
  std::getline(in, header);

  std::string tmp_string;
  int tmp_int;


  PokemonOutput poke_out;
  PokemonInput poke_in;

  while (in.good()) {
    in >> tmp_int >> poke_out >> tmp_int >> poke_in >> tmp_int >> tmp_string;
    if (!in.good()) break;

    outs.push_back(poke_out);
    ins.push_back(poke_in);
  }

  in.close();
}

void
PokemonNetwork()
{
  std::vector<PokemonOutput> outs;
  std::vector<PokemonInput> ins;
  ReadPokemonData(outs, ins);

  nn::input::CategoryStatistics<std::string> name_stats;
  nn::input::CategoryStatistics<std::string> type1_stats;
  nn::input::CategoryStatistics<std::string> type2_stats;

  nn_CALCULATE_FIELD_STATS(outs, PokemonOutput, name, name_stats);
  nn_CALCULATE_FIELD_STATS(outs, PokemonOutput, type1, type1_stats);
  nn_CALCULATE_FIELD_STATS(outs, PokemonOutput, type2, type2_stats);

  auto name_encoder = std::make_shared<nn::input::CategoryEncoder<std::string, nn::input::IntegerToBinaryEncoder>>(name_stats.GetCategories());
  auto type1_encoder = std::make_shared<nn::input::CategoryEncoder<std::string, nn::input::IntegerToBinaryEncoder>>(type1_stats.GetCategories());
  auto type2_encoder = std::make_shared<nn::input::CategoryEncoder<std::string, nn::input::IntegerToBinaryEncoder>>(type2_stats.GetCategories());
  nn::input::InputEncoder<PokemonOutput> output_encoder;
  nn_ADD_FIELD_ENCODER(output_encoder, PokemonOutput, name, name_encoder);
  nn_ADD_FIELD_ENCODER(output_encoder, PokemonOutput, type1, type1_encoder);
  nn_ADD_FIELD_ENCODER(output_encoder, PokemonOutput, type2, type2_encoder);

  nn::input::InputEncoder<PokemonInput> input_encoder;
  auto hp_encoder = std::make_shared<nn::input::DoubleScaleEncoder>(1, 255, -1, 1);
  auto attack_encoder = std::make_shared<nn::input::DoubleScaleEncoder>(5, 190, -1, 1);
  auto defense_encoder = std::make_shared<nn::input::DoubleScaleEncoder>(5, 230, -1, 1);
  auto sp_attack_encoder = std::make_shared<nn::input::DoubleScaleEncoder>(10, 194, -1, 1);
  auto sp_defense_encoder = std::make_shared<nn::input::DoubleScaleEncoder>(20, 230, -1, 1);
  auto speed_encoder = std::make_shared<nn::input::DoubleScaleEncoder>(5, 180, -1, 1);

  nn_ADD_FIELD_ENCODER(input_encoder, PokemonInput, hp, hp_encoder);
  nn_ADD_FIELD_ENCODER(input_encoder, PokemonInput, attack, attack_encoder);
  nn_ADD_FIELD_ENCODER(input_encoder, PokemonInput, defense, defense_encoder);
  nn_ADD_FIELD_ENCODER(input_encoder, PokemonInput, sp_attack, sp_attack_encoder);
  nn_ADD_FIELD_ENCODER(input_encoder, PokemonInput, sp_defense, sp_defense_encoder);
  nn_ADD_FIELD_ENCODER(input_encoder, PokemonInput, speed, speed_encoder);

  const int BATCH_SIZE = 200;
  const int NUM_BATCHES = 4;

  nn::TrainingData<PokemonInput, PokemonOutput> td(BATCH_SIZE, NUM_BATCHES, input_encoder.Length(), output_encoder.Length(), &input_encoder, &output_encoder);

  for (int i = 0; i < ins.size(); ++i) {
    td.AddPair(ins[i], outs[i]);
  }


  nn::utility::Timer train_timer;

  auto hid_act = std::make_shared<nn::TanhActivation>();
  auto out_act = std::make_shared<nn::SigmoidActivation>(0, 1);
  auto err_function = std::make_shared<nn::CrossEntropyError>();

  nn::Network network({ input_encoder.Length(), 290, 230, output_encoder.Length() }, BATCH_SIZE, hid_act, out_act, err_function);
  nn::train::BackpropTrainingParameters params{ 0.0005, 0.5, 0, false, 10'000, 0.1 };

  //nn::ErrorStatistics<double> err_stats(10, network);
  nn::ErrorPrinter err_printer(25, network, &train_timer);

  //network.Attach(&err_stats);
  network.Attach(&err_printer);


  auto tr = std::make_unique<nn::train::BackpropTrainingAlgorithm>(network, params);

  tr->InitializeNetwork();
  tr->SetTrainingData(&td.Batches());

  train_timer.Start();
  tr->Train();
  train_timer.Stop();

  int fails = 0;

  for (auto& batch : td.Batches()) {
    const auto& output = network.FeedForward(batch.Input());
    const auto& target = batch.Output();

    PokemonOutput outp;
    PokemonOutput targp;

    for (int i = 0; i < output.Rows(); ++i) {
      output_encoder.Decode(output.GetRowValues(i), &outp);
      output_encoder.Decode(target.GetRowValues(i), &targp);
      if (outp.name == targp.name && outp.type1 == targp.type1 && outp.type2 == targp.type2) {
        std::cout << "     ";
      } else {
        std::cout << "FAIL ";
        ++fails;
      }
      std::cout << "Target: (" << targp << ") -> (" << outp << ")" << std::endl;
    }
  }

  std::cout << "Total time was " << train_timer.GetElapsedTimeAsString() << std::endl;
  std::cout << "Num fails: " << fails << std::endl;
}