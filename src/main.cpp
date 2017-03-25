#include "../examples/examples.h"

#include "matrix.hpp"
#include "network.hpp"

int
main(int argc, char *argv[])
{
  //IrisNetwork();
  for (int i = 0; i < 5; ++i)
    PokemonNetwork();

  //nn::dblmatrix A({{1, 2}, {3, 4}});
  //A.print();

  //nn::dblmatrix B(A);
  //B.print();
  //B.SetRowValues(0, {8, 9});
  //std::cout << std::endl;
  //B.print();
  //std::cout << std::endl;
  //A.print();

  nn::Matrix<double> A(2, 2);

  nn::InputLayer<float> in_layer{"blah", 10, 150};

}
