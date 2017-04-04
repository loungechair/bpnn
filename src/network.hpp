#pragma once

#include "matrix.hpp"
#include "activation.hpp"
#include "error.hpp"
#include "utility.hpp"
#include "errormsg.hpp"

#include <vector>
#include <map>
#include <memory>
#include <algorithm>

#include <iostream>
#include <iomanip>

#include <cassert>


namespace nn
{
class Connection;
class FeedForwardConnection;
class Layer;
class Network;
template <typename T, typename U> class TrainingData;

namespace train
{
class NetworkTrainer;
}


template <typename T>
class TrainingData2
{
public:
  bool HasActivation(double, const std::string&) const { return true; }
  const Matrix<T>& GetActivation(double, const std::string&) const { return Matrix<T>(0,0); }
};


template <typename T>
class ConnectionInterface
{
public:
  virtual void AccumulateNetInput(Matrix<T>& net_input) const = 0;

private:
};


template <typename T>
class LayerInterface
{
  CHECK_TEMPLATE_REAL_TYPE(T, LayerInterface);

public:
  virtual int BatchSize() const = 0;         // the size of the current batch
  virtual int MaxmimumBatchSize() const = 0; // the maximum batch size the layer can process

  virtual const Matrix<T>& GetActivation() const = 0;
  virtual T TotalError(const Matrix<T>& target_pattern) const = 0;
  virtual const std::string& GetName() const = 0;

  virtual void UpdateActivation(double time, const nn::TrainingData2<T>& td) = 0;
  virtual void CalculateActivationDerivative(Matrix<T>& activation_df) const = 0;

  void AddIncomingConnection(ConnectionInterface<T>* in) { incoming.push_back(in); }
  void AddOutgoingConnection(ConnectionInterface<T>* out) { outgoing.push_back(out); }

private:
  std::vector<ConnectionInterface<T>*> incoming;
  std::vector<ConnectionInterface<T>*> outgoing;
};


// Takes its activation from an input pattern.  If there's no values
// specified for the layer, then it returns 0.
template <typename T>
class InputLayer : LayerInterface<T>
{
public:
  InputLayer(const std::string& name_use, size_t size_use, size_t max_batch_size_use)
    : name(name_use),
      size(size_use),
      max_batch_size(max_batch_size_use),
      batch_size(0),
      zero_activation(size, 0.0),
      activation_ptr(&zero_activation)
  {}

  int BatchSize() const override { return batch_size; }
  int MaxmimumBatchSize() const override { return max_batch_size; }

  const Matrix<T>& GetActivation() const override { return *activation_ptr; }
  T TotalError(const Matrix<T>& target_pattern) const override { return 0; }
  const std::string& GetName() const override { return name; }

  virtual void UpdateActivation(double time, const nn::TrainingData2<T>& td) override
  {
    activation_ptr = nullptr;
    if (td.HasActivation(time, name)) {
      activation_ptr = &(td.GetActivation(time, name));
      batch_size = activation_ptr->Rows();

      if (activation_ptr->Cols() != size) {
        NonFatalError("invalid size of activation pattern");
      }
    }
  }

  virtual void CalculateActivationDerivative(Matrix<T>& activation_df) const
  {
    activation_df = zero_activation;
  }

private:
  std::string name;
  int size;
  int max_batch_size;
  int batch_size;
  Matrix<T> zero_activation;
  const Matrix<T>* activation_ptr;
};


template <typename T>
class ActivationLayer : LayerInterface<T>
{
public:
  //template <typename ActivationFunctionType, typename ErrorFunctionType>
  ActivationLayer(const std::string& name_use, size_t size_use, size_t max_batch_size_use)
    : name(name_use),
    batch_size(0),
    max_batch_size()
  {}

  int BatchSize() const override { return batch_size; }
  int MaxmimumBatchSize() const override { return max_batch_size; }

  const Matrix<T>& GetActivation() const override { return activation; }
  T TotalError(const Matrix<T>& target_pattern) const override;
  const std::string& GetName() const override;

  void UpdateActivation(double time, const nn::TrainingData2<T>& td) override
  {
    // copy bias in for each pattern
    for (int row = 0; row < activation.Rows(); ++row) {
      net_input.SetRowValues(row, bias);
    }

    for (auto& in_conn : incoming) {
      in_conn->AccumulateNetInput(net_input);
    }

    std::transform(begin(net_input), end(net_input), begin(activation),
                   [&](auto& x) { return activation_fn->f(x); });
  }

  void CalculateActivationDerivative(Matrix<T>& activation_df) const override
  {
    std::transform(begin(net_input), end(net_input), begin(activation), begin(activation_df),
                   [&](auto& net_in, act) { return activation_fn->df(net_in, act); });
  }

private:
  std::string name;

  ActivationFunction* activation_fn;
  ErrorFunction*      error_fn;

  size_t batch_size;
  size_t max_batch_size;

  Vector<T> bias;
  Matrix<T> net_input;
  Matrix<T> activation;
};


template <typename T>
class InputActivationLayer : public LayerInterface<T>
{
public:
  InputActivationLayer(const std::string& name_use, size_t size_use, size_t max_batch_size_use)
    : name(name_use),
      activation_layer(std::make_unique<ActivationLayer<T>>(name_use, size_use, max_batch_size_use)),
      input_layer(std::make_unique<InputLayer<T>>(name_use, size_use, max_batch_size_use)),
      current_layer(activation_layer.get())
  {}

  virtual int BatchSize() const override { return current_layer->BatchSize(); }
  virtual int MaxmimumBatchSize() const override { return current_layer->MaxmimumBatchSize(); }

  virtual const Matrix<T>& GetActivation() const override { return current_layer->GetActivation(); }

  virtual T TotalError(const Matrix<T>& target_pattern) const override
  {
    return current_layer->TotalError(target_pattern);
  }

  virtual const std::string& GetName() const { return name; }

  virtual void UpdateActivation(double time, const nn::TrainingData2<T>& td)
  {
    if (td.HasActivation(time, name)) {
      current_layer = input_layer.get();
    } else {
      current_layer = activation_layer.get();
    }
    current_layer->UpdateActivation(time, td);
  }

  virtual void CalculateActivationDerivative(Matrix<T>& activation_df) const
  {
    current_layer->CalculateActivationDerivative(activation_df);
  }

private:
  std::string name;
  std::unique_ptr<InputLayer<T>>      input_layer;
  std::unique_ptr<ActivationLayer<T>> activation_layer;

  LayerInterface<T>  current_layer;
};





class Layer
{
  friend train::NetworkTrainer;
  
public:

  Layer(int size_use, int batch_size_use, std::shared_ptr<ActivationFunction> activation_fn_use);

  void SetActivation(const dblmatrix& in) { activation = in; } // for input layers
  void CalculateActivation();                                  // for hidden layers

  int BatchSize() const { return batch_size; }

  const dblmatrix& GetActivation() const { return activation; }
  dblmatrix& GetActivation() { return activation; }

  dblscalar TotalError(const dblmatrix& target_pattern, const ErrorFunction* error_fn);

  int Size() const { return size; }

  auto GetActivationFunction() const { return activation_fn; }

  void AddIncomingConnection(Connection* in)  { incoming.push_back(in); }
  void AddOutgoingConnection(Connection* out) { outgoing.push_back(out); }

private:
  const int size;
  int batch_size;
  dblmatrix net_input;
  dblmatrix activation;
  dblvector bias;

  std::shared_ptr<ActivationFunction> activation_fn;

  std::vector<Connection *> incoming;
  std::vector<Connection *> outgoing;
};



class Connection
{
public:
  virtual int Rows() const = 0;
  virtual int Cols() const = 0;
  virtual int Size() const = 0;

  virtual void AccumulateNetInput(dblmatrix& net_input) const = 0;
  virtual dblmatrix& GetWeights() = 0;

  virtual Layer* LayerFrom() const = 0;
  virtual Layer* LayerTo() const = 0;
};



class ReferenceConnection : public Connection
{
public:
  ReferenceConnection(Connection* reference_conn_use, Layer* from, Layer* to);

  int Rows() const override { return connection->Rows(); }
  int Cols() const override { return connection->Cols(); }
  int Size() const override { return connection->Size(); }

  void AccumulateNetInput(dblmatrix& net_input) const override
  {
    return connection->AccumulateNetInput(net_input);
  }

  dblmatrix& GetWeights() override { return connection->GetWeights(); }

  Layer* LayerFrom() const override { return connection->LayerFrom(); }
  Layer* LayerTo() const override { return connection->LayerTo(); }

private:
  Connection* connection;
};



class FeedForwardConnection : public Connection
{
  friend train::NetworkTrainer;

public:
  FeedForwardConnection(Layer* from, Layer* to);

  int Rows() const override { return rows; }
  int Cols() const override { return cols; }
  int Size() const override { return size; }

  void AccumulateNetInput(dblmatrix& net_input) const override
  {
    nn::accum_A_BCt(net_input, layer_from->GetActivation(), weights);
  }

  dblmatrix& GetWeights() override { return weights; }

  Layer* LayerFrom() const override { return layer_from; }
  Layer* LayerTo() const override { return layer_to; }

private:
  Layer* layer_from;
  Layer* layer_to;
  
  int    rows;
  int    cols;
  int    size;

  dblmatrix weights;
};



class Network : public utility::Observable
{
  friend train::NetworkTrainer;

  static const int INPUT_LAYER = 0;
  
public:
  // Network(const std::string& file_name);        // load a network from a file

  Network(const std::vector<size_t>& layer_sizes,     // create a network with specified layer sizes
          int batch_size_use,
          std::shared_ptr<ActivationFunction> hid_act_fn,
          std::shared_ptr<ActivationFunction> out_act_fn,
          std::shared_ptr<ErrorFunction> err_function_use);

  Network() {} // create an empty network
  
  void AddLayer(size_t size, std::shared_ptr<ActivationFunction> act_fn)
  {
    layers.emplace_back(std::make_shared<Layer>(size, batch_size, act_fn));
  }

  //void AddLayer(const Layer* )

  int AddDefaultConnections();

  void AddConnection(size_t from, size_t to);


  dblmatrix FeedForward(const dblmatrix& input_pattern);
  dblscalar TotalError(const dblmatrix& target_pattern);

  int GetCurrentEpoch() const { return current_epoch; }
  double GetLastError() const { return last_error; }

private:
  int batch_size;
  int current_epoch;

  double last_error; // total error across all patterns from last call to FeedForward

  std::vector<std::shared_ptr<Layer>> layers;
  std::vector<std::shared_ptr<Connection>> connections;

  std::shared_ptr<ErrorFunction> err_function;

  void AddConnection(Layer* from, Layer* to);
};


class SimpleRecurrentNetwork
{
public:
  SimpleRecurrentNetwork(const std::vector<size_t>& layer_sizes,     // create a network with specified layer sizes
                         const std::vector<size_t>& context_layers,  // indices of layers that have context units
                         int batch_size_use,
                         std::shared_ptr<ActivationFunction> hid_act_fn,
                         std::shared_ptr<ActivationFunction> out_act_fn,
                         std::shared_ptr<ErrorFunction> err_function_use);


  void UnrollNetwork(size_t time_steps)
  {
    std::vector<size_t> layer_sizes;
    std::vector<size_t> context_layers;
    std::vector<FeedForwardConnection*> connections;
    std::shared_ptr<ActivationFunction> hid_act_fn;
    std::shared_ptr<ActivationFunction> out_act_fn;

    int num_hid = layer_sizes.size() - 1;

    for (int layer = 0; layer < num_hid; ++layer) {
      ff_network.AddLayer(layer_sizes[layer], hid_act_fn);
    }
    ff_network.AddLayer(layer_sizes[num_hid], out_act_fn);

    // add connections at time i
    for (int layer = 1; layer < layer_sizes.size(); ++layer) {
      ff_network.AddConnection(GetLayerIndex(0, layer - 1), GetLayerIndex(0, layer));
    }

    for (int time = 1; time < time_steps; ++time) {
      // add layers at time i
      for (int layer = 0; layer < num_hid; ++layer) {
        ff_network.AddLayer(layer_sizes[layer], hid_act_fn);
      }
      ff_network.AddLayer(layer_sizes[num_hid], out_act_fn);

      // add connections at time i
      for (int layer = 1; layer < layer_sizes.size(); ++layer) {
        ff_network.AddConnection(GetLayerIndex(time, layer-1), GetLayerIndex(time, layer));
      }

      // add context connections from time i - 1 to i
      for (auto& clayer : context_layers) {
        ff_network.AddConnection(GetLayerIndex(time - 1, clayer), GetLayerIndex(time, clayer));
      }
    }
  }

private:
  Network ff_network;

  int num_layers;
  int num_time_steps;

  size_t GetLayerIndex(size_t time, size_t layer_num)
  {
    return time * num_layers + layer_num;
  }
};


template <typename T>
class ErrorStatistics : public utility::Observer
{
public:
  ErrorStatistics(int save_freq_use, const Network& network_use)
    : save_frequency(save_freq_use),
    network(network_use)
  {}

  void UpdateBatch() override
  {
    int epoch = network.GetCurrentEpoch();
    AccumulateErrorStatistics(epoch);
  }

  void UpdateEpoch() override {}

  void AccumulateErrorStatistics(int epoch)
  {
    if ((epoch == 0) || (epoch % save_frequency == 0)) {
      total_error[epoch] += network.GetLastError();
    }
  }

  T GetTotalError(int epoch) const { return total_error[epoch]; }

  auto begin() { return total_error.begin(); }
  auto end() { return total_error.end(); }
  auto begin() const { return total_error.begin(); }
  auto end() const { return total_error.end(); }

private:
  const Network& network;
  int save_frequency; // how many epochs between saving data.  set to 1 to save all the time
  std::map<int, T> total_error;
};


class ErrorPrinter : public utility::Observer
{
public:
  ErrorPrinter(int save_freq_use, const Network& network_use, const utility::Timer* timer_use = nullptr)
    : save_frequency(save_freq_use),
      total_error(0.0),
      network(network_use),
      timer(timer_use)
  {}

  void UpdateBatch() override
  {
    total_error += network.GetLastError();
  }

  void UpdateEpoch() override
  {
    int epoch = network.GetCurrentEpoch();
    if ((epoch == 0) || (epoch % save_frequency == 0)) {
      std::cout << std::setw(6) << epoch << ' ';
      std::cout << std::setw(17) << std::setprecision(4) << std::fixed << total_error;
      if (timer) {
        std::cout << "   " << timer->GetElapsedTimeAsString();
      }
      std::cout << std::endl;
    }
    total_error = 0.0;
  }

private:
  int save_frequency; // how many epochs between saving data.  set to 1 to save all the time
  double total_error;
  const Network& network;
  const utility::Timer* timer;
};


}
