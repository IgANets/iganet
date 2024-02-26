/**
   @file include/iganet.hpp

   @brief Isogeometric analysis network

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <any>

#include <boundary.hpp>
#include <functionspace.hpp>
#include <layer.hpp>
#include <utils/concat.hpp>
#include <utils/fqn.hpp>
#include <utils/zip.hpp>

namespace iganet {

/// @brief Enumerator for the status of the various data
enum class status : short_t {
  none = 0,                  /*!< nothing needs update  */
  inputs = 1 << 0,           /*!< inputs need update  */
  geometryMap_collPts = 1 << 1, /*!< geometry map collocation points need update */
  variable_collPts = 1 << 2  /*!< variable collocation points need update */
};

  /// @brief Enumerator for the collocation point specifier
  enum class collPts : short_t {
    greville = 0, /*!< Greville points */
    greville_interior = 1 /*!< Greville points in the interior */
  };
  
/// @brief Returns the sum of two status objects
status operator+(enum status lhs, enum status rhs) {
  return status(static_cast<short_t>(lhs) + static_cast<short_t>(rhs));
}

/// @brief Returns true if flag is set in the given status
bool operator&(enum status status, enum status flag) {
  return bool(static_cast<short_t>(status) & static_cast<short_t>(flag));
}

/// @brief IgANetOptions
struct IgANetOptions {
  TORCH_ARG(int64_t, max_epoch) = 100;
  TORCH_ARG(int64_t, batch_size) = 1000;
  TORCH_ARG(double, min_loss) = 1e-4;
};

/// @brief IgANetGeneratorImpl
///
/// @note Following the discussion of module overship here
///
/// https://pytorch.org/tutorials/advanced/cpp_frontend.html#module-ownership
///
/// we implement a generator implementation class following
///
/// https://pytorch.org/tutorials/advanced/cpp_frontend.html#the-generator-module
template <typename real_t>
class IgANetGeneratorImpl : public torch::nn::Module {
public:
  /// @brief Default constructor
  IgANetGeneratorImpl() = default;

  /// @brief Constructor
  explicit IgANetGeneratorImpl(
      const std::vector<int64_t> &layers,
      const std::vector<std::vector<std::any>> &activations) {
    assert(layers.size() == activations.size() + 1);

    // Generate vector of linear layers and register them as layer[i]
    for (auto i = 0; i < layers.size() - 1; ++i) {
      layers_.emplace_back(
          register_module("layer[" + std::to_string(i) + "]",
                          torch::nn::Linear(layers[i], layers[i + 1])));
      layers_.back()->to(dtype<real_t>());

      torch::nn::init::xavier_uniform_(layers_.back()->weight);
      torch::nn::init::constant_(layers_.back()->bias, 0.0);
    }

    // Generate vector of activation functions
    for (const auto &a : activations)
      switch (std::any_cast<activation>(a[0])) {
        // No activation function
      case activation::none:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new None{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Batch Normalization
      case activation::batch_norm:
        switch (a.size()) {
        case 8:
          activations_.emplace_back(new BatchNorm{
              std::any_cast<torch::Tensor>(a[1]),
              std::any_cast<torch::Tensor>(a[2]),
              std::any_cast<torch::Tensor>(a[3]),
              std::any_cast<torch::Tensor>(a[4]), std::any_cast<double>(a[5]),
              std::any_cast<double>(a[6]), std::any_cast<bool>(a[7])});
          break;
        case 7:
          activations_.emplace_back(new BatchNorm{
              std::any_cast<torch::Tensor>(a[1]),
              std::any_cast<torch::Tensor>(a[2]),
              std::any_cast<torch::Tensor>(a[3]),
              std::any_cast<torch::Tensor>(a[4]), std::any_cast<double>(a[5]),
              std::any_cast<double>(a[6])});
          break;
        case 4:
          activations_.emplace_back(new BatchNorm{
              std::any_cast<torch::Tensor>(a[1]),
              std::any_cast<torch::Tensor>(a[2]),
              std::any_cast<torch::nn::functional::BatchNormFuncOptions>(
                  a[3])});
          break;
        case 3:
          activations_.emplace_back(
              new BatchNorm{std::any_cast<torch::Tensor>(a[1]),
                            std::any_cast<torch::Tensor>(a[2])});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // CELU
      case activation::celu:
        switch (a.size()) {
        case 3:
          activations_.emplace_back(
              new CELU{std::any_cast<double>(a[1]), std::any_cast<bool>(a[2])});
          break;
        case 2:
          try {
            activations_.emplace_back(new CELU{
                std::any_cast<torch::nn::functional::CELUFuncOptions>(a[1])});
          } catch (...) {
            activations_.emplace_back(new CELU{std::any_cast<double>(a[1])});
          }
          break;
        case 1:
          activations_.emplace_back(new CELU{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // ELU
      case activation::elu:
        switch (a.size()) {
        case 3:
          activations_.emplace_back(
              new ELU{std::any_cast<double>(a[1]), std::any_cast<bool>(a[2])});
          break;
        case 2:
          try {
            activations_.emplace_back(new ELU{
                std::any_cast<torch::nn::functional::ELUFuncOptions>(a[1])});
          } catch (...) {
            activations_.emplace_back(new ELU{std::any_cast<double>(a[1])});
          }
          break;
        case 1:
          activations_.emplace_back(new ELU{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // GELU
      case activation::gelu:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new GELU{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // GLU
      case activation::glu:
        switch (a.size()) {
        case 2:
          try {
            activations_.emplace_back(new GLU{
                std::any_cast<torch::nn::functional::GLUFuncOptions>(a[1])});
          } catch (...) {
            activations_.emplace_back(new GLU{std::any_cast<int64_t>(a[1])});
          }
          break;
        case 1:
          activations_.emplace_back(new GLU{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Group Normalization
      case activation::group_norm:
        switch (a.size()) {
        case 5:
          activations_.emplace_back(new GroupNorm{
              std::any_cast<int64_t>(a[1]), std::any_cast<torch::Tensor>(a[2]),
              std::any_cast<torch::Tensor>(a[3]), std::any_cast<double>(a[4])});
          break;
        case 2:
          try {
            activations_.emplace_back(new GroupNorm{
                std::any_cast<torch::nn::functional::GroupNormFuncOptions>(
                    a[1])});
          } catch (...) {
            activations_.emplace_back(
                new GroupNorm{std::any_cast<int64_t>(a[1])});
          }
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Gumbel-Softmax
      case activation::gumbel_softmax:
        switch (a.size()) {
        case 4:
          activations_.emplace_back(new GumbelSoftmax{
              std::any_cast<double>(a[1]), std::any_cast<int>(a[2]),
              std::any_cast<bool>(a[3])});
          break;
        case 2:
          activations_.emplace_back(new GumbelSoftmax{
              std::any_cast<torch::nn::functional::GumbelSoftmaxFuncOptions>(
                  a[1])});
          break;
        case 1:
          activations_.emplace_back(new GumbelSoftmax{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Hard shrinkish
      case activation::hardshrink:
        switch (a.size()) {
        case 2:
          try {
            activations_.emplace_back(new Hardshrink{
                std::any_cast<torch::nn::functional::HardshrinkFuncOptions>(
                    a[1])});
          } catch (...) {
            activations_.emplace_back(
                new Hardshrink{std::any_cast<double>(a[1])});
          }
          break;
        case 1:
          activations_.emplace_back(new Hardshrink{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Hardsigmoid
      case activation::hardsigmoid:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new Hardsigmoid{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Hardswish
      case activation::hardswish:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new Hardswish{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Hardtanh
      case activation::hardtanh:
        switch (a.size()) {
        case 4:
          activations_.emplace_back(new Hardtanh{std::any_cast<double>(a[1]),
                                                 std::any_cast<double>(a[2]),
                                                 std::any_cast<bool>(a[3])});
          break;
        case 3:
          activations_.emplace_back(new Hardtanh{std::any_cast<double>(a[1]),
                                                 std::any_cast<double>(a[2])});
          break;
        case 2:
          activations_.emplace_back(new Hardtanh{
              std::any_cast<torch::nn::functional::HardtanhFuncOptions>(a[1])});
          break;
        case 1:
          activations_.emplace_back(new Hardtanh{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Instance Normalization
      case activation::instance_norm:
        switch (a.size()) {
        case 8:
          activations_.emplace_back(new InstanceNorm{
              std::any_cast<torch::Tensor>(a[1]),
              std::any_cast<torch::Tensor>(a[2]),
              std::any_cast<torch::Tensor>(a[3]),
              std::any_cast<torch::Tensor>(a[4]), std::any_cast<double>(a[5]),
              std::any_cast<double>(a[6]), std::any_cast<bool>(a[7])});
          break;
        case 7:
          activations_.emplace_back(new InstanceNorm{
              std::any_cast<torch::Tensor>(a[1]),
              std::any_cast<torch::Tensor>(a[2]),
              std::any_cast<torch::Tensor>(a[3]),
              std::any_cast<torch::Tensor>(a[4]), std::any_cast<double>(a[5]),
              std::any_cast<double>(a[6])});
          break;
        case 2:
          activations_.emplace_back(new InstanceNorm{
              std::any_cast<torch::nn::functional::InstanceNormFuncOptions>(
                  a[1])});
          break;
        case 1:
          activations_.emplace_back(new InstanceNorm{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Layer Normalization
      case activation::layer_norm:
        switch (a.size()) {
        case 5:
          activations_.emplace_back(new LayerNorm{
              std::any_cast<std::vector<int64_t>>(a[1]),
              std::any_cast<torch::Tensor>(a[2]),
              std::any_cast<torch::Tensor>(a[3]), std::any_cast<double>(a[4])});
          break;
        case 2:
          try {
            activations_.emplace_back(new LayerNorm{
                std::any_cast<torch::nn::functional::LayerNormFuncOptions>(
                    a[1])});
          } catch (...) {
            activations_.emplace_back(
                new LayerNorm{std::any_cast<std::vector<int64_t>>(a[1])});
          }
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Leaky ReLU
      case activation::leaky_relu:
        switch (a.size()) {
        case 3:
          activations_.emplace_back(new LeakyReLU{std::any_cast<double>(a[1]),
                                                  std::any_cast<bool>(a[2])});
          break;
        case 2:
          try {
            activations_.emplace_back(new LeakyReLU{
                std::any_cast<torch::nn::functional::LeakyReLUFuncOptions>(
                    a[1])});
          } catch (...) {
            activations_.emplace_back(
                new LeakyReLU{std::any_cast<double>(a[1])});
          }
          break;
        case 1:
          activations_.emplace_back(new LeakyReLU{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Local response Normalization
      case activation::local_response_norm:
        switch (a.size()) {
        case 5:
          activations_.emplace_back(new LocalResponseNorm{
              std::any_cast<int64_t>(a[1]), std::any_cast<double>(a[2]),
              std::any_cast<double>(a[3]), std::any_cast<double>(a[4])});
          break;
        case 2:
          try {
            activations_.emplace_back(new LocalResponseNorm{std::any_cast<
                torch::nn::functional::LocalResponseNormFuncOptions>(a[1])});
          } catch (...) {
            activations_.emplace_back(
                new LocalResponseNorm{std::any_cast<int64_t>(a[1])});
          }
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // LogSigmoid
      case activation::logsigmoid:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new LogSigmoid{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // LogSoftmax
      case activation::logsoftmax:
        switch (a.size()) {
        case 2:
          try {
            activations_.emplace_back(new LogSoftmax{
                std::any_cast<torch::nn::functional::LogSoftmaxFuncOptions>(
                    a[1])});
          } catch (...) {
            activations_.emplace_back(
                new LogSoftmax{std::any_cast<int64_t>(a[1])});
          }
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Mish
      case activation::mish:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new Mish{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Lp Normalization
      case activation::normalize:
        switch (a.size()) {
        case 4:
          activations_.emplace_back(new Normalize{
              std::any_cast<double>(a[1]), std::any_cast<double>(a[2]),
              std::any_cast<int64_t>(a[3])});
          break;
        case 2:
          activations_.emplace_back(new Normalize{
              std::any_cast<torch::nn::functional::NormalizeFuncOptions>(
                  a[1])});
          break;
        case 1:
          activations_.emplace_back(new Normalize{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // PReLU
      case activation::prelu:
        switch (a.size()) {
        case 2:
          activations_.emplace_back(
              new PReLU{std::any_cast<torch::Tensor>(a[1])});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // ReLU
      case activation::relu:
        switch (a.size()) {
        case 2:
          try {
            activations_.emplace_back(new ReLU{
                std::any_cast<torch::nn::functional::ReLUFuncOptions>(a[1])});
          } catch (...) {
            activations_.emplace_back(new ReLU{std::any_cast<bool>(a[1])});
          }
          break;
        case 1:
          activations_.emplace_back(new ReLU{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Relu6
      case activation::relu6:
        switch (a.size()) {
        case 2:
          try {
            activations_.emplace_back(new ReLU6{
                std::any_cast<torch::nn::functional::ReLU6FuncOptions>(a[1])});
          } catch (...) {
            activations_.emplace_back(new ReLU6{std::any_cast<bool>(a[1])});
          }
          break;
        case 1:
          activations_.emplace_back(new ReLU6{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Randomized ReLU
      case activation::rrelu:
        switch (a.size()) {
        case 4:
          activations_.emplace_back(new RReLU{std::any_cast<double>(a[1]),
                                              std::any_cast<double>(a[2]),
                                              std::any_cast<bool>(a[3])});
          break;
        case 3:
          activations_.emplace_back(new RReLU{std::any_cast<double>(a[1]),
                                              std::any_cast<double>(a[2])});
          break;
        case 2:
          activations_.emplace_back(new RReLU{
              std::any_cast<torch::nn::functional::RReLUFuncOptions>(a[1])});
          break;
        case 1:
          activations_.emplace_back(new RReLU{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // SELU
      case activation::selu:
        switch (a.size()) {
        case 2:
          try {
            activations_.emplace_back(new SELU{
                std::any_cast<torch::nn::functional::SELUFuncOptions>(a[1])});
          } catch (...) {
            activations_.emplace_back(new SELU{std::any_cast<bool>(a[1])});
          }
          break;
        case 1:
          activations_.emplace_back(new SELU{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Sigmoid
      case activation::sigmoid:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new Sigmoid{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // SiLU
      case activation::silu:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new SiLU{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Softmax
      case activation::softmax:
        switch (a.size()) {
        case 2:
          try {
            activations_.emplace_back(new Softmax{
                std::any_cast<torch::nn::functional::SoftmaxFuncOptions>(
                    a[1])});
          } catch (...) {
            activations_.emplace_back(
                new Softmax{std::any_cast<int64_t>(a[1])});
          }
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Softmin
      case activation::softmin:
        switch (a.size()) {
        case 2:
          try {
            activations_.emplace_back(new Softmin{
                std::any_cast<torch::nn::functional::SoftminFuncOptions>(
                    a[1])});
          } catch (...) {
            activations_.emplace_back(
                new Softmin{std::any_cast<int64_t>(a[1])});
          }
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Softplus
      case activation::softplus:
        switch (a.size()) {
        case 3:
          activations_.emplace_back(new Softplus{std::any_cast<double>(a[1]),
                                                 std::any_cast<double>(a[2])});
          break;
        case 2:
          activations_.emplace_back(new Softplus{
              std::any_cast<torch::nn::functional::SoftplusFuncOptions>(a[1])});
          break;
        case 1:
          activations_.emplace_back(new Softplus{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Softshrink
      case activation::softshrink:
        switch (a.size()) {
        case 2:
          try {
            activations_.emplace_back(new Softshrink{
                std::any_cast<torch::nn::functional::SoftshrinkFuncOptions>(
                    a[1])});
          } catch (...) {
            activations_.emplace_back(
                new Softshrink{std::any_cast<double>(a[1])});
          }
          break;
        case 1:
          activations_.emplace_back(new Softshrink{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Softsign
      case activation::softsign:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new Softsign{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Tanh
      case activation::tanh:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new Tanh{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Tanhshrink
      case activation::tanhshrink:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new Tanhshrink{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Threshold
      case activation::threshold:
        switch (a.size()) {
        case 4:
          activations_.emplace_back(new Threshold{std::any_cast<double>(a[1]),
                                                  std::any_cast<double>(a[2]),
                                                  std::any_cast<bool>(a[3])});
          break;
        case 3:
          activations_.emplace_back(new Threshold{std::any_cast<double>(a[1]),
                                                  std::any_cast<double>(a[2])});
          break;
        case 2:
          activations_.emplace_back(new Threshold{
              std::any_cast<torch::nn::functional::ThresholdFuncOptions>(
                  a[1])});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

      default:
        throw std::runtime_error("Invalid activation function");
      }
  }

  /// @brief Forward evaluation
  torch::Tensor forward(torch::Tensor x) {
    torch::Tensor x_in = x.clone();

    // Standard feed-forward neural network
    for (auto [layer, activation] : utils::zip(layers_, activations_))
      x = activation->apply(layer->forward(x));

    // Skip connections
    // x = torch::where(torch::linspace(1, x.size(0), x.size(0)) <= 7,
    // x_in.index({torch::indexing::Slice(147, torch::indexing::None)}),
    // x);

    // x.view({7, 7}).index_put_({"...", 0},
    //                           x_in.index({torch::indexing::Slice(147, 154)}));
    // x.view({7, 7}).index_put_({"...", -1},
    //                           x_in.index({torch::indexing::Slice(154, 161)}));
    // x.view({7, 7}).index_put_({0, "..."},
    //                           x_in.index({torch::indexing::Slice(161, 168)}));
    // x.view({7, 7}).index_put_({-1, "..."},
    //                           x_in.index({torch::indexing::Slice(168, 175)}));

    // std::cout << x_in.index({torch::indexing::Slice(147,
    // torch::indexing::None)});

    return x;
  }

  /// @brief Writes the IgANet into a torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "iganet") const {
    assert(layers_.size() == activations_.size());

    archive.write(key + ".layers", torch::full({1}, (int64_t)layers_.size()));
    for (std::size_t i = 0; i < layers_.size(); ++i) {
      archive.write(
          key + ".layer[" + std::to_string(i) + "].in_features",
          torch::full({1}, (int64_t)layers_[i]->options.in_features()));
      archive.write(
          key + ".layer[" + std::to_string(i) + "].outputs_features",
          torch::full({1}, (int64_t)layers_[i]->options.out_features()));
      archive.write(key + ".layer[" + std::to_string(i) + "].bias",
                    torch::full({1}, (int64_t)layers_[i]->options.bias()));

      activations_[i]->write(archive, key + ".layer[" + std::to_string(i) +
                                          "].activation");
    }

    return archive;
  }

  /// @brief Reads the IgANet from a torch::serialize::InputArchive object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "iganet") {
    torch::Tensor layers, in_features, outputs_features, bias, activation;

    archive.read(key + ".layers", layers);
    for (int64_t i = 0; i < layers.item<int64_t>(); ++i) {
      archive.read(key + ".layer[" + std::to_string(i) + "].in_features",
                   in_features);
      archive.read(key + ".layer[" + std::to_string(i) + "].outputs_features",
                   outputs_features);
      archive.read(key + ".layer[" + std::to_string(i) + "].bias", bias);
      layers_.emplace_back(register_module(
          "layer[" + std::to_string(i) + "]",
          torch::nn::Linear(
              torch::nn::LinearOptions(in_features.item<int64_t>(),
                                       outputs_features.item<int64_t>())
                  .bias(bias.item<bool>()))));

      archive.read(key + ".layer[" + std::to_string(i) + "].activation.type",
                   activation);
      switch (static_cast<enum activation>(activation.item<int64_t>())) {
      case activation::none:
        activations_.emplace_back(new None{});
        break;
      case activation::batch_norm:
        activations_.emplace_back(
            new BatchNorm{torch::Tensor{}, torch::Tensor{}});
        break;
      case activation::celu:
        activations_.emplace_back(new CELU{});
        break;
      case activation::elu:
        activations_.emplace_back(new ELU{});
        break;
      case activation::gelu:
        activations_.emplace_back(new GELU{});
        break;
      case activation::glu:
        activations_.emplace_back(new GLU{});
        break;
      case activation::group_norm:
        activations_.emplace_back(new GroupNorm{0});
        break;
      case activation::gumbel_softmax:
        activations_.emplace_back(new GumbelSoftmax{});
        break;
      case activation::hardshrink:
        activations_.emplace_back(new Hardshrink{});
        break;
      case activation::hardsigmoid:
        activations_.emplace_back(new Hardsigmoid{});
        break;
      case activation::hardswish:
        activations_.emplace_back(new Hardswish{});
        break;
      case activation::hardtanh:
        activations_.emplace_back(new Hardtanh{});
        break;
      case activation::instance_norm:
        activations_.emplace_back(new InstanceNorm{});
        break;
      case activation::layer_norm:
        activations_.emplace_back(new LayerNorm{{}});
        break;
      case activation::leaky_relu:
        activations_.emplace_back(new LeakyReLU{});
        break;
      case activation::local_response_norm:
        activations_.emplace_back(new LocalResponseNorm{0});
        break;
      case activation::logsigmoid:
        activations_.emplace_back(new LogSigmoid{});
        break;
      case activation::logsoftmax:
        activations_.emplace_back(new LogSoftmax{0});
        break;
      case activation::mish:
        activations_.emplace_back(new Mish{});
        break;
      case activation::normalize:
        activations_.emplace_back(new Normalize{0, 0, 0});
        break;
      case activation::prelu:
        activations_.emplace_back(new PReLU{torch::Tensor{}});
        break;
      case activation::relu:
        activations_.emplace_back(new ReLU{});
        break;
      case activation::relu6:
        activations_.emplace_back(new ReLU6{});
        break;
      case activation::rrelu:
        activations_.emplace_back(new RReLU{});
        break;
      case activation::selu:
        activations_.emplace_back(new SELU{});
        break;
      case activation::sigmoid:
        activations_.emplace_back(new Sigmoid{});
        break;
      case activation::silu:
        activations_.emplace_back(new SiLU{});
        break;
      case activation::softmax:
        activations_.emplace_back(new Softmax{0});
        break;
      case activation::softmin:
        activations_.emplace_back(new Softmin{0});
        break;
      case activation::softplus:
        activations_.emplace_back(new Softplus{});
        break;
      case activation::softshrink:
        activations_.emplace_back(new Softshrink{});
        break;
      case activation::softsign:
        activations_.emplace_back(new Softsign{});
        break;
      case activation::tanh:
        activations_.emplace_back(new Tanh{});
        break;
      case activation::tanhshrink:
        activations_.emplace_back(new Tanhshrink{});
        break;
      case activation::threshold:
        activations_.emplace_back(new Threshold{0, 0});
        break;
      default:
        throw std::runtime_error("Invalid activation function");
      }
      activations_.back()->read(archive, key + ".layer[" + std::to_string(i) +
                                             "].activation");
    }
    return archive;
  }

  inline virtual void
  pretty_print(std::ostream &os = std::cout) const noexcept override {
    os << "(\n";

    int i = 0;
    for (const auto &activation : activations_)
      os << "activation[" << i++ << "]: " << *activation << "\n";
    os << ")\n";
  }

private:
  /// @brief Vector of linear layers
  std::vector<torch::nn::Linear> layers_;

  /// @brief Vector of activation functions
  std::vector<std::unique_ptr<iganet::ActivationFunction>> activations_;
};

/// @brief IgANetGenerator
///
/// @note: This class is normally generated by the TORCH_MODULE
/// macro. Since the latter cannot handle templated classes
/// correctly, we give the implementation explicitly
template <typename real_t>
class IgANetGenerator
    : public torch::nn::ModuleHolder<IgANetGeneratorImpl<real_t>> {

public:
  using torch::nn::ModuleHolder<IgANetGeneratorImpl<real_t>>::ModuleHolder;
  using Impl = IgANetGeneratorImpl<real_t>;
};

/// @brief IgANet
///
/// This class implements the core functionality of IgANets
template <typename Optimizer, typename GeometryMap, typename Variable>
class IgANet : public utils::Serializable, private utils::FullQualifiedName {
public:
  /// @brief Value type
  using value_type =
      typename std::common_type<typename GeometryMap::value_type,
                                typename Variable::value_type>::type;

  /// @brief Type of the optimizer
  using optimizer_type = Optimizer;
  
  /// @brief Type of the geometry map function space(s)
  using geometryMap_type = GeometryMap;

  /// @brief Type of the variable function space(s)
  using variable_type = Variable;
  
  /// @brief Type of the geometry map collocation points 
  using geometryMap_collPts_type =
      std::pair<typename GeometryMap::eval_type,
                typename GeometryMap::boundary_eval_type>;

  /// @brief Type of the variable collocation points
  using variable_collPts_type =
      std::pair<typename Variable::eval_type,
                typename Variable::boundary_eval_type>;

protected:
  /// @brief Spline representation of the geometry map
  GeometryMap G_;

  /// @brief Spline representation of the reference data
  Variable f_;

  /// @brief Spline representation of the solution
  Variable u_;

  /// @brief IgANet generator
  IgANetGenerator<value_type> net_;

  /// @brief Optimizer
  Optimizer opt_;

  /// @brief Options
  IgANetOptions options_;

  /// @brief Specifier for collocation points of the geometry map
  collPts geometryMap_collPts_;

  /// @brief Specifier for collocation points of the variables
  collPts variable_collPts_;

  /// @brief Constructor: number of layers, activation functions,
  /// and number of spline coefficients (different for Geometry
  /// and Variable types)
  template <typename... GeometryMapSplines, size_t... Is,
            typename... VariableSplines, size_t... Js>
  IgANet(const std::vector<int64_t> &layers,
         const std::vector<std::vector<std::any>> &activations,
         std::tuple<GeometryMapSplines...> geometryMap_splines,
         std::index_sequence<Is...>,
         std::tuple<VariableSplines...> variable_splines,
         std::index_sequence<Js...>, IgANetOptions defaults = {},
         iganet::Options<value_type> options = iganet::Options<value_type>{})
      : // Construct the different spline objects individually
        G_(std::get<Is>(geometryMap_splines)..., init::greville, options),
        f_(std::get<Js>(variable_splines)..., init::zeros, options),
        u_(std::get<Js>(variable_splines)..., init::random, options),

        // Construct the deep neural network
        net_(utils::concat(
                 std::vector<int64_t>{inputs(/* epoch */ 0).size(0)}, layers,
                 std::vector<int64_t>{u_.as_tensor_size(false)}),
             activations),

        // Construct the optimizer
        opt_(net_->parameters()),

        // Set options
        options_(defaults),

        // Set collocation point specifiers
        geometryMap_collPts_(collPts::greville),
        variable_collPts_(collPts::greville)
  {}

public:
  /// @brief Default constructor
  explicit IgANet(
      IgANetOptions defaults = {},
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : G_(), f_(), u_(), opt_(net_->parameters()),
        options_(defaults), geometryMap_collPts_(collPts::greville), variable_collPts_(collPts::greville) {}

  /// @brief Constructor: number of layers, activation functions, and
  /// number of spline coefficients (same for geometry map and
  /// variables)
  template <typename... Splines>
  IgANet(const std::vector<int64_t> &layers,
         const std::vector<std::vector<std::any>> &activations,
         std::tuple<Splines...> splines, IgANetOptions defaults = {},
         iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgANet(layers, activations, splines, splines, defaults, options) {}

  /// @brief Constructor: number of layers, activation functions, and
  /// number of spline coefficients (different for geometry map and
  /// variables)
  template <typename... GeometryMapSplines, typename... VariableSplines>
  IgANet(const std::vector<int64_t> &layers,
         const std::vector<std::vector<std::any>> &activations,
         std::tuple<GeometryMapSplines...> geometryMap_splines,
         std::tuple<VariableSplines...> variable_splines,
         IgANetOptions defaults = {},
         iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgANet(layers, activations, geometryMap_splines,
               std::make_index_sequence<sizeof...(GeometryMapSplines)>{},
               variable_splines,
               std::make_index_sequence<sizeof...(VariableSplines)>{}, defaults,
               options) {}

  /// @brief Returns a constant reference to the IgANet generator
  inline const IgANetGenerator<value_type> &net() const { return net_; }

  /// @brief Returns a non-constant reference to the IgANet generator
  inline IgANetGenerator<value_type> &net() { return net_; }

  /// @brief Returns a constant reference to the optimizer
  inline const Optimizer &opt() const { return opt_; }

  /// @brief Returns a non-constant reference to the optimizer
  inline Optimizer &opt() { return opt_; }

  /// @brief Returns a constant reference to the spline
  /// representation of the geometry map
  inline const GeometryMap &G() const { return G_; }

  /// @brief Returns a non-constant reference to the spline
  /// representation of the geometry map
  inline GeometryMap &G() { return G_; }

  /// @brief Returns a constant reference to the spline
  /// representation of the reference data
  inline const Variable &f() const { return f_; }

  /// @brief Returns a non-constant reference to the spline
  /// representation of the reference data
  inline Variable &f() { return f_; }

  /// @brief Returns a constant reference to the spline
  /// representation of the solution
  inline const Variable &u() const { return u_; }

  /// @brief Returns a non-constant reference to the spline
  /// representation of the solution
  inline Variable &u() { return u_; }

  /// @brief Returns a constant reference to the options structure
  inline const auto &options() const { return options_; }

  /// @brief Returns a non-constant reference to the options structure
  inline auto &options() { return options_; }

  /// @brief Sets the collocation point specifiers for the geometry
  /// map and the variables and returns the specifier
  inline enum collPts collPts(enum collPts collPts ) {
    geometryMap_collPts_ = collPts;
    variable_collPts_ = collPts;
    return collPts;
  }

  /// @brief Sets the collocation point specifier for the geometry map
  /// and returns the specifier
  inline enum collPts geometryMap_collPts(enum collPts collPts) {
    geometryMap_collPts_ = collPts;
    return collPts;
  }

  /// @brief Sets the collocation point specifier for the variables
  /// and returns the specifier
  inline enum collPts variable_collPts(enum collPts collPts) {
    variable_collPts_ = collPts;
    return collPts;
  }

  /// @brief Returns the collocation point specifier for the geometry map
  inline enum collPts geometryMap_collPts() const {
    return geometryMap_collPts_;
  }

  /// @brief Returns the collocation point specifier for the variables
  inline enum collPts variable_collPts() const {
    return variable_collPts_;
  }
    
private:
  /// @brief Returns the geometry map collocation points
  ///
  /// In the default implementation the collocation points are the Greville
  /// abscissae in the interior of the domain and on the boundary
  /// faces. This behavior can be changed by overriding this virtual
  /// function in a derived class.
  template <size_t... Is>
  geometryMap_collPts_type geometryMap_collPts(std::index_sequence<Is...>) const {
    geometryMap_collPts_type collPts;

    switch (geometryMap_collPts_) {

    case collPts::greville:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts.first) =
        std::get<Is>(G_).greville(/* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) =
        std::get<Is>(G_.boundary()).greville()),
       ...);
      break;
      
    case collPts::greville_interior:
      // Get Greville abscissae inside the domain
      ((std::get<Is>(collPts.first) =
        std::get<Is>(G_).greville(/* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) =
        std::get<Is>(G_.boundary()).greville()),
       ...);
      break;

    default:
      throw std::runtime_error("Invalid collocation point specifier");
    }
    
    return collPts;
  }

  /// @brief Returns the variable collocation points
  ///
  /// In the default implementation the collocation points are the Greville
  /// abscissae in the interior of the domain and on the boundary
  /// faces. This behavior can be changed by overriding this virtual
  /// function in a derived class.
  template <size_t... Is>
  variable_collPts_type variable_collPts(std::index_sequence<Is...>) const {
    variable_collPts_type collPts;

    switch (variable_collPts_) {

    case collPts::greville:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts.first) =
        std::get<Is>(f_).greville(/* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) =
        std::get<Is>(f_.boundary()).greville()),
       ...);
      break;
      
    case collPts::greville_interior:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts.first) =
        std::get<Is>(f_).greville(/* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) =
        std::get<Is>(f_.boundary()).greville()),
       ...);
      break;

    default:
      throw std::runtime_error("Invalid collocation point specifier");
    }
    
    return collPts;
  }

public:
  /// @brief Returns the geometry map collocation points
  ///
  /// In the default implementation the collocation points are the Greville
  /// abscissae in the interior of the domain and on the boundary
  /// faces. This behavior can be changed by overriding this virtual
  /// function in a derived class.
  virtual geometryMap_collPts_type geometryMap_collPts(int64_t epoch) const {
    if constexpr (GeometryMap::dim() == 1)

      switch (geometryMap_collPts_) {

      case collPts::greville:
        return {G_.greville(/* interior */ false),
                G_.boundary().greville()};

      case collPts::greville_interior:
        return {G_.greville(/* interior */ true),
                G_.boundary().greville()};

      default:
        throw std::runtime_error("Invalid collocation point specifier");
      }
    
    else
      return geometryMap_collPts(std::make_index_sequence<GeometryMap::dim()>{});
  }

  /// @brief Returns the variable collocation points
  ///
  /// In the default implementation the collocation points are the Greville
  /// abscissae in the interior of the domain and on the boundary
  /// faces. This behavior can be changed by overriding this virtual
  /// function in a derived class.
  virtual variable_collPts_type variable_collPts(int64_t epoch) const {
    if constexpr (Variable::dim() == 1)

      switch (variable_collPts_) {

      case collPts::greville:
        return {f_.greville(/* interior */ false),
                f_.boundary().greville()};

      case collPts::greville_interior:
        return {f_.greville(/* interior */ true),
                f_.boundary().greville()};

      default:
        throw std::runtime_error("Invalid collocation point specifier");
      }
    
    else
      return variable_collPts(std::make_index_sequence<Variable::dim()>{});
  }

  /// @brief Returns the network inputs
  ///
  /// In the default implementation the inputs are the controll
  /// points of the geometry and the reference spline objects. This
  /// behavior can be changed by overriding this virtual function in
  /// a derived class.
  virtual torch::Tensor inputs(int64_t epoch) const {
    return torch::cat(
        {G_.as_tensor(/* no boundary */ false), f_.as_tensor()});
  }

  /// @brief Initializes epoch
  virtual enum status epoch(int64_t) = 0;

  /// @brief Computes the loss function
  virtual torch::Tensor loss(const torch::Tensor &,
                             const geometryMap_collPts_type &,
                             const variable_collPts_type &, int64_t,
                             enum status) = 0;

  /// @brief Trains the IgANet
  virtual void train() {
    torch::Tensor inputs, outputs, loss;
    geometryMap_collPts_type geometryMap_collPts;
    variable_collPts_type variable_collPts;
    status status;

    // Loop over epochs
    for (int64_t epoch = 0; epoch != options_.max_epoch(); ++epoch) {
      // Update from user-defined callback function
      status = this->epoch(epoch);

      if (status & status::inputs)
        inputs = this->inputs(epoch);

      if (status & status::geometryMap_collPts)
        geometryMap_collPts = this->geometryMap_collPts(epoch);

      if (status & status::variable_collPts)
        variable_collPts = this->variable_collPts(epoch);

      auto closure = [&]() {
        // Reset gradients
        net_->zero_grad();

        // Execute the model on the inputs
        outputs = net_->forward(inputs);

        // Compute the loss value
        loss = this->loss(outputs, geometryMap_collPts, variable_collPts, epoch,
                          status);

        // Compute gradients of the loss w.r.t. the model parameters
        loss.backward({}, true, false);

        std::clog << loss.template item<value_type>() << std::endl;
        return loss;
      };

      // Update the parameters based on the calculated gradients
      opt_.step(closure);

      if (loss.template item<value_type>() < options_.min_loss())
        break;
    }
  }

  /// @brief Returns the IgANet object as JSON object
  inline virtual nlohmann::json to_json() const override {
    return "Not implemented yet";
  }

  /// @brief Returns a string representation of the IgANet object
  inline virtual void
  pretty_print(std::ostream &os = std::cout) const noexcept override {
    os << name() << "(\n"
       << "net = " << net_ << "\n"
       << "geo = " << G_ << "\n"
       << "ref = " << f_ << "\n"
       << "out = " << u_ << "\n)";
  }

  /// @brief Saves the IgANet to file
  inline void save(const std::string &filename,
                   const std::string &key = "iganet") const {
    torch::serialize::OutputArchive archive;
    write(archive, key).save_to(filename);
  }

  /// @brief Loads the IgANet from file
  inline void load(const std::string &filename,
                   const std::string &key = "iganet") {
    torch::serialize::InputArchive archive;
    archive.load_from(filename);
    read(archive, key);
  }

  /// @brief Writes the IgANet into a torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "iganet") const {
    G_.write(archive, key + ".geo");
    f_.write(archive, key + ".ref");
    u_.write(archive, key + ".out");

    net_->write(archive, key + ".net");
    torch::serialize::OutputArchive archive_net;
    net_->save(archive_net);
    archive.write(key + ".net.data", archive_net);

    torch::serialize::OutputArchive archive_opt;
    opt_.save(archive_opt);
    archive.write(key + ".opt", archive_opt);

    return archive;
  }

  /// @brief Loads the IgANet from a torch::serialize::InputArchive object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "iganet") {
    G_.read(archive, key + ".geo");
    f_.read(archive, key + ".ref");
    u_.read(archive, key + ".out");

    net_->read(archive, key + ".net");
    torch::serialize::InputArchive archive_net;
    archive.read(key + ".net.data", archive_net);
    net_->load(archive_net);

    opt_.add_parameters(net_->parameters());
    torch::serialize::InputArchive archive_opt;
    archive.read(key + ".opt", archive_opt);
    opt_.load(archive_opt);

    return archive;
  }

  /// @brief Returns true if both IgANet objects are the same
  bool operator==(const IgANet &other) const {
    bool result(true);

    result *= (G_ == other.G());
    result *= (f_ == other.f());
    result *= (u_ == other.u());

    return result;
  }

  /// @brief Returns true if both IgANet objects are different
  bool operator!=(const IgANet &other) const { return *this != other; }
};

/// @brief Print (as string) a IgANet object
template <typename Optimizer, typename GeometryMap, typename Variable>
inline std::ostream &
operator<<(std::ostream &os, const IgANet<Optimizer, GeometryMap, Variable> &obj) {
  obj.pretty_print(os);
  return os;
}

/// @brief IgANetCustomizable
///
/// This class implements a customizable variant of IgANets that
/// provides types and attributes for precomputing indices and basis
/// functions
template <typename Optimizer, typename GeometryMap, typename Variable>
class IgANetCustomizable {
public:
  /// @brief Type of the knot indices of the geometry map in the interior
  using geometryMap_interior_knot_indices_type =
      decltype(std::declval<GeometryMap>()
                   .template find_knot_indices<functionspace::interior>(
                       std::declval<typename GeometryMap::eval_type>()));

  /// @brief Type of the knot indices of the geometry map at the boundary
  using geometryMap_boundary_knot_indices_type =
      decltype(std::declval<GeometryMap>()
                   .template find_knot_indices<functionspace::boundary>(
                       std::declval<typename GeometryMap::boundary_eval_type>()));

  /// @brief Type of the knot indices of the variables in the interior
  using variable_interior_knot_indices_type =
      decltype(std::declval<Variable>()
                   .template find_knot_indices<functionspace::interior>(
                       std::declval<typename Variable::eval_type>()));

  /// @brief Type of the knot indices of boundary_eval_type type at the boundary
  using variable_boundary_knot_indices_type =
      decltype(std::declval<Variable>()
                   .template find_knot_indices<functionspace::boundary>(
                       std::declval<typename Variable::boundary_eval_type>()));

protected:
  /// @brief Knot indices of the geometry map in the interior
  geometryMap_interior_knot_indices_type geometryMap_interior_knot_indices_;

  /// @brief Knot indices of the geometry map at the boundary
  geometryMap_boundary_knot_indices_type geometryMap_boundary_knot_indices_;

  /// @brief Knot indices of the variables in the interior
  variable_interior_knot_indices_type variable_interior_knot_indices_;

  /// @brief Knot indices of the variables at the boundary
  variable_boundary_knot_indices_type variable_boundary_knot_indices_;

public:
  /// @brief Type of the coefficient indices of geometry type in the interior
  using geometryMap_interior_coeff_indices_type =
      decltype(std::declval<GeometryMap>()
                   .template find_coeff_indices<functionspace::interior>(
                       std::declval<typename GeometryMap::eval_type>()));

  /// @brief Type of the coefficient indices of geometry type at the boundary
  using geometryMap_boundary_coeff_indices_type =
      decltype(std::declval<GeometryMap>()
                   .template find_coeff_indices<functionspace::boundary>(
                       std::declval<typename GeometryMap::boundary_eval_type>()));

  /// @brief Type of the coefficient indices of variable type in the interior
  using variable_interior_coeff_indices_type =
      decltype(std::declval<Variable>()
                   .template find_coeff_indices<functionspace::interior>(
                       std::declval<typename Variable::eval_type>()));

  /// @brief Type of the coefficient indices of variable type at the boundary
  using variable_boundary_coeff_indices_type =
      decltype(std::declval<Variable>()
                   .template find_coeff_indices<functionspace::boundary>(
                       std::declval<typename Variable::boundary_eval_type>()));

protected:
  /// @brief Coefficient indices of geometry type in the interior
  geometryMap_interior_coeff_indices_type geometryMap_interior_coeff_indices_;

  /// @brief Coefficient indices of geometry type at the vboundary
  geometryMap_boundary_coeff_indices_type geometryMap_boundary_coeff_indices_;

  /// @brief Coefficient indices of variable type in the interior
  variable_interior_coeff_indices_type variable_interior_coeff_indices_;

  /// @brief Coefficient indices of variable type at the boundary
  variable_boundary_coeff_indices_type variable_boundary_coeff_indices_;
};

} // namespace iganet
