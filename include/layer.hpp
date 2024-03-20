/**
   @file include/layer.hpp

   @brief Network layer

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <core.hpp>
#include <utils/fqn.hpp>

namespace iganet {

/// @brief Enumerator for nonlinear activation functions
enum class activation : short_t {
  none = 0,
  batch_norm = 1,
  celu = 2,
  elu = 3,
  gelu = 4,
  glu = 5,
  group_norm = 6,
  gumbel_softmax = 7,
  hardshrink = 9,
  hardsigmoid = 8,
  hardswish = 10,
  hardtanh = 11,
  instance_norm = 12,
  layer_norm = 13,
  leaky_relu = 14,
  local_response_norm = 15,
  logsigmoid = 16,
  logsoftmax = 17,
  mish = 18,
  normalize = 19,
  prelu = 20,
  relu = 21,
  relu6 = 22,
  rrelu = 23,
  selu = 24,
  sigmoid = 25,
  silu = 26,
  softmax = 27,
  softmin = 28,
  softplus = 29,
  softshrink = 30,
  softsign = 31,
  tanh = 32,
  tanhshrink = 33,
  threshold = 34
};

/// @brief Abstract activation function structure
class ActivationFunction : protected utils::FullQualifiedName {
public:
  virtual ~ActivationFunction() = default;

  /// @brief Applies the activation function to the given input
  virtual torch::Tensor apply(const torch::Tensor &) const = 0;

  /// @brief Returns a string representation of the activation function
  virtual void pretty_print(std::ostream &os) const noexcept = 0;

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  virtual torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key) const = 0;

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  virtual torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive, const std::string &key) = 0;
};

/// @brief Print (as string) an ActivationFunction object
inline std::ostream &operator<<(std::ostream &os,
                                const ActivationFunction &obj) {
  obj.pretty_print(os);
  return os;
}

/// @brief No-op activation function
class None : public ActivationFunction {
public:
  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return input;
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void pretty_print(std::ostream &os) const noexcept override {
    os << utils::FullQualifiedName::name();
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "none") const override {
    archive.write(key + ".type", torch::full({1}, (int64_t)activation::none));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "none") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::none)
      throw std::runtime_error("activation mismatch");

    return archive;
  }
};

/// @brief Batch Normalization as described in the paper
///
/// Batch Normalization: Accelerating Deep Network Training by
/// Reducing Internal Covariate Shift,
/// https://arxiv.org/abs/1502.03167
class BatchNorm : public ActivationFunction {
public:
  explicit BatchNorm(const torch::Tensor &running_mean,
                     const torch::Tensor &running_var,
                     torch::nn::functional::BatchNormFuncOptions options = {})
      : running_mean_(running_mean), running_var_(running_var),
        options_(std::move(options)) {}

  explicit BatchNorm(const torch::Tensor &running_mean,
                     const torch::Tensor &running_var,
                     const torch::Tensor &weight, const torch::Tensor &bias,
                     double eps, double momentum, bool training = false)
      : running_mean_(running_mean), running_var_(running_var),
        options_(torch::nn::functional::BatchNormFuncOptions()
                     .weight(weight)
                     .bias(bias)
                     .eps(eps)
                     .momentum(momentum)
                     .training(training)) {}

  ~BatchNorm() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::batch_norm(input, running_mean_, running_var_,
                                             options_);
  }

  /// @brief Returns constant reference to running mean
  inline const torch::Tensor &running_mean() const { return running_mean_; }

  /// @brief Returns non-constant reference to running mean
  inline torch::Tensor &running_mean() { return running_mean_; }

  /// @brief Returns constant reference to running variance
  inline const torch::Tensor &running_var() const { return running_var_; }

  /// @brief Returns non-constant reference to running var
  inline torch::Tensor &running_var() { return running_var_; }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::BatchNormFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::BatchNormFuncOptions &options() {
    return options_;
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name() << "(\n  eps=" << options_.eps()
       << ", momentum=" << options_.momentum().value()
       << ", training=" << options_.training();

    if (is_verbose(os)) {
      os << "\n  running_mean = " << running_mean()
         << "\n  running_var = " << running_var()
         << "\n  weight = " << options_.weight()
         << "\n  bias = " << options_.bias();
    }

    os << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "batch_norm") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::batch_norm));
    archive.write(key + ".running_mean", this->running_mean());
    archive.write(key + ".running_var", this->running_var());
    archive.write(key + ".weight", this->options_.weight());
    archive.write(key + ".bias", this->options_.bias());
    archive.write(key + ".eps", torch::full({1}, (double)this->options_.eps()));
    archive.write(key + ".momentum",
                  torch::full({1}, (double)this->options_.momentum().value()));
    archive.write(key + ".training",
                  torch::full({1}, (bool)this->options_.training()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "batch_norm") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::batch_norm)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".running_mean", this->running_mean());
    archive.read(key + ".running_var", this->running_var());
    archive.read(key + ".weight", this->options_.weight());
    archive.read(key + ".bias", this->options_.bias());
    archive.read(key + ".eps", tensor);
    this->options_.eps(tensor.item<double>());
    archive.read(key + ".momentum", tensor);
    this->options_.momentum(tensor.item<double>());
    archive.read(key + ".training", tensor);
    this->options_.training(tensor.item<bool>());

    return archive;
  }

private:
  torch::nn::functional::BatchNormFuncOptions options_;
  torch::Tensor running_mean_, running_var_;
};

/// @brief Continuously Differentiable Exponential Linear Units activation
/// function
///
/// \f[
///     \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha)-1))
/// \f]
class CELU : public ActivationFunction {
public:
  explicit CELU(torch::nn::functional::CELUFuncOptions options = {})
      : options_(options) {}

  explicit CELU(double alpha, bool inplace = false)
      : options_(torch::nn::functional::CELUFuncOptions().alpha(alpha).inplace(
            inplace)) {}

  ~CELU() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::celu(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::CELUFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::CELUFuncOptions &options() { return options_; }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name() << "(\n  alpha=" << options_.alpha()
       << ", inplace=" << options_.inplace() << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "celu") const override {
    archive.write(key + ".type", torch::full({1}, (int64_t)activation::celu));
    archive.write(key + ".alpha",
                  torch::full({1}, (double)this->options_.alpha()));
    archive.write(key + ".inplace",
                  torch::full({1}, (bool)this->options_.inplace()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "celu") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::celu)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".alpha", tensor);
    this->options_.alpha(tensor.item<double>());
    archive.read(key + ".inplace", tensor);
    this->options_.inplace(tensor.item<bool>());

    return archive;
  }

private:
  torch::nn::functional::CELUFuncOptions options_;
};

/// @brief Exponential Linear Units activation function
///
/// \f[
///     \text{ELU}(x) =
///       \begin{cases}
///         x                    & \text{ if } x > 0
///         \alpha * (\exp(x)-1) & \text{ if } x \le 0
///       \end{cases}
/// \f]
class ELU : public ActivationFunction {
public:
  explicit ELU(torch::nn::functional::ELUFuncOptions options = {})
      : options_(options) {}

  explicit ELU(double alpha, bool inplace = false)
      : options_(torch::nn::functional::ELUFuncOptions().alpha(alpha).inplace(
            inplace)) {}

  ~ELU() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::elu(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::ELUFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::ELUFuncOptions &options() { return options_; }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name() << "(\n  alpha=" << options_.alpha()
       << ", inplace=" << options_.inplace() << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "elu") const override {
    archive.write(key + ".type", torch::full({1}, (int64_t)activation::elu));
    archive.write(key + ".alpha",
                  torch::full({1}, (double)this->options_.alpha()));
    archive.write(key + ".inplace",
                  torch::full({1}, (bool)this->options_.inplace()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "elu") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::elu)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".alpha", tensor);
    this->options_.alpha(tensor.item<double>());
    archive.read(key + ".inplace", tensor);
    this->options_.inplace(tensor.item<bool>());

    return archive;
  }

private:
  torch::nn::functional::ELUFuncOptions options_;
};

/// @brief Gaussian Error Linear Units activation function
///
/// \f[
///     \text{GELU}(x) = x * \Psi(x),
/// \f]
///
/// where \f$\Psi(x)\f$ is the Cumulative Distribution Function for
/// Gaussian Distribution
class GELU : public ActivationFunction {
public:
  explicit GELU() = default;

  ~GELU() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::gelu(input);
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name();
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "gelu") const override {
    archive.write(key + ".type", torch::full({1}, (int64_t)activation::gelu));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "gelu") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::gelu)
      throw std::runtime_error("activation mismatch");

    return archive;
  }
};

/// @brief Grated Linear Units activation function
///
/// \f[
///     \text{GLU}(a,b) = a \otimes \sigma(b),
/// \f]
///
/// where input is split in half along dim to form \f$ a \f$ and \f$ b \f$,
/// \f$ \sigma \f$ is the sigmoid function and \f$ \otimes \f$
/// is the element-wise product between matrices.
class GLU : public ActivationFunction {
public:
  explicit GLU(torch::nn::functional::GLUFuncOptions options = {})
      : options_(options) {}

  explicit GLU(int64_t dim)
      : options_(torch::nn::functional::GLUFuncOptions().dim(dim)) {}

  ~GLU() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::glu(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::GLUFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::GLUFuncOptions &options() { return options_; }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name() << "(\n  dim=" << options_.dim()
       << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "glu") const override {
    archive.write(key + ".type", torch::full({1}, (int64_t)activation::glu));
    archive.write(key + ".dim", torch::full({1}, (int)this->options_.dim()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "glu") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::glu)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".dim", tensor);
    this->options_.dim(tensor.item<int>());

    return archive;
  }

private:
  torch::nn::functional::GLUFuncOptions options_;
};

/// @brief Group Normalization over a mini-batch of inputs as described in
/// the paper Group Normalization, https://arxiv.org/abs/1803.08494
class GroupNorm : public ActivationFunction {
public:
  explicit GroupNorm(int64_t num_groups)
      : options_(torch::nn::functional::GroupNormFuncOptions(num_groups)) {}

  explicit GroupNorm(torch::nn::functional::GroupNormFuncOptions options)
      : options_(std::move(options)) {}

  explicit GroupNorm(int64_t num_groups, const torch::Tensor &weight,
                     const torch::Tensor &bias, double eps)
      : options_(torch::nn::functional::GroupNormFuncOptions(num_groups)
                     .weight(weight)
                     .bias(bias)
                     .eps(eps)) {}

  ~GroupNorm() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::group_norm(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::GroupNormFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::GroupNormFuncOptions &options() {
    return options_;
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name() << "(\n  eps=" << options_.eps();

    if (is_verbose(os)) {
      os << "\n  weight = " << options_.weight()
         << "\n  bias = " << options_.bias();
    }

    os << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "group_norm") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::group_norm));
    archive.write(key + ".weight", this->options_.weight());
    archive.write(key + ".bias", this->options_.bias());
    archive.write(key + ".eps", torch::full({1}, (double)this->options_.eps()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "group_norm") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::group_norm)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".weight", this->options_.weight());
    archive.read(key + ".bias", this->options_.bias());
    archive.read(key + ".eps", tensor);
    this->options_.eps(tensor.item<double>());

    return archive;
  }

private:
  torch::nn::functional::GroupNormFuncOptions options_;
};

/// @brief Gumbel-Softmax distribution activation function
class GumbelSoftmax : public ActivationFunction {
public:
  explicit GumbelSoftmax(
      torch::nn::functional::GumbelSoftmaxFuncOptions options = {})
      : options_(options) {}

  explicit GumbelSoftmax(double tau, int dim, bool hard)
      : options_(torch::nn::functional::GumbelSoftmaxFuncOptions()
                     .tau(tau)
                     .dim(dim)
                     .hard(hard)) {}

  ~GumbelSoftmax() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::gumbel_softmax(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::GumbelSoftmaxFuncOptions &
  options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::GumbelSoftmaxFuncOptions &options() {
    return options_;
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name() << "(\n  tau=" << options_.tau()
       << ", dim=" << options_.dim() << ", hard=" << options_.hard() << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "gumbel_softmax") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::gumbel_softmax));
    archive.write(key + ".tau", torch::full({1}, (double)this->options_.tau()));
    archive.write(key + ".dim", torch::full({1}, (int)this->options_.dim()));
    archive.write(key + ".hard", torch::full({1}, (bool)this->options_.hard()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "gumbel_softmax") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::gumbel_softmax)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".tau", tensor);
    this->options_.tau(tensor.item<double>());
    archive.read(key + ".dim", tensor);
    this->options_.dim(tensor.item<int>());
    archive.read(key + ".hard", tensor);
    this->options_.hard(tensor.item<bool>());

    return archive;
  }

private:
  torch::nn::functional::GumbelSoftmaxFuncOptions options_;
};

/// @brief Hard shrinkish activation function
class Hardshrink : public ActivationFunction {
public:
  explicit Hardshrink(torch::nn::functional::HardshrinkFuncOptions options = {})
      : options_(options) {}

  explicit Hardshrink(double lambda)
      : options_(
            torch::nn::functional::HardshrinkFuncOptions().lambda(lambda)) {}

  ~Hardshrink() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::hardshrink(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::HardshrinkFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::HardshrinkFuncOptions &options() {
    return options_;
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name()
       << "(\n  lambda=" << options_.lambda() << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "hardshrink") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::hardshrink));
    archive.write(key + ".lambda",
                  torch::full({1}, (double)this->options_.lambda()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "hardshrink") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::hardshrink)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".lambda", tensor);
    this->options_.lambda(tensor.item<double>());

    return archive;
  }

private:
  torch::nn::functional::HardshrinkFuncOptions options_;
};

/// @brief Hardsigmoid activation function
///
/// \f[
///     \text{Hardsigmoid}(x) =
///       \begin{cases}
///         0         & \text{ if } x \le -3
///         1         & \text{ if } x \ge +3
///         x/6 + 1/2 & \text{ otherwise }
///       \end{cases}
/// \f]
class Hardsigmoid : public ActivationFunction {
public:
  explicit Hardsigmoid() = default;

  ~Hardsigmoid() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::hardsigmoid(input);
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name();
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "hardsigmoid") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::hardsigmoid));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "hardsigmoid") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::hardsigmoid)
      throw std::runtime_error("activation mismatch");

    return archive;
  }
};

/// @brief Hardswish activation function
///
/// \f[
///     \text{Hardswish}(x) =
///       \begin{cases}
///         0         & \text{ if } x \le -3
///         x         & \text{ if } x \ge +3
///         x*(x+3)/6 & \text{ otherwise }
///       \end{cases}
/// \f]
class Hardswish : public ActivationFunction {
public:
  explicit Hardswish() = default;

  ~Hardswish() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::hardswish(input);
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name();
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "hardswish") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::hardswish));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "hardswish") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::hardswish)
      throw std::runtime_error("activation mismatch");

    return archive;
  }
};

/// @brief Hardtanh activation function
///
/// \f[
///     \text{Hardtanh}(x) =
///       \begin{cases}
///         +1 & \text{ if } x > +1
///         -1 & \text{ if } x < -1
///          x & \text{ otherwise }
///       \end{cases}
/// \f]
class Hardtanh : public ActivationFunction {
public:
  explicit Hardtanh(torch::nn::functional::HardtanhFuncOptions options = {})
      : options_(options) {}

  explicit Hardtanh(double min_val, double max_val, bool inplace = false)
      : options_(torch::nn::functional::HardtanhFuncOptions()
                     .min_val(min_val)
                     .max_val(max_val)
                     .inplace(inplace)) {}

  ~Hardtanh() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::hardtanh(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::HardtanhFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::HardtanhFuncOptions &options() {
    return options_;
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name()
       << "(\n  min_val=" << options_.min_val()
       << ", max_val=" << options_.max_val()
       << ", inplace=" << options_.inplace() << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "hardtanh") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::hardtanh));
    archive.write(key + ".min_val",
                  torch::full({1}, (double)this->options_.min_val()));
    archive.write(key + ".max_val",
                  torch::full({1}, (double)this->options_.max_val()));
    archive.write(key + ".inplace",
                  torch::full({1}, (bool)this->options_.inplace()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "hardtanh") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::hardtanh)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".min_val", tensor);
    this->options_.min_val(tensor.item<double>());
    archive.read(key + ".max_val", tensor);
    this->options_.max_val(tensor.item<double>());
    archive.read(key + ".inplace", tensor);
    this->options_.inplace(tensor.item<bool>());

    return archive;
  }

private:
  torch::nn::functional::HardtanhFuncOptions options_;
};

/// @brief Instance Normalization as described in the paper
///
/// Instance Normalization: The Missing Ingredient for Fast
/// Stylization, https://arxiv.org/abs/1607.08022
class InstanceNorm : public ActivationFunction {
public:
  explicit InstanceNorm(
      torch::nn::functional::InstanceNormFuncOptions options = {})
      : options_(std::move(options)) {}

  explicit InstanceNorm(const torch::Tensor &running_mean,
                        const torch::Tensor &running_var,
                        const torch::Tensor &weight, const torch::Tensor &bias,
                        double eps, double momentum,
                        bool use_input_stats = true)
      : options_(torch::nn::functional::InstanceNormFuncOptions()
                     .running_mean(running_mean)
                     .running_var(running_var)
                     .weight(weight)
                     .bias(bias)
                     .eps(eps)
                     .momentum(momentum)
                     .use_input_stats(use_input_stats)) {}

  ~InstanceNorm() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::instance_norm(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::InstanceNormFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::InstanceNormFuncOptions &options() {
    return options_;
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name() << "(\n  eps=" << options_.eps()
       << ", momentum=" << options_.momentum()
       << ", use_input_stats=" << options_.use_input_stats();

    if (is_verbose(os)) {
      os << "\n  running_mean = " << options_.running_mean()
         << "\n  running_var = " << options_.running_var()
         << "\n  weight = " << options_.weight()
         << "\n  bias = " << options_.bias();
    }

    os << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "instance_norm") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::instance_norm));
    archive.write(key + ".running_mean", this->options_.running_mean());
    archive.write(key + ".var", this->options_.running_var());
    archive.write(key + ".weight", this->options_.weight());
    archive.write(key + ".bias", this->options_.bias());
    archive.write(key + ".eps", torch::full({1}, (double)this->options_.eps()));
    archive.write(key + ".momentum",
                  torch::full({1}, (double)this->options_.momentum()));
    archive.write(key + ".use_input_stats",
                  torch::full({1}, (bool)this->options_.use_input_stats()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "instance_norm") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::instance_norm)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".running_mean", this->options_.running_mean());
    archive.read(key + ".running_var", this->options_.running_var());
    archive.read(key + ".weight", this->options_.weight());
    archive.read(key + ".bias", this->options_.bias());
    archive.read(key + ".eps", tensor);
    this->options_.eps(tensor.item<double>());
    archive.read(key + ".momentum", tensor);
    this->options_.momentum(tensor.item<double>());
    archive.read(key + ".use_input_stats", tensor);
    this->options_.use_input_stats(tensor.item<bool>());

    return archive;
  }

private:
  torch::nn::functional::InstanceNormFuncOptions options_;
};

/// @brief Layer Normalization as described in the paper
///
/// Layer Normalization, https://arxiv.org/abs/1607.06450
class LayerNorm : public ActivationFunction {
public:
  explicit LayerNorm(std::vector<int64_t> normalized_shape)
      : options_(torch::nn::functional::LayerNormFuncOptions(
            std::move(normalized_shape))) {}

  explicit LayerNorm(torch::nn::functional::LayerNormFuncOptions options)
      : options_(std::move(options)) {}

  explicit LayerNorm(std::vector<int64_t> normalized_shape,
                     const torch::Tensor &weight, const torch::Tensor &bias,
                     double eps)
      : options_(torch::nn::functional::LayerNormFuncOptions(
                     std::move(normalized_shape))
                     .weight(weight)
                     .bias(bias)
                     .eps(eps)) {}

  ~LayerNorm() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::layer_norm(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::LayerNormFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::LayerNormFuncOptions &options() {
    return options_;
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name() << "(\n  eps=" << options_.eps();

    if (is_verbose(os)) {
      os << "\n  normalized_shape = " << options_.normalized_shape()
         << "\n  weight = " << options_.weight()
         << "\n  bias = " << options_.bias();
    }

    os << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "layer_norm") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::layer_norm));
    archive.write(key + ".weight", this->options_.weight());
    archive.write(key + ".bias", this->options_.bias());
    archive.write(key + ".eps", torch::full({1}, (double)this->options_.eps()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "layer_norm") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::layer_norm)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".weight", this->options_.weight());
    archive.read(key + ".bias", this->options_.bias());
    archive.read(key + ".eps", tensor);
    this->options_.eps(tensor.item<double>());

    return archive;
  }

private:
  torch::nn::functional::LayerNormFuncOptions options_;
};

/// @brief Leaky ReLU activation function
///
/// \f[
///     \text{LeakyReLU}(x) =
///       \begin{cases}
///         x                         & \text{ if } x \ge 0
///         \text{negative_slope} * x & \text{ otherwise }
///       \end{cases}
/// \f]
class LeakyReLU : public ActivationFunction {
public:
  explicit LeakyReLU(torch::nn::functional::LeakyReLUFuncOptions options = {})
      : options_(options) {}

  explicit LeakyReLU(double negative_slope, bool inplace = false)
      : options_(torch::nn::functional::LeakyReLUFuncOptions()
                     .negative_slope(negative_slope)
                     .inplace(inplace)) {}

  ~LeakyReLU() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::leaky_relu(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::LeakyReLUFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::LeakyReLUFuncOptions &options() {
    return options_;
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name()
       << "(\n  negative_slope=" << options_.negative_slope()
       << ", inplace=" << options_.inplace() << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "leaky_relu") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::leaky_relu));

    archive.write(key + ".negative_slope",
                  torch::full({1}, (double)this->options_.negative_slope()));
    archive.write(key + ".inplace",
                  torch::full({1}, (bool)this->options_.inplace()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "leaky_relu") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::leaky_relu)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".negative_slope", tensor);
    this->options_.negative_slope(tensor.item<double>());
    archive.read(key + ".inplace", tensor);
    this->options_.inplace(tensor.item<bool>());

    return archive;
  }

private:
  torch::nn::functional::LeakyReLUFuncOptions options_;
};

/// @brief Local response Normalization
class LocalResponseNorm : public ActivationFunction {
public:
  explicit LocalResponseNorm(int64_t size)
      : options_(torch::nn::functional::LocalResponseNormFuncOptions(size)) {}

  explicit LocalResponseNorm(
      const torch::nn::functional::LocalResponseNormFuncOptions &options)
      : options_(options) {}

  explicit LocalResponseNorm(int64_t size, double alpha, double beta, double k)
      : options_(torch::nn::functional::LocalResponseNormFuncOptions(size)
                     .alpha(alpha)
                     .beta(beta)
                     .k(k)) {}

  ~LocalResponseNorm() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::local_response_norm(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::LocalResponseNormFuncOptions &
  options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::LocalResponseNormFuncOptions &options() {
    return options_;
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name() << "(\n  size=" << options_.size()
       << ", alpha=" << options_.alpha() << ", beta=" << options_.beta()
       << ", k=" << options_.k() << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "local_response_norm") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::local_response_norm));

    archive.write(key + ".size",
                  torch::full({1}, (int64_t)this->options_.size()));
    archive.write(key + ".alpha",
                  torch::full({1}, (double)this->options_.alpha()));
    archive.write(key + ".beta",
                  torch::full({1}, (double)this->options_.beta()));
    archive.write(key + ".k", torch::full({1}, (double)this->options_.k()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "local_response_norm") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::local_response_norm)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".size", tensor);
    this->options_.size(tensor.item<int64_t>());
    archive.read(key + ".alpha", tensor);
    this->options_.alpha(tensor.item<double>());
    archive.read(key + ".beta", tensor);
    this->options_.beta(tensor.item<double>());
    archive.read(key + ".k", tensor);
    this->options_.k(tensor.item<double>());

    return archive;
  }

private:
  torch::nn::functional::LocalResponseNormFuncOptions options_;
};

/// @brief LogSigmoid activation function
///
/// \f[
///     \text{LogSigmoid}(x) = \log \left( \frac{1}{1+\exp(-x)} \right)
/// \f]
class LogSigmoid : public ActivationFunction {
public:
  explicit LogSigmoid() = default;

  ~LogSigmoid() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::log_sigmoid(input);
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name();
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "logsigmoid") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::logsigmoid));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "logsigmoid") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::logsigmoid)
      throw std::runtime_error("activation mismatch");

    return archive;
  }
};

/// @brief LogSoftmax activation function
///
/// \f[
///     \text{LogSigmoid}(x_i) = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)}
///     \right)
/// \f]
///
/// where \f$ x \f$ is an \f$n\f$-dimensional input tensor
class LogSoftmax : public ActivationFunction {
public:
  explicit LogSoftmax(int64_t dim)
      : options_(torch::nn::functional::LogSoftmaxFuncOptions(dim)) {}

  explicit LogSoftmax(
      const torch::nn::functional::LogSoftmaxFuncOptions &options)
      : options_(options) {}

  ~LogSoftmax() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::log_softmax(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::LogSoftmaxFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::LogSoftmaxFuncOptions &options() {
    return options_;
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name() << "(\n  dim=" << options_.dim()
       << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "logsoftmax") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::logsoftmax));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "logsoftmax") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::logsoftmax)
      throw std::runtime_error("activation mismatch");

    return archive;
  }

private:
  torch::nn::functional::LogSoftmaxFuncOptions options_;
};

/// @brief Mish activation function
///
/// \f[
///     \text{Mish}(x) = x * \tanh(\text{Softplus}(x))
/// \f]
class Mish : public ActivationFunction {
public:
  explicit Mish() = default;

  ~Mish() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::mish(input);
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name();
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "mish") const override {
    archive.write(key + ".type", torch::full({1}, (int64_t)activation::mish));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "mish") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::mish)
      throw std::runtime_error("activation mismatch");

    return archive;
  }
};

/// @brief Lp Normalization
class Normalize : public ActivationFunction {
public:
  explicit Normalize(torch::nn::functional::NormalizeFuncOptions options = {})
      : options_(std::move(options)) {}

  explicit Normalize(double p, double eps, int64_t dim)
      : options_(
            torch::nn::functional::NormalizeFuncOptions().p(p).eps(eps).dim(
                dim)) {}

  ~Normalize() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::normalize(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::NormalizeFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::NormalizeFuncOptions &options() {
    return options_;
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name() << "(\n  eps=" << options_.eps()
       << "(\n  p=" << options_.p() << "(\n  dim=" << options_.dim() << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "normalize") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::normalize));
    archive.write(key + ".p", torch::full({1}, (double)this->options_.p()));
    archive.write(key + ".eps", torch::full({1}, (double)this->options_.eps()));
    archive.write(key + ".dim",
                  torch::full({1}, (int64_t)this->options_.dim()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "normalize") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::normalize)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".p", tensor);
    this->options_.p(tensor.item<double>());
    archive.read(key + ".eps", tensor);
    this->options_.eps(tensor.item<double>());
    archive.read(key + ".dim", tensor);
    this->options_.dim(tensor.item<int64_t>());

    return archive;
  }

private:
  torch::nn::functional::NormalizeFuncOptions options_;
};

/// @brief PReLU activation function
class PReLU : public ActivationFunction {
public:
  explicit PReLU(const torch::Tensor &weight) : weight_(weight) {}

  ~PReLU() override = default;

  /// @brief Returns constant reference to weights
  const torch::Tensor &weight() const { return weight_; }

  /// @brief Returns non-constant reference to weights
  torch::Tensor &weight() { return weight_; }

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::prelu(input, weight());
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name();

    if (is_verbose(os))
      os << "(\n  weight = " << weight() << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "prelu") const override {
    archive.write(key + ".type", torch::full({1}, (int64_t)activation::prelu));
    archive.write(key + ".weight", this->weight());

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "prelu") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::prelu)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".weight", this->weight());

    return archive;
  }

private:
  torch::Tensor weight_;
};

/// @brief ReLU activation function
///
/// \f[
///     \text{ReLU}(x) = \max(0,x)
/// \f]
class ReLU : public ActivationFunction {
public:
  explicit ReLU(torch::nn::functional::ReLUFuncOptions options = {})
      : options_(options) {}

  explicit ReLU(bool inplace)
      : options_(torch::nn::functional::ReLUFuncOptions().inplace(inplace)) {}

  ~ReLU() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::relu(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::ReLUFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::ReLUFuncOptions &options() { return options_; }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name()
       << "(\n  inplace=" << options_.inplace() << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "relu") const override {
    archive.write(key + ".type", torch::full({1}, (int64_t)activation::relu));
    archive.write(key + ".inplace",
                  torch::full({1}, (bool)this->options_.inplace()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "relu") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::relu)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".inplace", tensor);
    this->options_.inplace(tensor.item<bool>());

    return archive;
  }

private:
  torch::nn::functional::ReLUFuncOptions options_;
};

/// @brief ReLU6 activation function
///
/// \f[
///     \text{ReLU6}(x) = \min(\max(0,x),6)
/// \f]
class ReLU6 : public ActivationFunction {
public:
  explicit ReLU6(torch::nn::functional::ReLU6FuncOptions options = {})
      : options_(options) {}

  explicit ReLU6(bool inplace)
      : options_(torch::nn::functional::ReLU6FuncOptions().inplace(inplace)) {}

  ~ReLU6() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::relu6(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::ReLU6FuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::ReLU6FuncOptions &options() { return options_; }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name()
       << "(\n  inplace=" << options_.inplace() << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "relu6") const override {
    archive.write(key + ".type", torch::full({1}, (int64_t)activation::relu6));
    archive.write(key + ".inplace",
                  torch::full({1}, (bool)this->options_.inplace()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "relu6") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::relu6)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".inplace", tensor);
    this->options_.inplace(tensor.item<bool>());

    return archive;
  }

private:
  torch::nn::functional::ReLU6FuncOptions options_;
};

/// @brief Randomized ReLU activation function
///
/// \f[
///     \text{RReLU}(x) =
///     \begin{cases}
///           x & \text{ if } x \ge 0
///       a * x & \text{ otherwise }
///     \end{cases}
/// \f]
class RReLU : public ActivationFunction {
public:
  explicit RReLU(torch::nn::functional::RReLUFuncOptions options = {})
      : options_(options) {}

  explicit RReLU(double lower, double upper, bool inplace = false)
      : options_(torch::nn::functional::RReLUFuncOptions()
                     .lower(lower)
                     .upper(upper)
                     .inplace(inplace)) {}

  ~RReLU() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::rrelu(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::RReLUFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::RReLUFuncOptions &options() { return options_; }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name() << "(\n  lower=" << options_.lower()
       << ",  upper=" << options_.upper() << ",  inplace=" << options_.inplace()
       << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "rrelu") const override {
    archive.write(key + ".type", torch::full({1}, (int64_t)activation::rrelu));
    archive.write(key + ".lower",
                  torch::full({1}, (double)this->options_.lower()));
    archive.write(key + ".upper",
                  torch::full({1}, (double)this->options_.upper()));
    archive.write(key + ".inplace",
                  torch::full({1}, (bool)this->options_.inplace()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "rrelu") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::rrelu)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".lower", tensor);
    this->options_.lower(tensor.item<double>());
    archive.read(key + ".upper", tensor);
    this->options_.upper(tensor.item<double>());
    archive.read(key + ".inplace", tensor);
    this->options_.inplace(tensor.item<bool>());

    return archive;
  }

private:
  torch::nn::functional::RReLUFuncOptions options_;
};

/// @brief SELU activation function
///
/// \f[
///     \text{SELU}(x) = s * ( \max(0,x) + \min(0, \alpha*(\exp(x)-1 ) ) )
/// \f]
///
/// with \f$ s = 1.0507009873554804934193349852946 \f$ and
/// \f$ \alpha = 1.6732632423543772848170429916717 \f$.
class SELU : public ActivationFunction {
public:
  explicit SELU(torch::nn::functional::SELUFuncOptions options = {})
      : options_(options) {}

  explicit SELU(bool inplace)
      : options_(torch::nn::functional::SELUFuncOptions().inplace(inplace)) {}

  ~SELU() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::selu(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::SELUFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::SELUFuncOptions &options() { return options_; }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name()
       << "(\n  inplace=" << options_.inplace() << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "selu") const override {
    archive.write(key + ".type", torch::full({1}, (int64_t)activation::selu));
    archive.write(key + ".inplace",
                  torch::full({1}, (bool)this->options_.inplace()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "selu") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::selu)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".inplace", tensor);
    this->options_.inplace(tensor.item<bool>());

    return archive;
  }

private:
  torch::nn::functional::SELUFuncOptions options_;
};

/// @brief Sigmoid activation function
///
/// \f[
///    \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1+\exp(-x)}
/// \f]
class Sigmoid : public ActivationFunction {
public:
  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::sigmoid(input);
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name();
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "sigmoid") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::sigmoid));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "sigmoid") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::sigmoid)
      throw std::runtime_error("activation mismatch");

    return archive;
  }
};

/// @brief Sigmoid Linear Unit activation function
///
/// \f[
///    \text{SiLU}(x) = x * \sigma(x)
/// \f]
class SiLU : public ActivationFunction {
public:
  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::silu(input);
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name();
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "silu") const override {
    archive.write(key + ".type", torch::full({1}, (int64_t)activation::silu));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "silu") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::silu)
      throw std::runtime_error("activation mismatch");

    return archive;
  }
};

/// @brief Softmax activation function
///
/// \f[
///     \text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
/// \f]
///
/// where \f$ x \f$ is an \f$n\f$-dimensional input tensor
class Softmax : public ActivationFunction {
public:
  explicit Softmax(int64_t dim)
      : options_(torch::nn::functional::SoftmaxFuncOptions(dim)) {}

  explicit Softmax(const torch::nn::functional::SoftmaxFuncOptions &options)
      : options_(options) {}

  ~Softmax() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::softmax(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::SoftmaxFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::SoftmaxFuncOptions &options() {
    return options_;
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name() << "(\n  dim=" << options_.dim()
       << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "softmax") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::softmax));
    archive.write(key + ".dim",
                  torch::full({1}, (int64_t)this->options_.dim()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "softmax") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::softmax)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".dim", tensor);
    this->options_.dim(tensor.item<int64_t>());

    return archive;
  }

private:
  torch::nn::functional::SoftmaxFuncOptions options_;
};

/// @brief Softmin activation function
///
/// \f[
///     \text{Softmin}(x) = \text{Softmax}(-x)
/// \f]
class Softmin : public ActivationFunction {
public:
  explicit Softmin(int64_t dim)
      : options_(torch::nn::functional::SoftminFuncOptions(dim)) {}

  explicit Softmin(const torch::nn::functional::SoftminFuncOptions &options)
      : options_(options) {}

  ~Softmin() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::softmin(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::SoftminFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::SoftminFuncOptions &options() {
    return options_;
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name() << "(\n  dim=" << options_.dim()
       << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "softmin") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::softmin));
    archive.write(key + ".dim",
                  torch::full({1}, (int64_t)this->options_.dim()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "softmin") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::softmin)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".dim", tensor);
    this->options_.dim(tensor.item<int64_t>());

    return archive;
  }

private:
  torch::nn::functional::SoftminFuncOptions options_;
};

/// @brief Softplus activation function
///
/// \f[
///     \text{Softplus}(x) = \frac{1}{\beta} * \log( 1+\exp(\beta * x) )
/// \f]
class Softplus : public ActivationFunction {
public:
  explicit Softplus(torch::nn::functional::SoftplusFuncOptions options = {})
      : options_(options) {}

  explicit Softplus(double beta, double threshold)
      : options_(
            torch::nn::functional::SoftplusFuncOptions().beta(beta).threshold(
                threshold)) {}

  ~Softplus() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::softplus(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::SoftplusFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::SoftplusFuncOptions &options() {
    return options_;
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name() << "(\n  beta=" << options_.beta()
       << ",  theshold=" << options_.threshold() << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "softplus") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::softplus));
    archive.write(key + ".beta",
                  torch::full({1}, (double)this->options_.beta()));
    archive.write(key + ".threshold",
                  torch::full({1}, (double)this->options_.threshold()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "softplus") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::softplus)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".beta", tensor);
    this->options_.beta(tensor.item<double>());
    archive.read(key + ".threshold", tensor);
    this->options_.threshold(tensor.item<double>());

    return archive;
  }

private:
  torch::nn::functional::SoftplusFuncOptions options_;
};

/// @brief Softshrink activation function
///
/// \f[
///     \text{Softshrink}(x) =
///     \begin{cases}
///       x - \lambda & \text{ if } x > \lambda
///       x + \lambda & \text{ if } x < \lambda
///       0           & \text{ otherwise }
///     \end{cases}
/// \f]
class Softshrink : public ActivationFunction {
public:
  explicit Softshrink(torch::nn::functional::SoftshrinkFuncOptions options = {})
      : options_(options) {}

  explicit Softshrink(double lambda)
      : options_(
            torch::nn::functional::SoftshrinkFuncOptions().lambda(lambda)) {}

  ~Softshrink() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::softshrink(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::SoftshrinkFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::SoftshrinkFuncOptions &options() {
    return options_;
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name()
       << "(\n  lambda=" << options_.lambda() << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "softshrink") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::softshrink));
    archive.write(key + ".lambda",
                  torch::full({1}, (double)this->options_.lambda()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "softshrink") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::softshrink)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".lambda", tensor);
    this->options_.lambda(tensor.item<double>());

    return archive;
  }

private:
  torch::nn::functional::SoftshrinkFuncOptions options_;
};

/// @brief Softsign activation function
///
/// \f[
///    \text{Softsign}(x) = \frac{x}{1+\abs{x}}
/// \f]
class Softsign : public ActivationFunction {
public:
  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::softsign(input);
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name();
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "softsign") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::softsign));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "softsign") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::softsign)
      throw std::runtime_error("activation mismatch");

    return archive;
  }
};

/// @brief Tanh activation function
///
/// \f[
///    \text{Tanh}(x) = \frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}
/// \f]
class Tanh : public ActivationFunction {
public:
  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::tanh(input);
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name();
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "tanh") const override {
    archive.write(key + ".type", torch::full({1}, (int64_t)activation::tanh));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "tanh") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::tanh)
      throw std::runtime_error("activation mismatch");

    return archive;
  }
};

/// @brief Tanhshrink activation function
///
/// \f[
///    \text{Tanhshrink}(x) = x - \text{Tanh}(x)
/// \f]
class Tanhshrink : public ActivationFunction {
public:
  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::tanhshrink(input);
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name();
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "tanhshrink") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::tanhshrink));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "tanhshrink") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::tanhshrink)
      throw std::runtime_error("activation mismatch");

    return archive;
  }
};

/// @brief Threshold activation function
///
/// \f[
///     \text{Threshold}(x) =
///     \begin{cases}
///       x     & \text{ if } x > threshold
///       value & \text{ otherwise }
///     \end{cases}
/// \f]
class Threshold : public ActivationFunction {
public:
  explicit Threshold(torch::nn::functional::ThresholdFuncOptions options)
      : options_(options) {}

  explicit Threshold(double threshold, double value, bool inplace = false)
      : options_(torch::nn::functional::ThresholdFuncOptions(threshold, value)
                     .inplace(inplace)) {}

  ~Threshold() override = default;

  /// @brief Applies the activation function to the given input
  inline torch::Tensor apply(const torch::Tensor &input) const override {
    return torch::nn::functional::threshold(input, options_);
  }

  /// @brief Returns constant reference to options
  inline const torch::nn::functional::ThresholdFuncOptions &options() const {
    return options_;
  }

  /// @brief Returns non-constant reference to options
  inline torch::nn::functional::ThresholdFuncOptions &options() {
    return options_;
  }

  /// @brief Returns a string representation of the activation function
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << utils::FullQualifiedName::name()
       << "(\n  threshold=" << options_.threshold()
       << ",  value=" << options_.value() << ",  inplace=" << options_.inplace()
       << "\n)";
  }

  /// @brief Writes the activation function into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "threshold") const override {
    archive.write(key + ".type",
                  torch::full({1}, (int64_t)activation::threshold));
    archive.write(key + ".threshold",
                  torch::full({1}, (double)this->options_.threshold()));
    archive.write(key + ".value",
                  torch::full({1}, (double)this->options_.value()));
    archive.write(key + ".inplace",
                  torch::full({1}, (bool)this->options_.inplace()));

    return archive;
  }

  /// @brief Reads the activation function from a torch::serialize::InputArchive
  /// object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "threshold") override {
    torch::Tensor tensor;

    archive.read(key + ".type", tensor);
    if (tensor.item<int64_t>() != (int64_t)activation::threshold)
      throw std::runtime_error("activation mismatch");

    archive.read(key + ".threshold", tensor);
    this->options_.threshold(tensor.item<double>());
    archive.read(key + ".value", tensor);
    this->options_.value(tensor.item<double>());
    archive.read(key + ".inplace", tensor);
    this->options_.inplace(tensor.item<bool>());

    return archive;
  }

private:
  torch::nn::functional::ThresholdFuncOptions options_;
};

} // namespace iganet
