/**
   @file include/utils/blocktensor.hpp

   @brief Compile-time block tensor

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <array>
#include <exception>
#include <initializer_list>
#include <memory>
#include <type_traits>

#include <core.hpp>
#include <utils/fqn.hpp>

namespace iganet {
namespace utils {

/// @brief Type trait checks if template argument is of type std::shared_ptr<T>
/// @{
template <typename T> struct is_shared_ptr : std::false_type {};

template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};
/// @}

/// @brief Returns an std::shared_ptr<T> object from arg
template <typename T> inline auto make_shared(T &&arg) {
  if constexpr (is_shared_ptr<typename std::decay<T>::type>::value)
    return std::forward<typename std::decay<T>::type>(arg);
  else
    return std::make_shared<typename std::decay<T>::type>(std::forward<T>(arg));
}

/// @brief Forward declaration of BlockTensor
template <typename T, std::size_t... Dims> class BlockTensor;

/// @brief Compile-time block tensor core
template <typename T, std::size_t... Dims>
class BlockTensorCore : protected iganet::utils::FullQualifiedName {

protected:
  /// @brief Array storing the data
  std::array<std::shared_ptr<T>, (Dims * ...)> data_;

public:
  /// @brief Default constructor
  BlockTensorCore() = default;

  /// @brief Constructur from BlockTensorCore objects
  template <typename... Ts, std::size_t... dims>
  BlockTensorCore(BlockTensorCore<Ts, dims...> &&...other) {
    auto it = data_.begin();
    (std::transform(other.data().begin(), other.data().end(), it,
                    [&it](auto &&d) {
                      it++;
                      return std::move(d);
                    }),
     ...);
  }

  /// @brief Constructur from BlockTensor objects
  template <typename... Ts, std::size_t... dims>
  BlockTensorCore(BlockTensor<Ts, dims...> &&...other) {
    auto it = data_.begin();
    (std::transform(other.data().begin(), other.data().end(), it,
                    [&it](auto &&d) {
                      it++;
                      return std::move(d);
                    }),
     ...);
  }

  /// @brief Constructor from variadic templates
  template <typename... Ts>
  explicit BlockTensorCore(Ts &&...data)
      : data_({make_shared<Ts>(std::forward<Ts>(data))...}) {}

  /// @brief Returns all dimensions as array
  inline static constexpr auto dims() {
    return std::array<std::size_t, sizeof...(Dims)>({Dims...});
  }

  /// @brief Returns the i-th dimension
  template <std::size_t i> inline static constexpr std::size_t dim() {
    if constexpr (i < sizeof...(Dims))
      return std::get<i>(std::forward_as_tuple(Dims...));
    else
      return 0;
  }

  /// @brief Returns the number of dimensions
  inline static constexpr std::size_t size() { return sizeof...(Dims); }

  /// @brief Returns the total number of entries
  inline static constexpr std::size_t entries() { return (Dims * ...); }

  /// @brief Returns a constant reference to the data array
  inline const std::array<std::shared_ptr<T>, (Dims * ...)> &data() const {
    return data_;
  }

  /// @brief Returns a non-constant reference to the data array
  inline std::array<std::shared_ptr<T>, (Dims * ...)> &data() { return data_; }

  /// @brief Returns a constant shared pointer to entry (idx)
  inline const std::shared_ptr<T> &operator[](std::size_t idx) const {
    assert(0 <= idx && idx < (Dims * ...));
    return data_[idx];
  }

  /// @brief Returns a non-constant shared pointer to entry (idx)
  inline std::shared_ptr<T> &operator[](std::size_t idx) {
    assert(0 <= idx && idx < (Dims * ...));
    return data_[idx];
  }

  /// @brief Returns a constant reference to entry (idx)
  inline const T &operator()(std::size_t idx) const {
    assert(0 <= idx && idx < (Dims * ...));
    return *data_[idx];
  }

  /// @brief Returns a non-constant reference to entry (idx)
  inline T &operator()(std::size_t idx) {
    assert(0 <= idx && idx < (Dims * ...));
    return *data_[idx];
  }

  /// @brief Stores the given data object at the given index
  template <typename Data> inline T &set(std::size_t idx, Data &&data) {
    assert(0 <= idx && idx < (Dims * ...));
    data_[idx] = make_shared<Data>(std::forward<Data>(data));
    return *data_[idx];
  }

  /// Returns a string representation of the BlockTensorCore object
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept = 0;
};

/// Prints (as string) a compile-time block tensor object
template <typename T, std::size_t... Dims>
inline std::ostream &operator<<(std::ostream &os,
                                const BlockTensorCore<T, Dims...> &obj) {
  obj.pretty_print(os);
  return os;
}

/// @brief Compile-time rank-1 block tensor (row vector)
template <typename T, std::size_t Rows>
class BlockTensor<T, Rows> : public BlockTensorCore<T, Rows> {
private:
  using Base = BlockTensorCore<T, Rows>;

public:
  using BlockTensorCore<T, Rows>::BlockTensorCore;

  /// @brief Returns the number of rows
  inline static constexpr std::size_t rows() { return Rows; }

  /// Returns a string representation of the BlockTensor object
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << Base::name() << "\n";
    for (std::size_t row = 0; row < Rows; ++row)
      os << "[" << row << "] = \n" << *Base::data_[row] << "\n";
  }
};

/// @brief Compile-time rank-2 block tensor (matrix)
///
/// Data is store in row-major order, i.e. all entries of a row are
/// stored contiguously in memory and the entries of the next row
/// are stored with an offset of Cols
template <typename T, std::size_t Rows, std::size_t Cols>
class BlockTensor<T, Rows, Cols> : public BlockTensorCore<T, Rows, Cols> {
private:
  using Base = BlockTensorCore<T, Rows, Cols>;

public:
  using BlockTensorCore<T, Rows, Cols>::BlockTensorCore;

  /// @brief Returns the number of rows
  inline static constexpr std::size_t rows() { return Rows; }

  /// @brief Returns the number of columns
  inline static constexpr std::size_t cols() { return Cols; }

  using Base::operator();

  /// @brief Returns a constant reference to entry (row, col)
  inline const T &operator()(std::size_t row, std::size_t col) const {
    assert(0 <= row && row < Rows && 0 <= col && col < Cols);
    return *Base::data_[Cols * row + col];
  }

  /// @brief Returns a non-constant reference to entry (row, col)
  inline T &operator()(std::size_t row, std::size_t col) {
    assert(0 <= row && row < Rows && 0 <= col && col < Cols);
    return *Base::data_[Cols * row + col];
  }

  using Base::set;

  /// @brief Stores the given data object at the given position
  template <typename D>
  inline T &set(std::size_t row, std::size_t col, D &&data) {
    assert(0 <= row && row < Rows && 0 <= col && col < Cols);
    Base::data_[Cols * row + col] = make_shared<D>(std::forward<D>(data));
    return *Base::data_[Cols * row + col];
  }

  /// @brief Returns the transpose of the block tensor
  inline auto tr() const {
    BlockTensor<T, Cols, Rows> result;
    for (std::size_t row = 0; row < Rows; ++row)
      for (std::size_t col = 0; col < Cols; ++col)
        result[Rows * col + row] = Base::data_[Cols * row + col];
    return result;
  }

  /// @brief Returns the determinant of a square block tensor
  ///
  /// This function computes the determinant of a square block tensor

  inline auto det() const {
    if constexpr (Rows == 1 && Cols == 1) {
      auto result = *Base::data_[0];
      return result;
    } else if constexpr (Rows == 2 && Cols == 2) {
      auto result = torch::mul(*Base::data_[0], *Base::data_[3]) -
                    torch::mul(*Base::data_[1], *Base::data_[2]);
      return result;
    } else if constexpr (Rows == 3 && Cols == 3) {
      auto result =
          torch::mul(*Base::data_[0],
                     torch::mul(*Base::data_[4], *Base::data_[8]) -
                         torch::mul(*Base::data_[5], *Base::data_[7])) -
          torch::mul(*Base::data_[1],
                     torch::mul(*Base::data_[3], *Base::data_[8]) -
                         torch::mul(*Base::data_[5], *Base::data_[6])) +
          torch::mul(*Base::data_[2],
                     torch::mul(*Base::data_[3], *Base::data_[7]) -
                         torch::mul(*Base::data_[4], *Base::data_[6]));
      return result;
    } else if constexpr (Rows == 4 && Cols == 4) {
      auto a11 = torch::mul(*Base::data_[5],
                            (torch::mul(*Base::data_[10], *Base::data_[15]) -
                             torch::mul(*Base::data_[11], *Base::data_[14]))) -
                 torch::mul(*Base::data_[9],
                            (torch::mul(*Base::data_[6], *Base::data_[15]) -
                             torch::mul(*Base::data_[7], *Base::data_[14]))) -
                 torch::mul(*Base::data_[13],
                            (torch::mul(*Base::data_[7], *Base::data_[10]) -
                             torch::mul(*Base::data_[6], *Base::data_[11])));

      auto a21 = torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[11], *Base::data_[14]) -
                             torch::mul(*Base::data_[10], *Base::data_[15]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[7], *Base::data_[14]) -
                             torch::mul(*Base::data_[6], *Base::data_[15]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[6], *Base::data_[11]) -
                             torch::mul(*Base::data_[7], *Base::data_[10])));

      auto a31 = torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[9], *Base::data_[15]) -
                             torch::mul(*Base::data_[11], *Base::data_[13]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[5], *Base::data_[15]) -
                             torch::mul(*Base::data_[7], *Base::data_[13]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[7], *Base::data_[9]) -
                             torch::mul(*Base::data_[5], *Base::data_[11])));

      auto a41 = torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[10], *Base::data_[13]) -
                             torch::mul(*Base::data_[9], *Base::data_[14]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[6], *Base::data_[13]) -
                             torch::mul(*Base::data_[5], *Base::data_[14]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[5], *Base::data_[10]) -
                             torch::mul(*Base::data_[6], *Base::data_[9])));

      auto result =
          torch::mul(*Base::data_[0], a11) + torch::mul(*Base::data_[1], a21) +
          torch::mul(*Base::data_[2], a31) + torch::mul(*Base::data_[3], a41);

      return result;
    } else {
      throw std::runtime_error("Unsupported block tensor dimension");
      return *this;
    }
  }

  /// @brief Returns the inverse of the block tensor
  ///
  /// This function computes the inverse of the block tensor.
  inline auto inv() const {

    auto det_ = this->det();

    if constexpr (Rows == 1 && Cols == 1) {
      BlockTensor<T, Rows, Cols> result;
      result[0] = std::make_shared<T>(torch::reciprocal(*Base::data_[0]));
      return result;
    } else if constexpr (Rows == 2 && Cols == 2) {

      BlockTensor<T, Rows, Cols> result;
      result[0] = std::make_shared<T>(torch::div(*Base::data_[3], det_));
      result[1] = std::make_shared<T>(torch::div(*Base::data_[2], -det_));
      result[2] = std::make_shared<T>(torch::div(*Base::data_[1], -det_));
      result[3] = std::make_shared<T>(torch::div(*Base::data_[0], det_));
      return result;
    } else if constexpr (Rows == 3 && Cols == 3) {

      auto a11 = torch::mul(*Base::data_[4], *Base::data_[8]) -
                 torch::mul(*Base::data_[5], *Base::data_[7]);
      auto a12 = torch::mul(*Base::data_[2], *Base::data_[7]) -
                 torch::mul(*Base::data_[1], *Base::data_[8]);
      auto a13 = torch::mul(*Base::data_[1], *Base::data_[5]) -
                 torch::mul(*Base::data_[2], *Base::data_[4]);
      auto a21 = torch::mul(*Base::data_[5], *Base::data_[6]) -
                 torch::mul(*Base::data_[3], *Base::data_[8]);
      auto a22 = torch::mul(*Base::data_[0], *Base::data_[8]) -
                 torch::mul(*Base::data_[2], *Base::data_[6]);
      auto a23 = torch::mul(*Base::data_[2], *Base::data_[3]) -
                 torch::mul(*Base::data_[0], *Base::data_[5]);
      auto a31 = torch::mul(*Base::data_[3], *Base::data_[7]) -
                 torch::mul(*Base::data_[4], *Base::data_[6]);
      auto a32 = torch::mul(*Base::data_[1], *Base::data_[6]) -
                 torch::mul(*Base::data_[0], *Base::data_[7]);
      auto a33 = torch::mul(*Base::data_[0], *Base::data_[4]) -
                 torch::mul(*Base::data_[1], *Base::data_[3]);

      BlockTensor<T, Rows, Cols> result;
      result[0] = std::make_shared<T>(torch::div(a11, det_));
      result[1] = std::make_shared<T>(torch::div(a12, det_));
      result[2] = std::make_shared<T>(torch::div(a13, det_));
      result[3] = std::make_shared<T>(torch::div(a21, det_));
      result[4] = std::make_shared<T>(torch::div(a22, det_));
      result[5] = std::make_shared<T>(torch::div(a23, det_));
      result[6] = std::make_shared<T>(torch::div(a31, det_));
      result[7] = std::make_shared<T>(torch::div(a32, det_));
      result[8] = std::make_shared<T>(torch::div(a33, det_));
      return result;
    } else if constexpr (Rows == 4 && Cols == 4) {
      auto a11 = torch::mul(*Base::data_[5],
                            (torch::mul(*Base::data_[10], *Base::data_[15]) -
                             torch::mul(*Base::data_[11], *Base::data_[14]))) -
                 torch::mul(*Base::data_[9],
                            (torch::mul(*Base::data_[6], *Base::data_[15]) -
                             torch::mul(*Base::data_[7], *Base::data_[14]))) -
                 torch::mul(*Base::data_[13],
                            (torch::mul(*Base::data_[7], *Base::data_[10]) -
                             torch::mul(*Base::data_[6], *Base::data_[11])));

      auto a12 = torch::mul(*Base::data_[1],
                            (torch::mul(*Base::data_[11], *Base::data_[14]) -
                             torch::mul(*Base::data_[10], *Base::data_[15]))) -
                 torch::mul(*Base::data_[9],
                            (torch::mul(*Base::data_[3], *Base::data_[14]) -
                             torch::mul(*Base::data_[2], *Base::data_[15]))) -
                 torch::mul(*Base::data_[13],
                            (torch::mul(*Base::data_[2], *Base::data_[11]) -
                             torch::mul(*Base::data_[3], *Base::data_[10])));

      auto a13 = torch::mul(*Base::data_[1],
                            (torch::mul(*Base::data_[6], *Base::data_[15]) -
                             torch::mul(*Base::data_[7], *Base::data_[14]))) -
                 torch::mul(*Base::data_[5],
                            (torch::mul(*Base::data_[2], *Base::data_[15]) -
                             torch::mul(*Base::data_[3], *Base::data_[14]))) -
                 torch::mul(*Base::data_[13],
                            (torch::mul(*Base::data_[3], *Base::data_[6]) -
                             torch::mul(*Base::data_[2], *Base::data_[7])));

      auto a14 = torch::mul(*Base::data_[1],
                            (torch::mul(*Base::data_[7], *Base::data_[10]) -
                             torch::mul(*Base::data_[6], *Base::data_[11]))) -
                 torch::mul(*Base::data_[5],
                            (torch::mul(*Base::data_[3], *Base::data_[10]) -
                             torch::mul(*Base::data_[2], *Base::data_[11]))) -
                 torch::mul(*Base::data_[9],
                            (torch::mul(*Base::data_[2], *Base::data_[7]) -
                             torch::mul(*Base::data_[3], *Base::data_[6])));

      auto a21 = torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[11], *Base::data_[14]) -
                             torch::mul(*Base::data_[10], *Base::data_[15]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[7], *Base::data_[14]) -
                             torch::mul(*Base::data_[6], *Base::data_[15]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[6], *Base::data_[11]) -
                             torch::mul(*Base::data_[7], *Base::data_[10])));

      auto a22 = torch::mul(*Base::data_[0],
                            (torch::mul(*Base::data_[10], *Base::data_[15]) -
                             torch::mul(*Base::data_[11], *Base::data_[14]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[2], *Base::data_[15]) -
                             torch::mul(*Base::data_[3], *Base::data_[14]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[3], *Base::data_[10]) -
                             torch::mul(*Base::data_[2], *Base::data_[11])));

      auto a23 = torch::mul(*Base::data_[0],
                            (torch::mul(*Base::data_[7], *Base::data_[14]) -
                             torch::mul(*Base::data_[6], *Base::data_[15]))) -
                 torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[3], *Base::data_[14]) -
                             torch::mul(*Base::data_[2], *Base::data_[15]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[2], *Base::data_[7]) -
                             torch::mul(*Base::data_[3], *Base::data_[6])));

      auto a24 = torch::mul(*Base::data_[0],
                            (torch::mul(*Base::data_[6], *Base::data_[11]) -
                             torch::mul(*Base::data_[7], *Base::data_[10]))) -
                 torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[2], *Base::data_[11]) -
                             torch::mul(*Base::data_[3], *Base::data_[10]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[3], *Base::data_[6]) -
                             torch::mul(*Base::data_[2], *Base::data_[7])));

      auto a31 = torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[9], *Base::data_[15]) -
                             torch::mul(*Base::data_[11], *Base::data_[13]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[5], *Base::data_[15]) -
                             torch::mul(*Base::data_[7], *Base::data_[13]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[7], *Base::data_[9]) -
                             torch::mul(*Base::data_[5], *Base::data_[11])));

      auto a32 = torch::mul(*Base::data_[0],
                            (torch::mul(*Base::data_[11], *Base::data_[13]) -
                             torch::mul(*Base::data_[9], *Base::data_[15]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[3], *Base::data_[13]) -
                             torch::mul(*Base::data_[1], *Base::data_[15]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[1], *Base::data_[11]) -
                             torch::mul(*Base::data_[3], *Base::data_[9])));

      auto a33 = torch::mul(*Base::data_[0],
                            (torch::mul(*Base::data_[5], *Base::data_[15]) -
                             torch::mul(*Base::data_[7], *Base::data_[13]))) -
                 torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[1], *Base::data_[15]) -
                             torch::mul(*Base::data_[3], *Base::data_[13]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[3], *Base::data_[5]) -
                             torch::mul(*Base::data_[1], *Base::data_[7])));

      auto a34 = torch::mul(*Base::data_[0],
                            (torch::mul(*Base::data_[7], *Base::data_[9]) -
                             torch::mul(*Base::data_[5], *Base::data_[11]))) -
                 torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[3], *Base::data_[9]) -
                             torch::mul(*Base::data_[1], *Base::data_[11]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[1], *Base::data_[7]) -
                             torch::mul(*Base::data_[3], *Base::data_[5])));

      auto a41 = torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[10], *Base::data_[13]) -
                             torch::mul(*Base::data_[9], *Base::data_[14]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[6], *Base::data_[13]) -
                             torch::mul(*Base::data_[5], *Base::data_[14]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[5], *Base::data_[10]) -
                             torch::mul(*Base::data_[6], *Base::data_[9])));

      auto a42 = torch::mul(*Base::data_[0],
                            (torch::mul(*Base::data_[9], *Base::data_[14]) -
                             torch::mul(*Base::data_[10], *Base::data_[13]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[1], *Base::data_[14]) -
                             torch::mul(*Base::data_[2], *Base::data_[13]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[2], *Base::data_[9]) -
                             torch::mul(*Base::data_[1], *Base::data_[10])));

      auto a43 = torch::mul(*Base::data_[0],
                            (torch::mul(*Base::data_[6], *Base::data_[13]) -
                             torch::mul(*Base::data_[5], *Base::data_[14]))) -
                 torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[2], *Base::data_[13]) -
                             torch::mul(*Base::data_[1], *Base::data_[14]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[1], *Base::data_[6]) -
                             torch::mul(*Base::data_[2], *Base::data_[5])));

      auto a44 = torch::mul(*Base::data_[0],
                            (torch::mul(*Base::data_[5], *Base::data_[10]) -
                             torch::mul(*Base::data_[6], *Base::data_[9]))) -
                 torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[1], *Base::data_[10]) -
                             torch::mul(*Base::data_[2], *Base::data_[9]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[2], *Base::data_[5]) -
                             torch::mul(*Base::data_[1], *Base::data_[6])));
      BlockTensor<T, Rows, Cols> result;
      result[0] = std::make_shared<T>(torch::div(a11, det_));
      result[1] = std::make_shared<T>(torch::div(a12, det_));
      result[2] = std::make_shared<T>(torch::div(a13, det_));
      result[3] = std::make_shared<T>(torch::div(a14, det_));
      result[4] = std::make_shared<T>(torch::div(a21, det_));
      result[5] = std::make_shared<T>(torch::div(a22, det_));
      result[6] = std::make_shared<T>(torch::div(a23, det_));
      result[7] = std::make_shared<T>(torch::div(a24, det_));
      result[8] = std::make_shared<T>(torch::div(a31, det_));
      result[9] = std::make_shared<T>(torch::div(a32, det_));
      result[10] = std::make_shared<T>(torch::div(a33, det_));
      result[11] = std::make_shared<T>(torch::div(a34, det_));
      result[12] = std::make_shared<T>(torch::div(a41, det_));
      result[13] = std::make_shared<T>(torch::div(a42, det_));
      result[14] = std::make_shared<T>(torch::div(a43, det_));
      result[15] = std::make_shared<T>(torch::div(a44, det_));
      return result;
    } else {
      throw std::runtime_error("Unsupported block tensor dimension");
      return *this;
    }
  }

  /// @brief Returns the (generalized) inverse of the block tensor
  ///
  /// This function computes the (generalized) inverse of the
  /// block tensor. For square matrices it computes the regular
  /// inverse matrix based on explicit iversion formulas assuming
  /// that the matrix is invertible. For rectangular matrices it
  /// computes the generalized inverse i.e. \f$(A^T A)^{-1} A^T\f$.
  inline auto ginv() const {
    if constexpr (Rows == Cols)
      return this->inv();
    else
      // Compute the generalized inverse, i.e. (A^T A)^{-1} A^T
      return (this->tr() * (*this)).inv() * this->tr();
  }

  /// @brief Returns the transpose of the inverse of the block
  /// tensor
  ///
  /// This function computes the transpose of the (generalized)
  /// inverse of the block tensor.
  inline auto invtr() const {

    auto det_ = this->det();

    if constexpr (Rows == 1 && Cols == 1) {
      BlockTensor<T, Cols, Rows> result;
      result[0] = std::make_shared<T>(torch::reciprocal(*Base::data_[0]));
      return result;
    } else if constexpr (Rows == 2 && Cols == 2) {

      BlockTensor<T, Cols, Rows> result;
      result[0] = std::make_shared<T>(torch::div(*Base::data_[3], det_));
      result[1] = std::make_shared<T>(torch::div(*Base::data_[1], -det_));
      result[2] = std::make_shared<T>(torch::div(*Base::data_[2], -det_));
      result[3] = std::make_shared<T>(torch::div(*Base::data_[0], det_));
      return result;
    } else if constexpr (Rows == 3 && Cols == 3) {

      auto a11 = torch::mul(*Base::data_[4], *Base::data_[8]) -
                 torch::mul(*Base::data_[5], *Base::data_[7]);
      auto a12 = torch::mul(*Base::data_[2], *Base::data_[7]) -
                 torch::mul(*Base::data_[1], *Base::data_[8]);
      auto a13 = torch::mul(*Base::data_[1], *Base::data_[5]) -
                 torch::mul(*Base::data_[2], *Base::data_[4]);
      auto a21 = torch::mul(*Base::data_[5], *Base::data_[6]) -
                 torch::mul(*Base::data_[3], *Base::data_[8]);
      auto a22 = torch::mul(*Base::data_[0], *Base::data_[8]) -
                 torch::mul(*Base::data_[2], *Base::data_[6]);
      auto a23 = torch::mul(*Base::data_[2], *Base::data_[3]) -
                 torch::mul(*Base::data_[0], *Base::data_[5]);
      auto a31 = torch::mul(*Base::data_[3], *Base::data_[7]) -
                 torch::mul(*Base::data_[4], *Base::data_[6]);
      auto a32 = torch::mul(*Base::data_[1], *Base::data_[6]) -
                 torch::mul(*Base::data_[0], *Base::data_[7]);
      auto a33 = torch::mul(*Base::data_[0], *Base::data_[4]) -
                 torch::mul(*Base::data_[1], *Base::data_[3]);

      BlockTensor<T, Cols, Rows> result;
      result[0] = std::make_shared<T>(torch::div(a11, det_));
      result[1] = std::make_shared<T>(torch::div(a21, det_));
      result[2] = std::make_shared<T>(torch::div(a31, det_));
      result[3] = std::make_shared<T>(torch::div(a12, det_));
      result[4] = std::make_shared<T>(torch::div(a22, det_));
      result[5] = std::make_shared<T>(torch::div(a32, det_));
      result[6] = std::make_shared<T>(torch::div(a13, det_));
      result[7] = std::make_shared<T>(torch::div(a23, det_));
      result[8] = std::make_shared<T>(torch::div(a33, det_));
      return result;
    } else if constexpr (Rows == 4 && Cols == 4) {

      auto a11 = torch::mul(*Base::data_[5],
                            (torch::mul(*Base::data_[10], *Base::data_[15]) -
                             torch::mul(*Base::data_[11], *Base::data_[14]))) -
                 torch::mul(*Base::data_[9],
                            (torch::mul(*Base::data_[6], *Base::data_[15]) -
                             torch::mul(*Base::data_[7], *Base::data_[14]))) -
                 torch::mul(*Base::data_[13],
                            (torch::mul(*Base::data_[7], *Base::data_[10]) -
                             torch::mul(*Base::data_[6], *Base::data_[11])));

      auto a12 = torch::mul(*Base::data_[1],
                            (torch::mul(*Base::data_[11], *Base::data_[14]) -
                             torch::mul(*Base::data_[10], *Base::data_[15]))) -
                 torch::mul(*Base::data_[9],
                            (torch::mul(*Base::data_[3], *Base::data_[14]) -
                             torch::mul(*Base::data_[2], *Base::data_[15]))) -
                 torch::mul(*Base::data_[13],
                            (torch::mul(*Base::data_[2], *Base::data_[11]) -
                             torch::mul(*Base::data_[3], *Base::data_[10])));

      auto a13 = torch::mul(*Base::data_[1],
                            (torch::mul(*Base::data_[6], *Base::data_[15]) -
                             torch::mul(*Base::data_[7], *Base::data_[14]))) -
                 torch::mul(*Base::data_[5],
                            (torch::mul(*Base::data_[2], *Base::data_[15]) -
                             torch::mul(*Base::data_[3], *Base::data_[14]))) -
                 torch::mul(*Base::data_[13],
                            (torch::mul(*Base::data_[3], *Base::data_[6]) -
                             torch::mul(*Base::data_[2], *Base::data_[7])));

      auto a14 = torch::mul(*Base::data_[1],
                            (torch::mul(*Base::data_[7], *Base::data_[10]) -
                             torch::mul(*Base::data_[6], *Base::data_[11]))) -
                 torch::mul(*Base::data_[5],
                            (torch::mul(*Base::data_[3], *Base::data_[10]) -
                             torch::mul(*Base::data_[2], *Base::data_[11]))) -
                 torch::mul(*Base::data_[9],
                            (torch::mul(*Base::data_[2], *Base::data_[7]) -
                             torch::mul(*Base::data_[3], *Base::data_[6])));

      auto a21 = torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[11], *Base::data_[14]) -
                             torch::mul(*Base::data_[10], *Base::data_[15]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[7], *Base::data_[14]) -
                             torch::mul(*Base::data_[6], *Base::data_[15]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[6], *Base::data_[11]) -
                             torch::mul(*Base::data_[7], *Base::data_[10])));

      auto a22 = torch::mul(*Base::data_[0],
                            (torch::mul(*Base::data_[10], *Base::data_[15]) -
                             torch::mul(*Base::data_[11], *Base::data_[14]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[2], *Base::data_[15]) -
                             torch::mul(*Base::data_[3], *Base::data_[14]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[3], *Base::data_[10]) -
                             torch::mul(*Base::data_[2], *Base::data_[11])));

      auto a23 = torch::mul(*Base::data_[0],
                            (torch::mul(*Base::data_[7], *Base::data_[14]) -
                             torch::mul(*Base::data_[6], *Base::data_[15]))) -
                 torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[3], *Base::data_[14]) -
                             torch::mul(*Base::data_[2], *Base::data_[15]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[2], *Base::data_[7]) -
                             torch::mul(*Base::data_[3], *Base::data_[6])));

      auto a24 = torch::mul(*Base::data_[0],
                            (torch::mul(*Base::data_[6], *Base::data_[11]) -
                             torch::mul(*Base::data_[7], *Base::data_[10]))) -
                 torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[2], *Base::data_[11]) -
                             torch::mul(*Base::data_[3], *Base::data_[10]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[3], *Base::data_[6]) -
                             torch::mul(*Base::data_[2], *Base::data_[7])));

      auto a31 = torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[9], *Base::data_[15]) -
                             torch::mul(*Base::data_[11], *Base::data_[13]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[5], *Base::data_[15]) -
                             torch::mul(*Base::data_[7], *Base::data_[13]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[7], *Base::data_[9]) -
                             torch::mul(*Base::data_[5], *Base::data_[11])));

      auto a32 = torch::mul(*Base::data_[0],
                            (torch::mul(*Base::data_[11], *Base::data_[13]) -
                             torch::mul(*Base::data_[9], *Base::data_[15]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[3], *Base::data_[13]) -
                             torch::mul(*Base::data_[1], *Base::data_[15]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[1], *Base::data_[11]) -
                             torch::mul(*Base::data_[3], *Base::data_[9])));

      auto a33 = torch::mul(*Base::data_[0],
                            (torch::mul(*Base::data_[5], *Base::data_[15]) -
                             torch::mul(*Base::data_[7], *Base::data_[13]))) -
                 torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[1], *Base::data_[15]) -
                             torch::mul(*Base::data_[3], *Base::data_[13]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[3], *Base::data_[5]) -
                             torch::mul(*Base::data_[1], *Base::data_[7])));

      auto a34 = torch::mul(*Base::data_[0],
                            (torch::mul(*Base::data_[7], *Base::data_[9]) -
                             torch::mul(*Base::data_[5], *Base::data_[11]))) -
                 torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[3], *Base::data_[9]) -
                             torch::mul(*Base::data_[1], *Base::data_[11]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[1], *Base::data_[7]) -
                             torch::mul(*Base::data_[3], *Base::data_[5])));

      auto a41 = torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[10], *Base::data_[13]) -
                             torch::mul(*Base::data_[9], *Base::data_[14]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[6], *Base::data_[13]) -
                             torch::mul(*Base::data_[5], *Base::data_[14]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[5], *Base::data_[10]) -
                             torch::mul(*Base::data_[6], *Base::data_[9])));

      auto a42 = torch::mul(*Base::data_[0],
                            (torch::mul(*Base::data_[9], *Base::data_[14]) -
                             torch::mul(*Base::data_[10], *Base::data_[13]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[1], *Base::data_[14]) -
                             torch::mul(*Base::data_[2], *Base::data_[13]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[2], *Base::data_[9]) -
                             torch::mul(*Base::data_[1], *Base::data_[10])));

      auto a43 = torch::mul(*Base::data_[0],
                            (torch::mul(*Base::data_[6], *Base::data_[13]) -
                             torch::mul(*Base::data_[5], *Base::data_[14]))) -
                 torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[2], *Base::data_[13]) -
                             torch::mul(*Base::data_[1], *Base::data_[14]))) -
                 torch::mul(*Base::data_[12],
                            (torch::mul(*Base::data_[1], *Base::data_[6]) -
                             torch::mul(*Base::data_[2], *Base::data_[5])));

      auto a44 = torch::mul(*Base::data_[0],
                            (torch::mul(*Base::data_[5], *Base::data_[10]) -
                             torch::mul(*Base::data_[6], *Base::data_[9]))) -
                 torch::mul(*Base::data_[4],
                            (torch::mul(*Base::data_[1], *Base::data_[10]) -
                             torch::mul(*Base::data_[2], *Base::data_[9]))) -
                 torch::mul(*Base::data_[8],
                            (torch::mul(*Base::data_[2], *Base::data_[5]) -
                             torch::mul(*Base::data_[1], *Base::data_[6])));

      BlockTensor<T, Rows, Cols> result;
      result[0] = std::make_shared<T>(torch::div(a11, det_));
      result[1] = std::make_shared<T>(torch::div(a21, det_));
      result[2] = std::make_shared<T>(torch::div(a31, det_));
      result[3] = std::make_shared<T>(torch::div(a41, det_));
      result[4] = std::make_shared<T>(torch::div(a12, det_));
      result[5] = std::make_shared<T>(torch::div(a22, det_));
      result[6] = std::make_shared<T>(torch::div(a32, det_));
      result[7] = std::make_shared<T>(torch::div(a42, det_));
      result[8] = std::make_shared<T>(torch::div(a13, det_));
      result[9] = std::make_shared<T>(torch::div(a23, det_));
      result[10] = std::make_shared<T>(torch::div(a33, det_));
      result[11] = std::make_shared<T>(torch::div(a43, det_));
      result[12] = std::make_shared<T>(torch::div(a14, det_));
      result[13] = std::make_shared<T>(torch::div(a24, det_));
      result[14] = std::make_shared<T>(torch::div(a34, det_));
      result[15] = std::make_shared<T>(torch::div(a44, det_));
      return result;
    } else {
      throw std::runtime_error("Unsupported block tensor dimension");
      return *this;
    }
  }

  /// @brief Returns the transpose of the (generalized) inverse of
  /// the block tensor
  ///
  /// This function computes the transpose of the (generalized)
  /// inverse of the block tensor. For square matrices it computes
  /// the regular inverse matrix based on explicit iversion formulas
  /// assuming that the matrix is invertible and transposed it
  /// afterwards. For rectangular matrices it computes the
  /// generalized inverse i.e. \f$((A^T A)^{-1} A^T)^T = A (A^T A)^{-T}\f$.
  inline auto ginvtr() const {
    if constexpr (Rows == Cols)
      return this->invtr();
    else
      // Compute the transpose of the generalized inverse, i.e. A (A^T A)^{-T}
      return (*this) * (this->tr() * (*this)).invtr();
  }

  /// Returns the trace of the block tensor
  inline auto trace() const {
    static_assert(Rows == Cols, "trace(.) requires square block tensor");

    if constexpr (Rows == 1)
      return BlockTensor<T, 1, 1>(Base::data_[0]);

    else if constexpr (Rows == 2)
      return BlockTensor<T, 1, 1>(*Base::data_[0] + *Base::data_[3]);

    else if constexpr (Rows == 3)
      return BlockTensor<T, 1, 1>(*Base::data_[0] + *Base::data_[4] +
                                  *Base::data_[8]);

    else if constexpr (Rows == 4)
      return BlockTensor<T, 1, 1>(*Base::data_[0] + *Base::data_[5] +
                                  *Base::data_[10] + *Base::data_[15]);

    else
      throw std::runtime_error("Unsupported block tensor dimension");
  }

  /// Returns a string representation of the BSplineCommon object
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << Base::name() << "\n";
    for (std::size_t row = 0; row < Rows; ++row)
      for (std::size_t col = 0; col < Cols; ++col)
        os << "[" << row << "," << col << "] = \n"
           << *Base::data_[Cols * row + col] << "\n";
  }
};

/// @brief Multiplies one compile-time rank-2 block tensor with
/// another compile-time rank-2 block tensor
template <typename T, typename U, std::size_t Rows, std::size_t Common,
          std::size_t Cols>
inline auto operator*(const BlockTensor<T, Rows, Common> &lhs,
                      const BlockTensor<U, Common, Cols> &rhs) {
  BlockTensor<typename std::common_type<T, U>::type, Rows, Cols> result;
  for (std::size_t row = 0; row < Rows; ++row)
    for (std::size_t col = 0; col < Cols; ++col) {
      T tmp = torch::mul(*lhs[Common * row], *rhs[col]);
      for (std::size_t idx = 1; idx < Common; ++idx)
        tmp += torch::mul(*lhs[Common * row + idx], *rhs[Cols * idx + col]);
      result[Cols * row + col] = std::make_shared<T>(tmp);
    }
  return result;
}

/// @brief Compile-time rank-3 block tensor (tensor)
///
/// Data is store in row-major order, i.e. all entries of a row are
/// stored contiguously in memory and the entries of the next row
/// are stored with an offset of Cols. The entries of the next slice
/// are store with an offset of Rows*Cols.
template <typename T, std::size_t Rows, std::size_t Cols, std::size_t Slices>
class BlockTensor<T, Rows, Cols, Slices>
    : public BlockTensorCore<T, Rows, Cols, Slices> {
private:
  using Base = BlockTensorCore<T, Rows, Cols, Slices>;

public:
  using BlockTensorCore<T, Rows, Cols, Slices>::BlockTensorCore;

  /// @brief Returns the number of rows
  inline static constexpr std::size_t rows() { return Rows; }

  /// @brief Returns the number of columns
  inline static constexpr std::size_t cols() { return Cols; }

  /// @brief Returns the number of slices
  inline static constexpr std::size_t slices() { return Slices; }

  using Base::operator();

  /// @brief Returns a constant reference to entry (row, col, slice)
  inline const T &operator()(std::size_t row, std::size_t col,
                             std::size_t slice) const {
    assert(0 <= row && row < Rows && 0 <= col && col < Cols && 0 <= slice &&
           slice < Slices);
    return *Base::data_[Rows * Cols * slice + Cols * row + col];
  }

  /// @brief Returns a non-constant reference to entry (row, col, slice)
  inline T &operator()(std::size_t row, std::size_t col, std::size_t slice) {
    assert(0 <= row && row < Rows && 0 <= col && col < Cols && 0 <= slice &&
           slice < Slices);
    return *Base::data_[Rows * Cols * slice + Cols * row + col];
  }

  using Base::set;

  /// @brief Stores the given data object at the given position
  template <typename D>
  inline T &set(std::size_t row, std::size_t col, std::size_t slice, D &&data) {
    Base::data_[Rows * Cols * slice + Cols * row + col] =
        make_shared<D>(std::forward<D>(data));
    return *Base::data_[Rows * Cols * slice + Cols * row + col];
  }

  /// @brief Returns a rank-2 tensor of the k-th slice
  inline auto slice(std::size_t slice) const {
    assert(0 <= slice && slice < Slices);
    BlockTensor<T, Rows, Cols> result;
    for (std::size_t row = 0; row < Rows; ++row)
      for (std::size_t col = 0; col < Cols; ++col)
        result[Cols * row + col] =
            Base::data_[Rows * Cols * slice + Cols * row + col];
    return result;
  }

  /// @brief Returns a new block vector with rows, columns, and
  ///  slices permuted according to (i,j,k) -> (i,k,j)
  inline auto reorder_ikj() const {
    BlockTensor<T, Rows, Slices, Cols> result;
    for (std::size_t slice = 0; slice < Slices; ++slice)
      for (std::size_t row = 0; row < Rows; ++row)
        for (std::size_t col = 0; col < Cols; ++col)
          result[Rows * Slices * col + Slices * row + slice] =
              Base::data_[Rows * Cols * slice + Cols * row + col];
    return result;
  }

  /// @brief Returns a new block vector with rows and columns
  /// transposed and slices remaining fixed. This is equivalent to
  /// looping through all slices and transposing each rank-2 tensor.
  inline auto reorder_jik() const {
    BlockTensor<T, Cols, Rows, Slices> result;
    for (std::size_t slice = 0; slice < Slices; ++slice)
      for (std::size_t row = 0; row < Rows; ++row)
        for (std::size_t col = 0; col < Cols; ++col)
          result[Rows * Cols * slice + Rows * col + row] =
              Base::data_[Rows * Cols * slice + Cols * row + col];
    return result;
  }

  /// @brief Returns a new block vector with rows, columns, and
  ///  slices permuted according to (i,j,k) -> (k,j,i)
  inline auto reorder_kji() const {
    BlockTensor<T, Slices, Cols, Rows> result;
    for (std::size_t slice = 0; slice < Slices; ++slice)
      for (std::size_t row = 0; row < Rows; ++row)
        for (std::size_t col = 0; col < Cols; ++col)
          result[Slices * Cols * row + Cols * slice + col] =
              Base::data_[Rows * Cols * slice + Cols * row + col];
    return result;
  }

  /// @brief Returns a new block vector with rows, columns, and
  ///  slices permuted according to (i,j,k) -> (k,i,j)
  inline auto reorder_kij() const {
    BlockTensor<T, Slices, Rows, Cols> result;
    for (std::size_t slice = 0; slice < Slices; ++slice)
      for (std::size_t row = 0; row < Rows; ++row)
        for (std::size_t col = 0; col < Cols; ++col)
          result[Slices * Rows * col + Rows * slice + row] =
              Base::data_[Rows * Cols * slice + Cols * row + col];
    return result;
  }

  /// Returns a string representation of the BSplineCommon object
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << Base::name() << "\n";
    for (std::size_t slice = 0; slice < Slices; ++slice)
      for (std::size_t row = 0; row < Rows; ++row)
        for (std::size_t col = 0; col < Cols; ++col)
          os << "[" << row << "," << col << "," << slice << "] = \n"
             << *Base::data_[Rows * Cols * slice + Cols * row + col] << "\n";
  }
};

/// @brief Multiplies one compile-time rank-2 block tensor from the
/// left with a compile-time rank-3 block tensor slice-by-slice
template <typename T, typename U, std::size_t Rows, std::size_t Common,
          std::size_t Cols, std::size_t Slices>
inline auto operator*(const BlockTensor<T, Rows, Common> &lhs,
                      const BlockTensor<U, Common, Cols, Slices> &rhs) {
  BlockTensor<typename std::common_type<T, U>::type, Rows, Cols, Slices> result;
  for (std::size_t slice = 0; slice < Slices; ++slice)
    for (std::size_t row = 0; row < Rows; ++row)
      for (std::size_t col = 0; col < Cols; ++col) {
        T tmp = torch::mul(*lhs[Common * row], *rhs[Rows * Cols * slice + col]);
        for (std::size_t idx = 1; idx < Common; ++idx)
          tmp += torch::mul(*lhs[Common * row + idx],
                            *rhs[Rows * Cols * slice + Cols * idx + col]);
        result[Rows * Cols * slice + Cols * row + col] =
            std::make_shared<T>(tmp);
      }
  return result;
}

/// @brief Multiplies one compile-time rank-3 block tensor from the
/// left with a compile-time rank-2 block tensor slice-by-slice
template <typename T, typename U, std::size_t Rows, std::size_t Common,
          std::size_t Cols, std::size_t Slices>
inline auto operator*(const BlockTensor<T, Rows, Common, Slices> &lhs,
                      const BlockTensor<U, Common, Cols> &rhs) {
  BlockTensor<typename std::common_type<T, U>::type, Rows, Cols, Slices> result;
  for (std::size_t slice = 0; slice < Slices; ++slice)
    for (std::size_t row = 0; row < Rows; ++row)
      for (std::size_t col = 0; col < Cols; ++col) {
        T tmp = torch::mul(*lhs[Rows * Cols * slice + Common * row], *rhs[col]);
        for (std::size_t idx = 1; idx < Common; ++idx)
          tmp += torch::mul(*lhs[Rows * Cols * slice + Common * row + idx],
                            *rhs[Cols * idx + col]);
        result[Rows * Cols * slice + Cols * row + col] =
            std::make_shared<T>(tmp);
      }
  return result;
}

#define unary_op(name)                                                         \
  template <typename T, std::size_t... Dims>                                   \
  inline auto name(const BlockTensor<T, Dims...> &input) {                     \
    BlockTensor<T, Dims...> result;                                            \
    for (std::size_t idx = 0; idx < (Dims * ...); ++idx)                       \
      result[idx] = std::make_shared<T>(torch::name(*input[idx]));             \
    return result;                                                             \
  }

#define unary_special_op(name)                                                 \
  template <typename T, std::size_t... Dims>                                   \
  inline auto name(const BlockTensor<T, Dims...> &input) {                     \
    BlockTensor<T, Dims...> result;                                            \
    for (std::size_t idx = 0; idx < (Dims * ...); ++idx)                       \
      result[idx] = std::make_shared<T>(torch::special::name(*input[idx]));    \
    return result;                                                             \
  }

#define binary_op(name)                                                        \
  template <typename T, typename U, std::size_t... Dims>                       \
  inline auto name(const BlockTensor<T, Dims...> &input,                       \
                   const BlockTensor<U, Dims...> &other) {                     \
    BlockTensor<typename std::common_type<T, U>::type, Dims...> result;        \
    for (std::size_t idx = 0; idx < (Dims * ...); ++idx)                       \
      result[idx] =                                                            \
          std::make_shared<T>(torch::name(*input[idx], *other[idx]));          \
    return result;                                                             \
  }

#define binary_special_op(name)                                                \
  template <typename T, typename U, std::size_t... Dims>                       \
  inline auto name(const BlockTensor<T, Dims...> &input,                       \
                   const BlockTensor<U, Dims...> &other) {                     \
    BlockTensor<typename std::common_type<T, U>::type, Dims...> result;        \
    for (std::size_t idx = 0; idx < (Dims * ...); ++idx)                       \
      result[idx] =                                                            \
          std::make_shared<T>(torch::special::name(*input[idx], *other[idx])); \
    return result;                                                             \
  }

/// @brief Returns a new block tensor with the absolute value of the
/// elements of `input`
unary_op(abs);

/// @brief Alias for `abs()`
unary_op(absolute);

/// @brief Returns a new block tensor with the inverse cosine of the
/// elements of `input`
unary_op(acos);

/// @brief Alias for `acos()`
unary_op(arccos);

/// @brief Returns a new block tensor with the inverse hyperbolic
/// cosine of the elements of `input`
unary_op(acosh);

/// @brief Alias for acosh()`
unary_op(arccosh);

/// @brief Returns a new block tensor with the elements of `other`,
/// scaled by `alpha`, added to the elements of `input`
template <typename T, typename U, typename V, std::size_t... Dims>
inline auto add(const BlockTensor<T, Dims...> &input,
                const BlockTensor<U, Dims...> &other, V alpha = 1.0) {
  BlockTensor<typename std::common_type<T, U>::type, Dims...> result;
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    result[idx] =
        std::make_shared<T>(torch::add(*input[idx], *other[idx], alpha));
  return result;
}

/// @brief Returns a new block tensor with the elements of `other`,
/// scaled by `alpha`, added to the elements of `input`
template <typename T, typename U, typename V, std::size_t... Dims>
inline auto add(const BlockTensor<T, Dims...> &input, U other, V alpha = 1.0) {
  BlockTensor<T, Dims...> result;
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    result[idx] = std::make_shared<T>(torch::add(*input[idx], other, alpha));
  return result;
}

/// @brief Returns a new block tensor with the elements of `other`,
/// scaled by `alpha`, added to the elements of `input`
template <typename T, typename U, typename V, std::size_t... Dims>
inline auto add(T input, const BlockTensor<U, Dims...> &other, V alpha = 1.0) {
  BlockTensor<U, Dims...> result;
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    result[idx] = std::make_shared<T>(torch::add(input, *other[idx], alpha));
  return result;
}

/// @brief Returns a new block tensor with the elements of `tensor1`
/// divided by the elements of `tensor2`, with the result multiplied
/// by the scalar `value` and added to the elements of `input`
template <typename T, typename U, typename V, typename W, std::size_t... Dims>
inline auto addcdiv(const BlockTensor<T, Dims...> &input,
                    const BlockTensor<U, Dims...> &tensor1,
                    const BlockTensor<V, Dims...> &tensor2, W value = 1.0) {
  BlockTensor<typename std::common_type<T, U, V>::type, Dims...> result;
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    result[idx] = std::make_shared<T>(
        torch::addcdiv(*input[idx], *tensor1[idx], *tensor2[idx], value));
  return result;
}

/// @brief Returns a new block tensor with the elements of `tensor1`
/// multiplied by the elements of `tensor2`, with the result
/// multiplied by the scalar `value` and added to the elements of
/// `input`
template <typename T, typename U, typename V, typename W, std::size_t... Dims>
inline auto addcmul(const BlockTensor<T, Dims...> &input,
                    const BlockTensor<U, Dims...> &tensor1,
                    const BlockTensor<V, Dims...> &tensor2, W value = 1.0) {
  BlockTensor<typename std::common_type<T, U, V>::type, Dims...> result;
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    result[idx] = std::make_shared<T>(
        torch::addcmul(*input[idx], *tensor1[idx], *tensor2[idx], value));
  return result;
}

/// @brief Returns a new block tensor with the angle (in radians) of
/// the elements of `input`
unary_op(angle);

/// @brief Returns a new block tensor with the arcsine of the
/// elements of `input`
unary_op(asin);

/// @brief Alias for asin()
unary_op(arcsin);

/// @brief Returns a new block tensor with the inverse hyperbolic
/// sine of the elements of `input`
unary_op(asinh);

/// @brief Alias for asinh()
unary_op(arcsinh);

/// @brief Returns a new block tensor with the arctangent of the
/// elements of `input`
unary_op(atan);

/// @brief Alias for atan()
unary_op(arctan);

/// @brief Returns a new block tensor with the inverse hyperbolic
/// tangent of the elements of `input`
unary_op(atanh)

    /// @brief Alias for atanh()
    unary_op(arctanh);

/// @brief Returns a new block tensor with the arctangent of the
/// elements in `input` and `other` with consideration of the
/// quadrant
binary_op(atan2);

#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11 ||                   \
    TORCH_VERSION_MAJOR >= 2
/// @brief Alias for atan2()
binary_op(arctan2);
#endif

/// @brief Returns a new block tensor with the bitwise NOT of the
/// elements of `input`
unary_op(bitwise_not);

/// @brief Returns a new block tensor with the bitwise AND of the
/// elements of `input` and `other`
binary_op(bitwise_and);

/// @brief Returns a new block tensor with the bitwise OR of the
/// elements of `input` and `other`
binary_op(bitwise_or);

/// @brief Returns a new block tensor with the bitwise XOR of the
/// elements of `input` and `other`
binary_op(bitwise_xor);

/// @brief Returns a new block tensor with the left arithmetic shift
/// of the elements of `input` by `other` bits
binary_op(bitwise_left_shift);

/// @brief Returns a new block tensor with the right arithmetic
/// shift of the element of `input` by `other` bits
binary_op(bitwise_right_shift);

/// @brief Returns a new block tensor with the ceil of the elements of
/// input, the smallest integer greater than or equal to each
/// element
unary_op(ceil);

/// @brief Returns a new block tensor with the elements of `input`
/// clamped into the range [ min, max ]
template <typename T, typename U, std::size_t... Dims>
inline auto clamp(const BlockTensor<T, Dims...> &input, U min, U max) {
  BlockTensor<T, Dims...> result;
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    result[idx] = std::make_shared<T>(torch::clamp(*input[idx], min, max));
  return result;
}

/// @brief Alias for clamp()
template <typename T, typename U, std::size_t... Dims>
inline auto clip(const BlockTensor<T, Dims...> &input, U min, U max) {
  BlockTensor<T, Dims...> result;
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    result[idx] = std::make_shared<T>(torch::clip(*input[idx], min, max));
  return result;
}

/// @brief Returns a new block tensor with the conjugate of the
/// elements of `input` tensor
unary_op(conj_physical);

/// @brief Returns a new block tensor with the magnitude of the
/// elements of `input` and the sign of the elements of `other`
binary_op(copysign);

/// @brief Returns a new block tensor with the cosine of the
/// elements of `input`
unary_op(cos);

/// @brief Returns a new block tensor with the hyperbolic cosine of
/// the elements of `input`
unary_op(cosh);

/// @brief Returns a new block tensor with the elements of `input`
/// converted from angles in degrees to radians
unary_op(deg2rad)

    /// @brief Returns a new block tensor with the elements of `input`
    /// divided by the elements of `other`
    binary_op(div);

/// @brief Alias for div()
binary_op(divide);

/// @brief Returns a new block tensor with the logarithmic
/// derivative of the gamma function of the elements of `input`
unary_op(digamma);

/// @brief Returns a new block tensor with the error function of the
/// elements of `input`
unary_op(erf);

/// @brief Returns a new block tensor with the complementary error
/// function of the elements of `input`
unary_op(erfc);

/// @brief Returns a new block tensor with the inverse error
/// function of the elements of `input`
unary_op(erfinv);

/// @brief Returns a new block tensor with the exponential of the
/// elements of `input`
unary_op(exp);

/// @brief Returns a new block tensor with the base-2 exponential of
/// the elements of `input`
unary_op(exp2);

/// @brief Returns a new block tensor with the exponential minus 1
/// of the elements of `input`
unary_op(expm1);

/// @brief Alias for trunc()
unary_op(fix);

/// @brief Returns a new block tensor with the elements of `input`
/// raised to the power of `exponent`, elementwise, in double
/// precision
binary_op(float_power);

/// @brief Returns a new block tensor with the floor of the elements
/// of `input`, the largest integer less than or equal to each element
unary_op(floor);

/// @brief Returns a new block tensor with the fmod of the elements
/// of `input` and `other`
binary_op(fmod);

/// @brief Returns a new block tensor with the fractional portion of
/// the elements of `input`
unary_op(frac);

/// @brief Returns a new block tensor with the decomposition of the
/// elements of `input` into mantissae and exponents
unary_op(frexp);

/// @brief Returns a new block tensor with the imaginary values of
/// the elements of `input`
unary_op(imag);

/// @brief Returns a new block tensor with the elements of `input`
/// multiplied by 2**other
binary_op(ldexp);

/// @brief Returns a new block tensor with the natural logarithm of
/// the absolute value of the gamma function of the elements of
/// `input`
unary_op(lgamma);

/// @brief Returns a new block tensor with the natural logarithm of
/// the elements of `input`
unary_op(log);

/// @brief Returns a new block tensor with the logarithm to the
/// base-10 of the elements of `input`
unary_op(log10);

/// @brief Returns a new block tensor with the natural logarithm of
/// (1 + the elements of `input`)
unary_op(log1p);

/// @brief Returns a new block tensor with the logarithm to the
/// base-2 of the elements of `input`
unary_op(log2);

/// @brief Returns a new block-vector with the logarithm of the sum
/// of exponentiations of the elements of `input`
binary_op(logaddexp);

/// @brief Returns a new block-vector with the logarithm of the sum
/// of exponentiations of the elements of `input` in base-2
binary_op(logaddexp2);

/// @brief Returns a new block tensor with the element-wise logical
/// AND of the elements of `input` and `other`
binary_op(logical_and)

    /// @brief Returns a new block tensor with the element-wise logical
    /// NOT of the elements of `input`
    unary_op(logical_not)

    /// @brief Returns a new block tensor with the element-wise logical
    /// OR of the elements of `input` and `other`
    binary_op(logical_or)

    /// @brief Returns a new block tensor with the element-wise logical
    /// XOR of the elements of `input` and `other`
    binary_op(logical_xor);

/// logit

/// @brief Given the legs of a right triangle, return its hypotenuse
binary_op(hypot);

/// @brief Returns a new block tensor with the element-wise zeroth
/// order modified Bessel function of the first kind for each
/// element of `input`
unary_op(i0);

/// @brief Returns a new block tensor with the regularized lower
/// incomplete gamma function of each element of `input`
binary_special_op(gammainc);

/// @brief Alias for gammainc()
binary_op(igamma);

/// @brief Returns a new block tensor with the regularized upper
/// incomplete gamma function of each element of `input`
binary_special_op(gammaincc);

/// @brief Alias for gammainc()
binary_op(igammac);

/// @brief Returns a new block tensor with the product of each
/// element of `input` and `other`
binary_op(mul);

/// @brief Alias for mul()
binary_op(multiply);

/// @brief Returns a new block tensor with the negative of the
/// elements of `input`
unary_op(neg);

/// @brief Alias for neg()
unary_op(negative);

/// @brief Return a new block tensor with the next elementwise
/// floating-point value after `input` towards `other`
binary_op(nextafter);

/// @brief Returns a new block tensor with the `input`
unary_op(positive);

/// @brief Returns a new block tensor with the power of each element
/// in `input` with exponent `other`
binary_op(pow);

/// @brief Returns a new block tensor with each of the elements of
/// `input` converted from angles in radians to degrees
unary_op(rad2deg);

/// @brief Returns a new block tensor with the real values of the
/// elements of `input`
unary_op(real);

/// @brief Returns a new block tensor with the reciprocal of the
/// elements of `input`
unary_op(reciprocal);

/// @brief Returns a new block tensor with the modulus of the
/// elements of `input`
binary_op(remainder);

/// @brief Returns a new block tensor with the elements of `input`
/// rounded to the nearest integer
unary_op(round);

/// @brief Returns a new block tensor with the reciprocal of the
/// square-root of the elements of `input`
unary_op(rsqrt);

/// @brief Returns a new block tensor with the expit (also known as
/// the logistic sigmoid function) of the elements of `input`
unary_special_op(expit);

/// @brief Alias for expit()
unary_op(sigmoid);

/// @brief Returns a new block tensor with the signs of the elements
/// of `input`
unary_op(sign);

/// @brief Returns a new block tensor with the signs of the elements
/// of `input`, extension to complex value
unary_op(sgn);

/// @brief Tests if each element of `input` has its sign bit set
/// (is less than zero) or not
unary_op(signbit);

/// @brief Returns a new block tensor with the sine of the elements
/// of `input`
unary_op(sin);

/// @brief Returns a new block tensor with the normalized sinc of
/// the elements of `input`
unary_op(sinc);

/// @brief Returns a new block tensor with the hyperbolic sine of
/// the elements of `input`
unary_op(sinh);

/// @brief Returns a new block tensor with the square-root of the
/// elements of `input`
unary_op(sqrt);

/// @brief Returns a new block tensor with the square of the
/// elements of `input`
unary_op(square);

/// @brief Subtracts other, scaled by alpha, from input
template <typename T, typename U, typename V, std::size_t... Dims>
inline auto sub(const BlockTensor<T, Dims...> &input,
                const BlockTensor<U, Dims...> &other, V alpha = 1.0) {
  BlockTensor<typename std::common_type<T, U>::type, Dims...> result;
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    result[idx] =
        std::make_shared<T>(torch::sub(*input[idx], *other[idx], alpha));
  return result;
}

/// @brief Alias for sub()
template <typename T, typename U, typename V, std::size_t... Dims>
inline auto subtract(const BlockTensor<T, Dims...> &input,
                     const BlockTensor<U, Dims...> &other, V alpha = 1.0) {
  BlockTensor<typename std::common_type<T, U>::type, Dims...> result;
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    result[idx] =
        std::make_shared<T>(torch::sub(*input[idx], *other[idx], alpha));
  return result;
}

/// @brief Returns a new tensor with the tangent of the elements of
/// input
unary_op(tan);

/// @brief Returns a new tensor with the hyperbolic tangent of the
/// elements of input
unary_op(tanh);

/// @brief Returns a new tensor with the truncated integer values of
/// the elements of input
unary_op(trunc)

    /// @brief Computes input * log(other)
    binary_op(xlogy);

/// @brief Adds one compile-time block tensor to another and returns
/// a new compile-time block tensor
template <typename T, typename U, std::size_t... Dims>
inline auto operator+(const BlockTensor<T, Dims...> &lhs,
                      const BlockTensor<U, Dims...> &rhs) {
  BlockTensor<typename std::common_type<T, U>::type, Dims...> result;
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    result[idx] = std::make_shared<T>(*lhs[idx] + *rhs[idx]);
  return result;
}

/// @brief Adds a compile-time block tensor to a scalar and returns
/// a new compile-time block tensor
template <typename T, typename U, std::size_t... Dims>
inline auto operator+(const BlockTensor<T, Dims...> &lhs, const U &rhs) {
  BlockTensor<T, Dims...> result;
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    result[idx] = std::make_shared<T>(*lhs[idx] + rhs);
  return result;
}

/// @brief Adds a scalar to a compile-time block tensor and returns
/// a new compile-time block tensor
template <typename T, typename U, std::size_t... Dims>
inline auto operator+(const T &lhs, const BlockTensor<U, Dims...> &rhs) {
  BlockTensor<U, Dims...> result;
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    result[idx] = std::make_shared<U>(lhs + *rhs[idx]);
  return result;
}

/// @brief Increments one compile-time block tensor by another
template <typename T, typename U, std::size_t... Dims>
inline auto operator+=(BlockTensor<T, Dims...> &lhs,
                       const BlockTensor<U, Dims...> &rhs) {
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    lhs[idx] = std::make_shared<T>(*lhs[idx] + *rhs[idx]);
  return lhs;
}

/// @brief Increments a compile-time block tensor by a scalar
template <typename T, typename U, std::size_t... Dims>
inline auto operator+=(BlockTensor<T, Dims...> &lhs, const U &rhs) {
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    lhs[idx] = std::make_shared<T>(*lhs[idx] + rhs);
  return lhs;
}

/// @brief Subtracts one compile-time block tensor from another and
/// returns a new compile-time block tensor
template <typename T, typename U, std::size_t... Dims>
inline auto operator-(const BlockTensor<T, Dims...> &lhs,
                      const BlockTensor<U, Dims...> &rhs) {
  BlockTensor<typename std::common_type<T, U>::type, Dims...> result;
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    result[idx] = std::make_shared<T>(*lhs[idx] - *rhs[idx]);
  return result;
}

/// @brief Subtracts a scalar from a compile-time block tensor and returns
/// a new compile-time block tensor
template <typename T, typename U, std::size_t... Dims>
inline auto operator-(const BlockTensor<T, Dims...> &lhs, const U &rhs) {
  BlockTensor<T, Dims...> result;
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    result[idx] = std::make_shared<T>(*lhs[idx] - rhs);
  return result;
}

/// @brief Subtracts a compile-time block tensor from a scalar and
/// returns a new compile-time block tensor
template <typename T, typename U, std::size_t... Dims>
inline auto operator-(const T &lhs, const BlockTensor<U, Dims...> &rhs) {
  BlockTensor<U, Dims...> result;
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    result[idx] = std::make_shared<U>(lhs - *rhs[idx]);
  return result;
}

/// @brief Decrements one compile-time block tensor by another
template <typename T, typename U, std::size_t... Dims>
inline auto operator-=(BlockTensor<T, Dims...> &lhs,
                       const BlockTensor<U, Dims...> &rhs) {
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    lhs[idx] = std::make_shared<T>(*lhs[idx] - *rhs[idx]);
  return lhs;
}

/// @brief Decrements a compile-time block tensor by a scalar
template <typename T, typename U, std::size_t... Dims>
inline auto operator-=(BlockTensor<T, Dims...> &lhs, const U &rhs) {
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    lhs[idx] = std::make_shared<T>(*lhs[idx] - rhs);
  return lhs;
}

/// @brief Multiplies a compile-time block tensor with a scalar and
/// returns a new compile-time block tensor
template <typename T, typename U, std::size_t... Dims>
inline auto operator*(const BlockTensor<T, Dims...> &lhs, const U &rhs) {
  BlockTensor<T, Dims...> result;
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    result[idx] = std::make_shared<T>(*lhs[idx] + rhs);
  return result;
}

/// @brief Multiplies a scalar with a compile-time block tensor and
/// returns a new compile-time block tensor
template <typename T, typename U, std::size_t... Dims>
inline auto operator*(const T &lhs, const BlockTensor<U, Dims...> &rhs) {
  BlockTensor<U, Dims...> result;
  for (std::size_t idx = 0; idx < (Dims * ...); ++idx)
    result[idx] = std::make_shared<U>(lhs * *rhs[idx]);
  return result;
}

/// @brief Returns true if both compile-time block tensors are equal
template <typename T, typename U, std::size_t... TDims, std::size_t... UDims>
inline bool operator==(const BlockTensor<T, TDims...> &lhs,
                       const BlockTensor<U, UDims...> &rhs) {
  if constexpr ((sizeof...(TDims) != sizeof...(UDims)) ||
                ((TDims != UDims) || ...))
    return false;

  bool result = true;
  for (std::size_t idx = 0; idx < (TDims * ...); ++idx)
    result = result && torch::equal(*lhs[idx], *rhs[idx]);

  return result;
}

/// @brief Returns true if both compile-time block tensors are not equal
template <typename T, typename U, std::size_t... TDims, std::size_t... UDims>
inline bool operator!=(const BlockTensor<T, TDims...> &lhs,
                       const BlockTensor<U, UDims...> &rhs) {
  return !(lhs == rhs);
}

} // namespace utils
} // namespace iganet
