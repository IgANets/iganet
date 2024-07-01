/**
   @file include/utils/type_traits.hpp

   @brief Type traits

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <type_traits>

namespace iganet {
namespace utils {

/// @brief Type trait for std::tuple type
/// @{
template <class T> struct is_tuple : std::false_type {};

template <class... Ts> struct is_tuple<std::tuple<Ts...>> : std::true_type {};
/// @}

template <class T> inline constexpr bool is_tuple_v = is_tuple<T>::value;

/// @brief Type trait for std::tuple<std::tuple> type
/// @{
template <class T> struct is_tuple_of_tuples : std::false_type {};

template <class... Ts>
struct is_tuple_of_tuples<std::tuple<Ts...>>
    : std::common_type<is_tuple<Ts>...>::type {};
/// @}

/// @brief Alias for is_tuple_of_tuples::type
template <class T>
using is_tuple_of_tuples_t = typename is_tuple_of_tuples<T>::type;

/// @brief Alias for is_tuple_of_tuples::value
template <class T>
inline constexpr auto is_tuple_of_tuples_v = is_tuple_of_tuples<T>::value;

/// @brief Type trait for concatenating std::tuples
/// @{
template <typename... Tuples> struct tuple_cat;

template <> struct tuple_cat<> {
  using type = std::tuple<>;
};

template <typename... Ts, typename... Tuples>
struct tuple_cat<std::tuple<Ts...>, Tuples...> {
  using type = decltype(std::tuple_cat(
      std::declval<std::tuple<Ts...>>(),
      std::declval<typename tuple_cat<Tuples...>::type>()));
};

template <typename T, typename... Tuples> struct tuple_cat<T, Tuples...> {
  using type = decltype(std::tuple_cat(
      std::declval<std::tuple<T>>(),
      std::declval<typename tuple_cat<Tuples...>::type>()));
};
/// @}

/// @brief Alias for tuple_cat::type
template <typename... Tuples>
using tuple_cat_t = typename tuple_cat<Tuples...>::type;

/// @brief Alias for tuple_cat::value
template <typename... Tuples>
inline constexpr auto tuple_cat_v = tuple_cat<Tuples...>::value;

} // namespace utils
} // namespace iganet
