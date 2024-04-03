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

template <class T>
inline constexpr bool is_tuple_of_tuples_v = is_tuple_of_tuples<T>::value;

} // namespace utils
} // namespace iganet
