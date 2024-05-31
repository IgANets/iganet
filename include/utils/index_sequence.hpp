/**
   @file include/utils/index_sequence.hpp

   @brief Integer sequence utility function

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <utility>

namespace iganet {
namespace utils {

  namespace detail {

    /// @brief Reverse index sequence helper
    template <std::size_t, typename>
    struct make_reverse_index_sequence_helper;

    template <std::size_t N, std::size_t...NN>
    struct make_reverse_index_sequence_helper<N, std::index_sequence<NN...>> 
      : std::index_sequence<(N - NN)...> {};
    
  } // namespace detail

  /// @brief Reverse index sequence
  template <std::size_t N>
  struct make_reverse_index_sequence 
    : detail::make_reverse_index_sequence_helper<N - 1, 
        decltype(std::make_index_sequence<N>{})> {};   
  
} // namespace utils
} // namespace iganet
