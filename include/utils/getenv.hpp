/**
   @file include/utils/getenv.hpp

   @brief Environment utility function

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <sstream>
#include <stdlib.h>

namespace iganet {
namespace utils {

/// @brief Returns the value from an environment variable
template <typename T> T getenv(std::string variable, const T &default_value) {

  char *env_value = std::getenv(variable.c_str());
  if (env_value != NULL)
    if constexpr (std::is_integral_v<T>)
      return static_cast<T>(std::atoll(env_value));
    else if constexpr (std::is_floating_point_v<T>)
      return static_cast<T>(std::atof(env_value));
    else
      return static_cast<T>(env_value);
  else
    return default_value;
}

/// @brief Returns the value from an environment variable
template <typename T>
std::vector<T> getenv(std::string variable,
                      std::initializer_list<T> default_value) {

  char *env_value = std::getenv(variable.c_str());
  if (env_value != NULL) {
    std::stringstream ss(env_value);
    std::vector<T> result;
    std::string item;

    if constexpr (std::is_integral_v<T>)
      while (std::getline(ss, item, ','))
        result.emplace_back(static_cast<T>(std::atoll(item.c_str())));
    else if constexpr (std::is_floating_point_v<T>)
      while (std::getline(ss, item, ','))
        result.emplace_back(static_cast<T>(std::atof(item.c_str())));
    else
      while (std::getline(ss, item, ','))
        result.emplace_back(static_cast<T>(item));
    return result;
  } else
    return std::vector<T>{default_value};
}

} // namespace utils
} // namespace iganet
