/**
   @file mexiganet/src/mexbspline.cpp

   @brief MexIgANet multivariate B-splines

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <mexAdapter.hpp>

class Test {
public:
  Test() { std::cout << "Creating Test object\n"; }
};

class MexFunction : public matlab::mex::Function {
public:
  void operator()(matlab::mex::ArgumentList outputs,
                  matlab::mex::ArgumentList inputs) {

    checkArguments(outputs, inputs);
  }

  void checkArguments(matlab::mex::ArgumentList outputs,
                      matlab::mex::ArgumentList inputs) {
    // Get pointer to engine
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

    // Get array factory
    matlab::data::ArrayFactory factory;

    switch (inputs.size()) {

    case 1:

      break;

    default:
      matlabPtr->feval(u"error", 0,
                       std::vector<matlab::data::Array>({factory.createScalar(
                           "Input must be double array")}));
    }
  }

private:
  Test test;
};
