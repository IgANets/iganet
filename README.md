# IgANets: Physics-informed isogeometric analysis networks

IgANets is a novel approach to combine the concept of deep operator
learning with the mathematical framework of isogeometric analysis.

## Installation instructions

IgANets require a C++17 compiler, CMake and LibTorch (the C++ API of
PyTorch).

Depending on the LibTorch version installed on your system,
IgANets will be compiled with support for CUDA, ROCm or the Intel
Extension for PyTorch. You can disable this feature by providing the
`-DIGANET_BUILD_CPUONLY=ON` flag to CMake.

By providing additional CMake flags you can configure IgANet to build the following optional components:

- `-DIGANET_BUILD_DOCS=ON` builds the documentation (default `OFF`). To build the documentation you need [Doxygen](https://www.doxygen.nl) and [Sphinx](https://www.sphinx-doc.org/en/master/) installed on you system.

- `-DIGANET_BUILD_EXAMPLES=ON` builds the examples (default `ON`).

- `-DIGANET_BUILD_PCH=ON` builds IgANets with precompiled headers (default `ON`).

- `-DIGANET_BUILD_PERFTESTS=ON` builds the performance tests (default `OFF`).

- `-DIGANET_BUILD_PYTHON=ON` builds the Python module `pyiganet` (default `OFF`). This option requires a Python interpreter to be installed on your system.

- `-DIGANET_BUILD_UNITTESTS=ON` builds the unit tests (default `OFF`).

- `-DIGANET_BUILD_WEBAPPS=ON` builds the websocket applications (default `OFF`).

In addition to the optional components, IgANets can be compiled with several optional features enabled/disabled:

- `-DIGANET_WITH_GISMO=ON` compiles IgANets with support for the open-source Geometry plus Simulation Modules library [G+Smo](https://github.com/gismo/gismo) enabled (default `OFF`).

- `-DIGANET_WITH_MATPLOT=ON` compiles IgANets with support for the open-source library [Matplot-cpp](https://github.com/lava/matplotlib-cpp) enabled (default `OFF`). _Note that this option can cause compilation errors with GCC._

- `-DIGANET_WITH_OPENMP=ON` compiles IgANets with OpenMP support enabled (default `ON`). _Note that this option can cause compilation errors with Clang._

### Linux

1.  Install prerequisites (CMake and LibTorch)

    #### Ubuntu
      ```shell
      apt-get install build-essential cmake unzip wget
      ```

    #### RedHat
      ```shell
      yum install make cmake gcc gcc-c++ unzip wget
      ```

    #### Install LibTorch

    Pre-compiled versions of LibTorch are available at [PyTorch.org](https://pytorch.org/get-started/locally/). Depending on your compiler toolchain you need to choose between the pre-cxx11 and the cxx11 ABI, i.e.

    ```shell
    wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.4.1%2Bcpu.zip -O libtorch.zip
    unzip libtorch.zip -d $HOME/
    rm -f libtorch
    ```
    or
    ```shell
    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcpu.zip -O libtorch.zip
    unzip libtorch.zip -d $HOME/
    rm -f libtorch
    ```

2.  Configure
    ```shell
    cmake .. -DTorch_DIR=${HOME}/libtorch/share/cmake/Torch
    ```

3.  Compile
    ```shell
    make -j 8
    ```

    Depending on the number of cores of your CPU you may want to change 8 to a different number.

### macOS

1.  Install prerequisites (CMake and LibTorch)
    ```shell
    brew install cmake pytorch
    ```

    Note that since version 2.2.0, official builds of the LibTorch library for ARM64 and X86_64 can be downloaded from PyTorch.org:

    - https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.4.1.zip
    - https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.4.1.zip

    If you decide to use these version download and unzip them as shown for the Linux installation. It is, however, recommended to install LibTorch through `brew` as described above since this method is tested regularly by the IgANets authors.

2.  Configure
    ```shell
    CMAKE_PREFIX_PATH=/opt/homebrew/Cellar/protobuf/28.2/share/protobuf cmake .. -DTorch_DIR=/opt/homebrew/Cellar/pytorch/2.4.1/share/cmake/Torch
    ```

    Note that the specific version of PyTorch and/or protobuf might be different on your system.

3.  Compile
    ```shell
    make
    ```

    Depending on the number of cores of your CPU you may want to change 8 to a different number.

## Compilation with CUDA support (only Linux)

1.  Install the CUDA-enabled version of LibTorch

    - https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.4.1%2Bcu121.zip
    - https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu121.zip

    Note that the version must be compatible with the CUDA version installed on your system.

2. Configure and compile

   All further steps are the same as described above (Linux)

## Compilation with ROCm support (only Linux)

1.  Install the ROCm-enabled version of LibTorch

    - https://download.pytorch.org/libtorch/rocm6.1/libtorch-shared-with-deps-2.4.1%2Brocm6.1.zip
    - https://download.pytorch.org/libtorch/rocm6.1/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Brocm6.1.zip

    Note that the version must be compatible with the ROCm version installed on your system.

2.  Configure
    ```shell
    CC=hipcc CXX=hipcc cmake .. -DTorch_DIR=$HOME/sfw/libtorch/2.4.1-latest-rocm61/share/cmake/Torch/ -DHIP_ROOT_DIR=/opt/rocm/hip -DIGANET_BUILD_PCH=OFF
    ```

    Note that the `HIP_ROOT_DIR` might be different on your system. Also note that building with precompiled headers does not work with the `hipcc` compiler so it must be disabled.

3. Compile

   ```shell
   make
   ```

   Depending on the number of cores of your CPU you may want to change 8 to a different number.


## Compilation with Intel Extensions for PyTorch support (Linux only)

1.  Install the Intel Extensions for PyTorch as described [here](https://github.com/intel/intel-extension-for-pytorch?tab=readme-ov-file).

2.  Add the CMake option `-DIPEX_DIR=<path/to/IPEX/installation>`

    Code Snippet for `liza.surf.nl`:
    ```shell
    module load intel/oneapi/tbb
    module load intel/oneapi/compiler-rt
    module load intel/oneapi/mkl

    cmake .. -DTorch_DIR=$HOME/sfw/libtorch/2.1.0-intel/share/cmake/Torch/ -DIPEX_DIR=$HOME/sfw/libtorch/2.1.0-intel/share/cmake/IPEX
    ```

## Python module

To compile the Python module `pyiganet` run
```shell
Torch_DIR=/opt/homebrew/Cellar/pytorch/2.1.2_1/share/cmake/Torch CMAKE_PREFIX_PATH=/opt/homebrew/Cellar/protobuf/25.2 python setup.py develop python setup.py develop
```

Again, the specific version of PyTorch and/or protobuf might be different on your system.

## Unit tests

To compile with unit tests enabled run CMake with the `-DIGANET_BUILD_UNITTESTS=ON` option and run
```shell
make test
```

## Performance tests

To compile with performance tests enabled run CMake with the `-DIGANET_BUILD_PERFTESTS=ON` option. By default, all performance tests are disabled and need to be enabled explicitly.

To obtain a list of available tests run (or another executable in the `perftests` folder)
```shell
./perftests/perftest_bspline_eval --gtest_filter="*" --gtest_list_tests
```

To execute one or more tests run
```shell
./perftests/perftest_bspline_eval --gtest_filter="*UniformBSpline_*parDim1*:-*Non*"
```

This specific command will run all `UniformBSpline` tests with 1 parametric dimension.
