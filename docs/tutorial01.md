# Tutorial 01: Hello IgANets

## TL;DR

In this tutorial you will get to know the structure of **IgANets** and
how to write your first program.

## Code repository

The source code can be obtained from https://github.com/IgANets/IgaNet
as well as https://gitlab.com/iganets/iganet.

It contains the following sub-folders:

-   `docs` contains the documentation

-   `examples` contains several examples

-   `filedata` contains some example data files

-   `include` contains the IgANet header files

-   `mex` contains some Matlab MEX functions

-   `ops` contains files for creating Docker images

-   `perftest` contains some performance tests

-   `python` contains the Python wrapper

-   `unittests` contains the unit tests

-   `webapps` contains the WebApp code

## Getting started

In order to utilize IgANets in your application you need to

1.  include the main header file `iganet.h`,
2.  call `iganet::init()` to initialize some internals at the start of the program, and
3.  call `iganet::finalize()` to clean up internals at the end of the program

The following toy application

```cpp
#include <iganet.h>

// Include the IgANet namespace
using namespace iganet;

int main() {
  // Initialize internals
  init();

  // Clean up internals
  finalize();

  return 0;
}
```

will output some information about the system it is run on, e.g.,

```
[INFO] IgANets - Isogeometric Analysis Networks (version 24.04.0)
Compiled by AppleClang 16.0.0.16000026 (C++ 202002, libc++ 180100, LibTorch 2.4.1)
Running on Apple M1 (memory 8 GB, #intraop threads: 8, #interop threads: 1, devices: CPU, MPS)
[INFO] Succeeded
```

The above information may vary depending on the compiler, the LibTorch
version, the CMake settings, and the system configuration.

### Controlling multi-threading on CPUs

If your application is compiled with `IGANET_WITH_OPENMP=YES` you can
set the number of OpenMP threads by running your application with

```bash
OMP_NUM_THREADS=2 ./iganet
```

Note that by default all OpenMP threads are used for intra-op
parallelization and the number of threads for inter-op parallelization
is set to one. The difference between intra- and inter-op
parallelization is explained in the [PyTorch
documentation](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html). You can change this behavior by running your application with

```bash
IGANET_INTRAOP_NUM_THREADS=3 IGANET_INTEROP_NUM_THREADS=2 ./iganet
```

which will change the information to

```
Running on Apple M1 (memory 8 GB, #intraop threads: 3, #interop threads: 2, devices: CPU, MPS)
```

### Setting the default device

IgANets checks all available devices during initialization. Unless the
environment variables `IGANET_DEVICE` and/or `IGANET_DEVICE_INDEX` are
set, it uses the first GPU as default device and falls back to CPU
mode otherwise.

The following device types are supported depending on your system

| `IGANET_DEVICE` | description |
|:---:|:---|
| `CUDA` | NVIDIA GPUs, requires CUDA-enabled LibTorch library |
| `HIP`  | AMD GPUs, requires ROCM-enabled LibTorch library    |
| `MPS`  | Apple Silicon Metal Performance Shaders             |
| `XLA`  | XLA devices such as Google's TPUs, requires XLA-enabled LibTorch library, see [documentation](https://github.com/pytorch/xla) |
| `XPU`  | Intel GPUs, see [documentation](https://pytorch.org/docs/stable/notes/get_start_xpu.html) |

### The `Options` class

If you want to know the default configuration of your system you can
create an `Options` object and print it

\snippet tutorial01.cxx Options

which might give you the following output

```
iganet::Options<double>(
options = TensorOptions(dtype=double, device=cpu, layout=Strided (default), requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt))
)
```

All parameters of the `Options` object can be retrieved individually, e.g.

\snippet tutorial01.cxx Print options

yields the output

```
cpu
-1
Double
Strided
0
0
0
```

The object's parameters cannot be changed in place. Instead, a new
`Options` object with changed parameters needs to be created, e.g.

\snippet tutorial01.cxx Derive options

yields the output

```
iganet::Options<float>(
options = TensorOptions(dtype=float, device=mps, layout=Strided (default), requires_grad=true (default), pinned_memory=false (default), memory_format=(nullopt))
)
```

### The logging mechanism

IgANets has its own logging mechanism. Instead of writing output to
`std::cout` and `std:cerr` it is recommended to write ouput to
`iganet::Log(...)` and specify the log level as follows

| log level | description |
|:---|:---|
| `iganet::log::none` | no logging |
| `iganet::log::fatal` | fatal error |
| `iganet::log::error` | error |
| `iganet::log::warning` | warning |
| `iganet::log::info` | information |
| `iganet::log::debug` | debug information |
| `iganet::log::verbose` | verbose information |

For example, the following code snippet

\snippet tutorial01.cxx Logging to screen

will print

```
[FATAL ERROR] Fatal error
[ERROR] Error
[WARNING] Warning
[INFO] Information
```

By default, `iganet::Log()` prints the output as `iganet::log::info`.

The log level can be set by calling `iganet::Log.setLogLevel(...)`
with the requested log level. Once set, only log levels of the set
type and below, i.e. more severe will be printed, e.g.,

\snippet tutorial01.cxx Log levels

will only print

```
[FATAL ERROR] Fatal error
[ERROR] Error
```

Finally, it is possible to write the output into a logfile by calling

\snippet tutorial01.cxx Logging to file

This will create a file `output.log` with the content

```
[FATAL ERROR] Fatal error
[ERROR] Error
```
