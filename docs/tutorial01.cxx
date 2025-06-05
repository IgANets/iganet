//! [Include namespace]
#include <iganet.h>

using namespace iganet;
//! [Include namespace]

int main() {
  //! [Initialize internals]
  init();
  //! [Initialize internals]

  //! [Options]
  Options<double> options;
  std::cout << options;
  //! [Options]

  //! [Print options]
  std::cout << options.device() << "\n"
            << options.device_index() << "\n"
            << options.dtype() << "\n"
            << options.layout() << "\n"
            << options.requires_grad() << "\n"
            << options.pinned_memory() << "\n"
            << options.is_sparse() << "\n";
  //! [Print options]

  //! [Derive options]
  auto options_ =
      options.device(torch::kMPS).dtype<float>().requires_grad(true);
  std::cout << options_;
  //! [Derive options]

  //! [Logging to screen]
  Log(log::fatal) << "Fatal error\n";
  Log(log::error) << "Error\n";
  Log(log::warning) << "Warning\n";
  Log(log::info) << "Information\n";
  Log(log::debug) << "Debug information\n";
  Log(log::verbose) << "Verbose information\n";
  //! [Logging to screen]

  //! [Log levels]
  Log.setLogLevel(log::error);

  Log(log::fatal) << "Fatal error\n";
  Log(log::error) << "Error\n";
  Log(log::warning) << "Warning\n";
  Log(log::info) << "Information\n";
  Log(log::debug) << "Debug information\n";
  Log(log::verbose) << "Verbose information\n";
  //! [Log levels]

  //! [Logging to file]
  Log.setLogFile("output.log");

  Log(log::fatal) << "Fatal error\n";
  Log(log::error) << "Error\n";
  Log(log::warning) << "Warning\n";
  Log(log::info) << "Information\n";
  Log(log::debug) << "Debug information\n";
  Log(log::verbose) << "Verbose information\n";
  //! [Logging to file]

  //! [Clean up internals]
  finalize();
  //! [Clean up internals]

  return 0;
}
