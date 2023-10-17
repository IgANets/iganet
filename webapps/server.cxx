/**
   @file webapps/server.cxx

   @brief Demonstration of a server application

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <App.h>
#include <iganet.h>
#include <modelmanager.hpp>
#include <popl.hpp>

#include <filesystem>

namespace iganet {
namespace webapp {

/// @brief Enumerator for specifying the status
enum class status : short_t {
  success = 0,                  /*!<  request was handled successfully */
  invalidRequest = 1,           /*!<  invalid request                  */
  invalidCreateRequest = 2,     /*!<  invalid create request           */
  invalidRemoveRequest = 3,     /*!<  invalid remove request           */
  invalidConnectRequest = 4,    /*!<  invalid connect request          */
  invalidDisconnectRequest = 5, /*!<  invalid disconnect request       */
  invalidGetRequest = 6,        /*!<  invalid get request              */
  invalidPutRequest = 7,        /*!<  invalid put request              */
  invalidEvalRequest = 8,       /*!<  invalid eval request             */
  invalidRefineRequest = 9,     /*!<  invalid refine request           */
  invalidLoadRequest = 10,      /*!<  invalid load request             */
  invalidSaveRequest = 11,      /*!<  invalid save request             */
  invalidImportRequest = 12,    /*!<  invalid import request           */
  invalidExportRequest = 13     /*!<  invalid export request           */
};

/// @brief InvalidSessionId exception
struct InvalidSessionIdException : public std::exception {
  const char *what() const throw() { return "Invalid session id"; }
};

/// @brief InvalidModelId exception
struct InvalidModelIdException : public std::exception {
  const char *what() const throw() { return "Invalid model id"; }
};

/// @brief InvalidModelType exception
struct InvalidModelTypeException : public std::exception {
  const char *what() const throw() { return "Invalid model type"; }
};

/// @brief Tokenize the input string
std::vector<std::string> tokenize(std::string str,
                                  std::string separator = "/") {
  std::vector<std::string> tokens;
  for (auto i = strtok(&str[0], &separator[0]); i != NULL;
       i = strtok(NULL, &separator[0]))
    tokens.push_back(i);
  return tokens;
}

/// @brief Session
template <typename T> struct Session {
private:
  /// @brief Session UUID
  const std::string uuid;

public:
  /// @brief Default constructor
  Session() : uuid(iganet::utils::uuid::create()) {}

  /// @brief Returns the UUID
  const std::string &getUUID() const { return uuid; }

  /// Returns the requested model or throws an exception
  std::shared_ptr<Model> getModel(int64_t id) {
    auto it = models.find(id);
    if (it == models.end())
      throw InvalidModelIdException();
    else
      return it->second;
  }

  /// Returns the model and removes it from the list of models
  std::shared_ptr<Model> removeModel(int64_t id) {
    auto it = models.find(id);
    if (it == models.end())
      throw InvalidModelIdException();
    else {
      auto model = it->second;
      models.erase(it);
      return model;
    }
  }

  /// @brief List of models
  std::map<int64_t, std::shared_ptr<Model>> models;
};

/// @brief Sessions structure
template <typename T> struct Sessions {
public:
  /// @brief Returns the requested session model or throws an exception
  inline std::shared_ptr<Session<T>> getSession(std::string uuid) {
    auto it = sessions.find(uuid);
    if (it == sessions.end())
      throw InvalidSessionIdException();
    else
      return it->second;
  }

  /// @brief Returns the session and removes it from the list of sessions
  inline std::shared_ptr<Session<T>> removeSession(std::string uuid) {
    auto it = sessions.find(uuid);
    if (it == sessions.end())
      throw InvalidSessionIdException();
    else {
      auto session = it->second;
      sessions.erase(it);
      return session;
    }
  }

  /// @brief Add path to model path
  inline static void addModelPath(const std::string &path) {
    models.addModelPath(path);
  }

  /// @brief Add list of paths to model path
  inline static void addModelPath(const std::vector<std::string> &path) {
    models.addModelPath(path);
  }

  /// @brief List of sessions shared between all sockets
  inline static std::map<std::string, std::shared_ptr<Session<T>>> sessions;

  /// @brief List of models
  inline static ModelManager models = ModelManager("webapps/models");
};

} // namespace webapp
} // namespace iganet

int main(int argc, char const *argv[]) {
  using PerSocketData = iganet::webapp::Sessions<iganet::real_t>;

  popl::OptionParser op("Allowed options");
  auto help_option = op.add<popl::Switch>("h", "help", "print help message");
  auto port_option =
      op.add<popl::Value<int>>("p", "port", "TCP port of the server", 9001);
  auto config_option = op.add<popl::Value<std::string>>(
      "f", "configfile", "configuration file", "");
  auto keyfile_option = op.add<popl::Value<std::string>>(
      "k", "keyfile", "key file for SSL encryption", "");
  auto certfile_option = op.add<popl::Value<std::string>>(
      "c", "certfile", "certificate file for SSL encryption", "");
  auto modelpath_option = op.add<popl::Value<std::string>>(
      "m", "modelpath", "path to model files", "");
  auto passphrase_option = op.add<popl::Value<std::string>>(
      "a", "passphrase", "passphrase for SSL encryption", "");

  op.parse(argc, argv);

  // Print auto-generated help message
  if (help_option->count() == 1) {
    std::cout << op << std::endl;
    exit(0);
  } else if (help_option->count() == 2) {
    std::cout << op.help(popl::Attribute::advanced) << std::endl;
    exit(0);
  } else if (help_option->count() > 2) {
    std::cout << op.help(popl::Attribute::expert) << std::endl;
    exit(0);
  }

  // Initialize backend
  iganet::init();

  // Load configuration from file
  nlohmann::json config;
  if (!config_option->value().empty()) {
    std::ifstream file(config_option->value());
    if (file) {
      try {
        config = nlohmann::json::parse(file);
      } catch (std::exception &e) {
        std::cerr << e.what();
        return -1;
      }
    } else {
      std::ifstream file(std::filesystem::path(__FILE__).replace_filename(
          config_option->value()));
      if (file) {
        try {
          config = nlohmann::json::parse(file);
        } catch (std::exception &e) {
          std::cerr << e.what();
          return -1;
        }
      }
    }
  }

  // Override commandline arguments
  if (!port_option->value())
    config["port"] = port_option->value();

  if (!keyfile_option->value().empty())
    config["keyFile"] = keyfile_option->value();

  if (!certfile_option->value().empty())
    config["certFile"] = certfile_option->value();

  if (!passphrase_option->value().empty())
    config["passphrase"] = passphrase_option->value();

  if (!modelpath_option->value().empty())
    config["modelPath"] = modelpath_option->value();

  // Check if key file is available
  if (config.contains("keyFile")) {
    if (!std::filesystem::exists(
            std::filesystem::path(config["keyFile"].get<std::string>()))) {
      if (std::filesystem::exists(
              std::filesystem::path(__FILE__).replace_filename(
                  config["keyFile"].get<std::string>()))) {
        config["keyFile"] = std::filesystem::path(__FILE__).replace_filename(
            config["keyFile"].get<std::string>());
      } else {
        throw std::runtime_error("Unable to open key file " +
                                 config["keyFile"].get<std::string>());
      }
    }
  }

  // Check if cert file is available
  if (config.contains("certFile")) {
    if (!std::filesystem::exists(
            std::filesystem::path(config["certFile"].get<std::string>()))) {
      if (std::filesystem::exists(
              std::filesystem::path(__FILE__).replace_filename(
                  config["certFile"].get<std::string>()))) {
        config["certFile"] = std::filesystem::path(__FILE__).replace_filename(
            config["certFile"].get<std::string>());
      } else {
        throw std::runtime_error("Unable to open cert file " +
                                 config["certFile"].get<std::string>());
      }
    }
  }

  // Add paths to model search path
  if (config.contains("modelPath"))
    PerSocketData::addModelPath(
        iganet::webapp::tokenize(config["modelPath"].get<std::string>(), ","));

  // Create WebSocket application
  try {
    uWS::SSLApp(
        {.key_file_name = (config.contains("keyFile")
                               ? config["keyFile"].get<std::string>().c_str()
                               : std::filesystem::path(__FILE__)
                                     .replace_filename("key.pem")
                                     .c_str()),
         .cert_file_name = (config.contains("certFile")
                                ? config["certFile"].get<std::string>().c_str()
                                : std::filesystem::path(__FILE__)
                                      .replace_filename("cert.pem")
                                      .c_str()),
         .passphrase = (config.contains("passphrase")
                            ? config["passphrase"].get<std::string>().c_str()
                            : "")})
        .ws<PerSocketData>(
            "/*",
            {/* Settings */
             .compression = uWS::CompressOptions(uWS::DEDICATED_COMPRESSOR_4KB |
                                                 uWS::DEDICATED_DECOMPRESSOR),
             .maxPayloadLength =
                 (config.contains("maxPayloadLength")
                      ? config["maxPayloadLength"].get<unsigned int>()
                      : 100 * 1024 * 1024),
             .idleTimeout = (config.contains("idleTimeout")
                                 ? config["idleTimeout"].get<unsigned short>()
                                 : static_cast<unsigned short>(16)),
             .maxBackpressure =
                 (config.contains("maxBackpressure")
                      ? config["maxBackpressure"].get<unsigned int>()
                      : 100 * 1024 * 1024),
             .closeOnBackpressureLimit =
                 (config.contains("closeOnBackpressureLimit")
                      ? config["closeOnBackpressureLimit"].get<bool>()
                      : false),
             .resetIdleTimeoutOnSend =
                 (config.contains("resetIdleTimeoutOnSend")
                      ? config["resetIdleTimeoutOnSend"].get<bool>()
                      : false),
             .sendPingsAutomatically =
                 (config.contains("sendPingsAutomatically")
                      ? config["sendPingsAutomatically"].get<bool>()
                      : true),
             /* Handlers */
             .upgrade = nullptr,
             .open =
                 [](auto *ws) {
#ifndef NDEBUG
                   std::clog << "Connection has been opened\n";
#endif
                 },
             .message =
                 [](auto *ws, std::string_view message, uWS::OpCode opCode) {
                   try {
                     // Tokenize request
                     auto request = nlohmann::json::parse(message);
                     auto tokens = iganet::webapp::tokenize(
                         request["request"].get<std::string>());

                     // Prepare response
                     nlohmann::json response;
                     response["request"] = request["id"];
                     response["status"] = iganet::webapp::status::success;

#ifndef NDEBUG
                     for (auto const &token : tokens)
                       std::clog << token << "/";
                     std::clog << std::endl;
#endif

                     // Dispatch request
                     if (tokens[0] == "get") {
                       //
                       // request: get/*
                       //

                       try {

                         if (tokens.size() == 1) {
                           //
                           // request: get
                           //

                           // Get list of all active sessions
                           std::vector<std::string> ids;
                           for (const auto &session :
                                ws->getUserData()->sessions)
                             ids.push_back(session.first);
                           response["data"]["ids"] = ids;
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);
                         }

                         else if (tokens.size() == 2) {
                           //
                           // request: get/<session-id>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Get list of all active models in session
                           std::vector<int64_t> ids;
                           auto models = nlohmann::json::array();
                           for (const auto &model : session->models) {
                             ids.push_back(model.first);
                             models.push_back(model.second->getModel());
                           }
                           response["data"]["ids"] = ids;
                           response["data"]["models"] = models;
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);
                         }

                         else if (tokens.size() == 3) {
                           //
                           // request: get/<session-id>/<model-instance>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Get model
                           auto model = session->getModel(stoi(tokens[2]));

                           // Serialize model to JSON
                           response["data"] = model->to_json("", "");
                           response["data"]["model"] = model->getModel();
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);
                         }

                         else if (tokens.size() == 4) {
                           //
                           // request:
                           // get/<session-id>/<model-instance>/<model-component>
                           //
                           // or
                           //
                           // request:
                           // get/<session-id>/<model-instance>/<global-attribute>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Get model
                           auto model = session->getModel(stoi(tokens[2]));

                           // Serialize model component to JSON
                           response["data"] = model->to_json(tokens[3], "");
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);
                         }

                         else if (tokens.size() == 5) {
                           //
                           // request:
                           // get/<session-id>/<model-instance>/<model-component>/<attribute>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Get model
                           auto model = session->getModel(stoi(tokens[2]));

                           // Serialize attribute of model component to JSON
                           response["data"] =
                               model->to_json(tokens[3], tokens[4]);
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);
                         }

                         else
                           throw std::runtime_error("GET");

                       } catch (...) {
                         response["status"] =
                             iganet::webapp::status::invalidGetRequest;
                         response["reason"] =
                             "Invalid GET request. Valid GET requests are "
                             "\"get\", \"get/<session-id>\", "
                             "\"get/<session-id>/<model-instance>\", and "
                             "\"get/<session-id>/<model-instance>/"
                             "<model-component>\", and "
                             "\"get/<session-id>/<model-instance>/"
                             "<model-component>/<attribute>\"";
                         ws->send(response.dump(), uWS::OpCode::TEXT, true);
                       }

                     } // GET

                     else if (tokens[0] == "put") {
                       //
                       // request: put/*
                       //

                       try {

                         if (tokens.size() == 4) {
                           //
                           // request:
                           // put/<session-id>/<model-instance>/<attribute>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Get model
                           auto model = session->getModel(stoi(tokens[2]));

                           // Update model attribute
                           response["data"] =
                               model->updateAttribute("", tokens[3], request);
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);

                           // Broadcast update of model instance
                           nlohmann::json broadcast;
                           broadcast["id"] = session->getUUID();
                           broadcast["request"] = "update/instance";
                           broadcast["data"]["id"] = stoi(tokens[2]);
                           broadcast["data"]["component"] = "";
                           broadcast["data"]["attribute"] = tokens[3];
                           ws->publish(session->getUUID(), broadcast.dump(),
                                       uWS::OpCode::TEXT);
                         }

                         else if (tokens.size() == 5) {
                           //
                           // request:
                           // put/<session-id>/<model-instance>/<model-component>/<attribute>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Get model
                           auto model = session->getModel(stoi(tokens[2]));

                           // Update model attribute
                           response["data"] = model->updateAttribute(
                               tokens[3], tokens[4], request);
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);

                           // Broadcast update of model instance
                           nlohmann::json broadcast;
                           broadcast["id"] = session->getUUID();
                           broadcast["request"] = "update/instance";
                           broadcast["data"]["id"] = stoi(tokens[2]);
                           broadcast["data"]["component"] = tokens[3];
                           broadcast["data"]["attribute"] = tokens[4];
                           ws->publish(session->getUUID(), broadcast.dump(),
                                       uWS::OpCode::TEXT);
                         }

                         else
                           throw std::runtime_error("PUT");

                       } catch (...) {
                         response["status"] =
                             iganet::webapp::status::invalidPutRequest;
                         response["reason"] =
                             "Invalid PUT request. Valid PUT requests are "
                             "\"put/<session-id>/<model-instance>/"
                             "<attribute>\", and "
                             "\"put/<session-id>/<model-instance>/"
                             "<model-component>/<attribute>\"";
                         ws->send(response.dump(), uWS::OpCode::TEXT, true);
                       }

                     } // PUT

                     else if (tokens[0] == "create") {
                       //
                       // request: create/*
                       //

                       try {

                         if (tokens.size() == 2 && tokens[1] == "session") {
                           //
                           // request: create/session
                           //

                           // Create a new session
                           auto session = std::make_shared<
                               iganet::webapp::Session<iganet::real_t>>();
                           std::string uuid = session->getUUID();
                           ws->getUserData()->sessions[uuid] = session;
                           response["data"]["id"] = uuid;
                           response["data"]["models"] =
                               ws->getUserData()->models.getModels();
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);

                           // Subscribe to new session
                           ws->subscribe(uuid);
                         }

                         else if (tokens.size() == 3) {
                           //
                           // request: create/<session-id>/<model-type>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Get new model's id
                           int64_t id =
                               (session->models.size() > 0
                                    ? session->models.crbegin()->first + 1
                                    : 0);

                           // Create a new model instance
                           session->models[id] =
                               ws->getUserData()->models.create(tokens[2],
                                                                request);
                           response["data"]["id"] = std::to_string(id);
                           response["data"]["model"] =
                               session->models[id]->getModel();
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);

                           // Broadcast creation of a new model instance
                           nlohmann::json broadcast;
                           broadcast["id"] = session->getUUID();
                           broadcast["request"] = "create/instance";
                           broadcast["data"]["id"] = id;
                           broadcast["data"]["model"] =
                               session->models[id]->getModel();
                           ws->publish(session->getUUID(), broadcast.dump(),
                                       uWS::OpCode::TEXT);
                         }

                         else
                           throw std::runtime_error("CREATE");

                       } catch (...) {
                         response["status"] =
                             iganet::webapp::status::invalidCreateRequest;
                         response["reason"] =
                             "Invalid CREATE request. Valid CREATE requests "
                             "are \"create/session\" and "
                             "\"create/<session-id>/<model-type>\"";
                         ws->send(response.dump(), uWS::OpCode::TEXT, true);
                       }

                     } // CREATE

                     else if (tokens[0] == "remove") {
                       //
                       // request: remove/*
                       //

                       try {

                         if (tokens.size() == 2) {
                           //
                           // request: remove/<session-id>
                           //

                           // Remove session
                           auto session =
                               ws->getUserData()->removeSession(tokens[1]);
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);

                           // Broadcast removal of session
                           nlohmann::json broadcast;
                           broadcast["id"] = session->getUUID();
                           broadcast["request"] = "remove/session";
                           broadcast["data"]["id"] = session->getUUID();
                           ws->publish(session->getUUID(), broadcast.dump(),
                                       uWS::OpCode::TEXT);
                         }

                         else if (tokens.size() == 3) {
                           //
                           // request: remove/<session-id>/<model-instance>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Remove model
                           auto model = session->removeModel(stoi(tokens[2]));
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);

                           // Broadcast removal of model instance
                           nlohmann::json broadcast;
                           broadcast["id"] = session->getUUID();
                           broadcast["request"] = "remove/instance";
                           broadcast["data"]["id"] = stoi(tokens[2]);
                           ws->publish(session->getUUID(), broadcast.dump(),
                                       uWS::OpCode::TEXT);
                         }

                         else
                           throw std::runtime_error("REMOVE");

                       } catch (...) {
                         response["status"] =
                             iganet::webapp::status::invalidRemoveRequest;
                         response["reason"] =
                             "Invalid remove request. Valid requests are "
                             "\"remove/<session-id>\" and "
                             "\"remove/<session-id>/<model-instance>\"";
                         ws->send(response.dump(), uWS::OpCode::TEXT, true);
                       }

                     } // REMOVE

                     else if (tokens[0] == "connect") {
                       //
                       // request: connect/*
                       //

                       try {

                         if (tokens.size() == 2) {
                           //
                           // request: connect/<session-id>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Connect to an existing session
                           response["data"]["id"] = session->getUUID();
                           response["data"]["models"] =
                               ws->getUserData()->models.getModels();
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);

                           // Subscribe to existing session
                           ws->subscribe(session->getUUID());
                         }

                         else
                           throw std::runtime_error("CONNECT");

                       } catch (...) {
                         response["status"] =
                             iganet::webapp::status::invalidConnectRequest;
                         response["reason"] =
                             "Invalid CONNECT request. Valid CONNECT requests "
                             "are \"connect/<session-id>\"";
                         ws->send(response.dump(), uWS::OpCode::TEXT, true);
                       }

                     } // CONNECT

                     else if (tokens[0] == "disconnect") {
                       //
                       // request: diconnect/*
                       //

                       try {

                         if (tokens.size() == 2) {
                           //
                           // request: diconnect/<session-id>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Disconnect from an existing session
                           response["data"]["id"] = session->getUUID();
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);

                           // Unsubscribe from existing session
                           ws->unsubscribe(session->getUUID());
                         }

                         else
                           throw std::runtime_error("DISCONNECT");

                       } catch (...) {
                         response["status"] =
                             iganet::webapp::status::invalidDisconnectRequest;
                         response["reason"] =
                             "Invalid disconnect request. Valid requests are "
                             "\"diconnect/<session-id>\"";
                         ws->send(response.dump(), uWS::OpCode::TEXT, true);
                       }

                     } // DISCONNECT

                     else if (tokens[0] == "eval") {
                       //
                       // request: eval/*
                       //

                       try {

                         if (tokens.size() == 4) {
                           //
                           // request:
                           // eval/<session-id>/<model-instance>/<model-component>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Get model
                           auto model = session->getModel(stoi(tokens[2]));

                           // Evaluate an existing model
                           if (auto m =
                                   std::dynamic_pointer_cast<iganet::ModelEval>(
                                       model))
                             response["data"] = m->eval(tokens[3], request);
                           else {
                             response["status"] =
                                 iganet::webapp::status::invalidEvalRequest;
                             response["reason"] =
                                 "Invalid eval request. Valid requests are "
                                 "\"eval/<session-id>/<model-instance>/"
                                 "<model-component>\"";
                           }
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);
                         }

                         else
                           throw std::runtime_error("EVAL");

                       } catch (...) {
                         response["status"] =
                             iganet::webapp::status::invalidEvalRequest;
                         response["reason"] =
                             "Invalid EVAL request. Valid EVAL requests are "
                             "\"eval/<session-id>/<model-instance>\"";
                         ws->send(response.dump(), uWS::OpCode::TEXT, true);
                       }

                     } // EVAL

                     else if (tokens[0] == "load") {
                       //
                       // request: load/*
                       //

                       try {

                         if (tokens.size() == 2 && tokens[1] == "session") {
                           //
                           // request: load/session
                           //

                           // Create a new session
                           auto session = std::make_shared<
                               iganet::webapp::Session<iganet::real_t>>();
                           std::string uuid = session->getUUID();
                           ws->getUserData()->sessions[uuid] = session;
                           response["data"]["id"] = uuid;
                           response["data"]["models"] =
                               ws->getUserData()->models.getModels();

                           // Create a new model instance from binary data
                           // stream
                           auto models = request["data"]["binary"];

                           for (const auto &model : models) {

                             // Get new model's id
                             int64_t id =
                                 (session->models.size() > 0
                                      ? session->models.crbegin()->first + 1
                                      : 0);

                             nlohmann::json request;
                             request["data"]["binary"] = model;

                             // Create a new model instance from binary data
                             // stream
                             session->models[id] =
                                 ws->getUserData()->models.load(request);
                           }

                           // Send response
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);

                           // Subscribe to new session
                           ws->subscribe(uuid);
                         }

                         else if (tokens.size() == 2) {
                           //
                           // request: load/<session-id>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Get new model's id
                           int64_t id =
                               (session->models.size() > 0
                                    ? session->models.crbegin()->first + 1
                                    : 0);

                           // Create a new model instance from binary data
                           // stream
                           session->models[id] =
                               ws->getUserData()->models.load(request);
                           response["data"]["id"] = std::to_string(id);
                           response["data"]["model"] =
                               session->models[id]->getModel();
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);

                           // Broadcast creation of a new model instance
                           nlohmann::json broadcast;
                           broadcast["id"] = session->getUUID();
                           broadcast["request"] = "create/instance";
                           broadcast["data"]["id"] = id;
                           broadcast["data"]["model"] =
                               session->models[id]->getModel();
                           ws->publish(session->getUUID(), broadcast.dump(),
                                       uWS::OpCode::TEXT);
                         }

                         else
                           throw std::runtime_error("LOAD");

                       } catch (...) {
                         response["status"] =
                             iganet::webapp::status::invalidLoadRequest;
                         response["reason"] =
                             "Invalid LOAD request. Valid LOAD requests are "
                             "\"load/session\" and \"load/<session-id>\"";
                         ws->send(response.dump(), uWS::OpCode::TEXT, true);
                       }

                     } // LOAD

                     else if (tokens[0] == "save") {
                       //
                       // request: save/*
                       //

                       try {

                         if (tokens.size() == 2) {
                           //
                           // request: save/<session-id>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Save all active models in session
                           auto models = nlohmann::json::array();
                           for (const auto &model : session->models) {
                             if (auto m = std::dynamic_pointer_cast<
                                     iganet::ModelSerialize>(model.second)) {
                               models.push_back(m->save());
                             }
                           }
                           response["data"] = models;
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);
                         }

                         else if (tokens.size() == 3) {
                           //
                           // request: save/<session-id>/<model-instance>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Get model
                           auto model = session->getModel(stoi(tokens[2]));

                           // Save model
                           if (auto m = std::dynamic_pointer_cast<
                                   iganet::ModelSerialize>(model))
                             response["data"] = m->save();
                           else {
                             response["status"] =
                                 iganet::webapp::status::invalidSaveRequest;
                             response["reason"] =
                                 "Invalid save request. Valid requests are "
                                 "\"save/<session-id>\" and "
                                 "\"save/<session-id>/<model-instance>\"";
                           }
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);
                         }

                         else
                           throw std::runtime_error("SAVE");

                       } catch (...) {
                         response["status"] =
                             iganet::webapp::status::invalidSaveRequest;
                         response["reason"] =
                             "Invalid save request. Valid requests are "
                             "\"save/<session-id>\" and "
                             "\"save/<session-id>/<model-instance>\"";
                         ws->send(response.dump(), uWS::OpCode::TEXT, true);
                       }

                     } // SAVE

                     else if (tokens[0] == "importxml") {
                       //
                       // request: importxml/*
                       //

                       try {

                         if (tokens.size() == 2) {
                           //
                           // request: importxml/<session-id>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Load all existing models from XML
                           for (const auto &model : session->models) {
                             if (auto m = std::dynamic_pointer_cast<
                                     iganet::ModelXML>(model.second)) {
                               m->importXML(request, "", model.first);
                             } else {
                               response["status"] =
                                   iganet::webapp::status::invalidImportRequest;
                               response["reason"] =
                                   "Invalid importrequest. Valid requests are "
                                   "\"importxml/<session-id>\", "
                                   "\"importxml/<session-id>/"
                                   "<model-instance>\" and "
                                   "\"importxml/<session-id>/<model-instance>/"
                                   "<model-component>\"";
                               ws->send(response.dump(), uWS::OpCode::TEXT,
                                        true);
                               break;
                             }
                           }
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);

                           // Broadcast update of model instances
                           std::vector<int64_t> ids;
                           for (const auto &model : session->models)
                             ids.push_back(model.first);

                           // Broadcast update of model instance
                           nlohmann::json broadcast;
                           broadcast["id"] = session->getUUID();
                           broadcast["request"] = "update/instance";
                           broadcast["data"]["ids"] = ids;
                           ws->publish(session->getUUID(), broadcast.dump(),
                                       uWS::OpCode::TEXT);
                         }

                         else if (tokens.size() == 3) {
                           //
                           // request: importxml/<session-id>/<model-instance>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Get model
                           auto model = session->getModel(stoi(tokens[2]));

                           // Import an existing model from XML
                           if (auto m =
                                   std::dynamic_pointer_cast<iganet::ModelXML>(
                                       model))
                             m->importXML(request, "", stoi(tokens[2]));
                           else {
                             response["status"] =
                                 iganet::webapp::status::invalidImportRequest;
                             response["reason"] =
                                 "Invalid import request. Valid requests are "
                                 "\"importxml/<session-id>\", "
                                 "\"importxml/<session-id>/<model-instance>\" "
                                 "and "
                                 "\"importxml/<session-id>/<model-instance>/"
                                 "<model-component>\"";
                           }
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);

                           // Broadcast update of model instance
                           nlohmann::json broadcast;
                           broadcast["id"] = session->getUUID();
                           broadcast["request"] = "update/instance";
                           broadcast["data"]["id"] = stoi(tokens[2]);
                           ws->publish(session->getUUID(), broadcast.dump(),
                                       uWS::OpCode::TEXT);
                         }

                         else if (tokens.size() == 4) {
                           //
                           // request:
                           // importxml/<session-id>/<model-instance>/<model-component>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Get model
                           auto model = session->getModel(stoi(tokens[2]));

                           // Import an existing model from XML
                           if (auto m =
                                   std::dynamic_pointer_cast<iganet::ModelXML>(
                                       model))
                             m->importXML(request, tokens[3], stoi(tokens[2]));
                           else {
                             response["status"] =
                                 iganet::webapp::status::invalidImportRequest;
                             response["reason"] =
                                 "Invalid import request. Valid requests are "
                                 "\"importxml/<session-id>\", "
                                 "\"importxml/<session-id>/<model-instance>\" "
                                 "and "
                                 "\"importxml/<session-id>/<model-instance>/"
                                 "<model-component>\"";
                           }
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);

                           // Broadcast update of model instance
                           nlohmann::json broadcast;
                           broadcast["id"] = session->getUUID();
                           broadcast["request"] = "update/instance";
                           broadcast["data"]["id"] = stoi(tokens[2]);
                           ws->publish(session->getUUID(), broadcast.dump(),
                                       uWS::OpCode::TEXT);
                         }

                         else
                           throw std::runtime_error("IMPORTXML");

                       } catch (...) {
                         response["status"] =
                             iganet::webapp::status::invalidImportRequest;
                         response["reason"] =
                             "Invalid IMPORTXML request. Valid IMPORTXML "
                             "requests are \"importxml/<session-id>\", "
                             "\"importxml/<session-id>/<model-instance>\" and "
                             "\"importxml/<session-id>/<model-instance>/"
                             "<model-component>\"";
                         ws->send(response.dump(), uWS::OpCode::TEXT, true);
                       }

                     } // IMPORTXML

                     else if (tokens[0] == "exportxml") {
                       //
                       // request: exportxml/*
                       //

                       try {

                         if (tokens.size() == 2) {
                           //
                           // request: exportxml/<session-id>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Export all existing models to XML
                           pugi::xml_document doc;
                           pugi::xml_node xml = doc.append_child("xml");

                           for (const auto &model : session->models) {
                             if (auto m = std::dynamic_pointer_cast<
                                     iganet::ModelXML>(model.second))
                               xml = m->exportXML(xml, "", model.first);
                             else
                               throw std::runtime_error("EXPORTXML");
                           }
                           std::ostringstream oss;
                           doc.save(oss);

                           response["data"]["xml"] = oss.str();
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);
                         }

                         else if (tokens.size() == 3) {
                           //
                           // request: exportxml/<session-id>/<model-instance>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Get model
                           auto model = session->getModel(stoi(tokens[2]));

                           // Export an existing model to XML
                           if (auto m =
                                   std::dynamic_pointer_cast<iganet::ModelXML>(
                                       model))
                             response["data"]["xml"] =
                                 m->exportXML("", stoi(tokens[2]));
                           else
                             throw std::runtime_error("EXPORTXML");

                           ws->send(response.dump(), uWS::OpCode::TEXT, true);
                         }

                         else if (tokens.size() == 4) {
                           //
                           // request:
                           // exportxml/<session-id>/<model-instance>/<model-component>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Get model
                           auto model = session->getModel(stoi(tokens[2]));

                           // Export an existing model to XML
                           if (auto m =
                                   std::dynamic_pointer_cast<iganet::ModelXML>(
                                       model))
                             response["data"]["xml"] =
                                 m->exportXML(tokens[3], stoi(tokens[2]));
                           else
                             throw std::runtime_error("EXPORTXML");

                           ws->send(response.dump(), uWS::OpCode::TEXT, true);
                         }

                         else
                           throw std::runtime_error("EXPORTXML");

                       } catch (...) {
                         response["status"] =
                             iganet::webapp::status::invalidExportRequest;
                         response["reason"] =
                             "Invalid EXPORTXML request. Valid EXPORTXML "
                             "requests are \"exportxml/<session-id>\", "
                             "\"exportxml/<session-id>/<model-instance>\" and "
                             "and "
                             "\"exportxml/<session-id>/<model-instance>/"
                             "<model-component>\"";
                         ws->send(response.dump(), uWS::OpCode::TEXT, true);
                       }

                     } // EXPORTXML

                     else if (tokens[0] == "refine") {
                       //
                       // request: refine/*
                       //

                       try {

                         if (tokens.size() == 3) {
                           //
                           // request: refine/<session-id>/<model-instance>
                           //

                           // Get session
                           auto session =
                               ws->getUserData()->getSession(tokens[1]);

                           // Get model
                           auto model = session->getModel(stoi(tokens[2]));

                           // Refine an existing model
                           if (auto m = std::dynamic_pointer_cast<
                                   iganet::ModelRefine>(model))
                             m->refine(request);
                           else {
                             response["status"] =
                                 iganet::webapp::status::invalidRefineRequest;
                             response["reason"] =
                                 "Invalid refine request. Valid requests are "
                                 "\"refine/<session-id>/<model-instance>\"";
                           }
                           ws->send(response.dump(), uWS::OpCode::TEXT, true);

                           // Broadcast refinement of model instance
                           nlohmann::json broadcast;
                           broadcast["id"] = session->getUUID();
                           broadcast["request"] = "refine/instance";
                           broadcast["data"]["id"] = stoi(tokens[2]);
                           ws->publish(session->getUUID(), broadcast.dump(),
                                       uWS::OpCode::TEXT);
                         }

                         else
                           throw std::runtime_error("REFINE");

                       } catch (...) {
                         response["status"] =
                             iganet::webapp::status::invalidRefineRequest;
                         response["reason"] =
                             "Invalid REFINE request. Valid REFINE requests "
                             "are \"refine/<session-id>/<model-instance>\"";
                         ws->send(response.dump(), uWS::OpCode::TEXT, true);
                       }

                     } // REFINE

                     else {
                       response["status"] =
                           iganet::webapp::status::invalidRequest;
                       response["reason"] = "Invalid request";
                       ws->send(response.dump(), uWS::OpCode::TEXT, true);
                     }
                   } catch (std::exception &e) {
                     nlohmann::json response;
                     try {
                       auto request = nlohmann::json::parse(message);
                       response["request"] = request["id"];
                       response["status"] =
                           iganet::webapp::status::invalidRequest;
                       response["reason"] = e.what();
                       ws->send(response.dump(), uWS::OpCode::TEXT, true);
                     } catch (...) {
                       response["request"] = "unknown";
                       response["status"] =
                           iganet::webapp::status::invalidRequest;
                       response["reason"] = "Invalid request";
                       ws->send(response.dump(), uWS::OpCode::TEXT, true);
                     }
                   }
                 },
             .drain =
                 [](auto *ws) {
                   /* Check ws->getBufferedAmount() here */
                 },
             .ping =
                 [](auto *ws, std::string_view) {
                   /* Not implemented yet */
                 },
             .pong =
                 [](auto *ws, std::string_view) {
                   /* Not implemented yet */
                 },
             .close =
                 [](auto *ws, int code, std::string_view message) {
    /* You may access ws->getUserData() here */
#ifndef NDEBUG
                   std::clog << "Connection has been closed\n";
#endif
                 }})
        .listen(port_option->value(),
                [&port_option](auto *listen_socket) {
                  if (listen_socket) {
                    std::clog << "Listening on port " << port_option->value()
                              << std::endl;
                  }
                })
        .run();
  } catch (std::exception &e) {
    std::cerr << e.what();
  }

  return 0;
}
