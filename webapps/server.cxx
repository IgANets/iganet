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
#include <algorithm>
#include <iganet.h>
#include <modelmanager.hpp>
#include <popl.hpp>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <thread>

namespace iganet {
namespace webapp {

/// @brief Enumerator for specifying the status
enum class status : short_t {
  success = 0,                       /*!<  request was handled successfully */
  invalidRequest = 1,                /*!<  invalid request                  */
  invalidCreateRequest = 2,          /*!<  invalid create request           */
  invalidRemoveRequest = 3,          /*!<  invalid remove request           */
  invalidConnectRequest = 4,         /*!<  invalid connect request          */
  invalidDisconnectRequest = 5,      /*!<  invalid disconnect request       */
  invalidGetRequest = 6,             /*!<  invalid get request              */
  invalidPutRequest = 7,             /*!<  invalid put request              */
  invalidEvalRequest = 8,            /*!<  invalid eval request             */
  invalidRefineRequest = 9,          /*!<  invalid refine request           */
  invalidElevateRequest = 10,        /*!<  invalid degree elevate request   */
  invalidIncreaseRequest = 11,       /*!<  invalid degree increase request  */
  invalidReparameterizeRequest = 12, /*!<  invalid reparameterize request   */
  invalidLoadRequest = 13,           /*!<  invalid load request             */
  invalidSaveRequest = 14,           /*!<  invalid save request             */
  invalidImportRequest = 15,         /*!<  invalid import request           */
  invalidExportRequest = 16,         /*!<  invalid export request           */
  invalidComputeErrorRequest = 17,   /*!<  invalid compute error request    */
  invalidAddPatchRequest = 18,       /*!<  invalid add patch request        */
  invalidRemovePatchRequest = 19     /*!<  invalid remove patch request     */  
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
  const std::string uuid_;

  /// @brief Hashed password
  const std::string hash_;

  /// @brief Creation time stamp
  std::chrono::system_clock::time_point creation_time_;

  /// @brief Access time stamp
  std::chrono::system_clock::time_point access_time_;

public:
  /// @brief Default constructor
  Session(std::string hash)
      : uuid_(iganet::utils::uuid::create()), hash_(hash),
        creation_time_(std::chrono::system_clock::now()),
        access_time_(std::chrono::system_clock::now()) {}

  /// @brief Returns the UUID
  inline const std::string &getUUID() const { return uuid_; }

  /// @brief Returns a constant reference to the list of models
  inline const auto &getModels() const { return models; }

  /// @brief Returns a non-constant reference to the list of models
  inline auto &getModels() { return models; }

  /// @brief Returns the requested model or throws an exception
  inline std::shared_ptr<Model<iganet::real_t>> getModel(int64_t id) {
    auto it = models.find(id);
    if (it == models.end())
      throw InvalidModelIdException();
    else
      return it->second;
  }

  /// @brief Returns the model and removes it from the list of models
  inline std::shared_ptr<Model<iganet::real_t>> removeModel(int64_t id) {
    auto it = models.find(id);
    if (it == models.end())
      throw InvalidModelIdException();
    else {
      auto model = it->second;
      models.erase(it);
      return model;
    }
  }

  /// @brief Returns true if the session has a non-zero hash
  inline bool hasHash() const { return (hash_ != ""); }

  /// @brief Returns true if the provided hash coincides with the session's hash
  inline bool checkHash(std::string hash) const { return (hash_ == hash); }

  /// @brief Updates the access time stamp
  inline void access() { access_time_ = std::chrono::system_clock::now(); }

  /// @brief Returns the creation time
  inline std::chrono::system_clock::time_point getCreationTime() const {
    return creation_time_;
  }

  /// @brief Returns the access time
  inline std::chrono::system_clock::time_point getAccessTime() const {
    return access_time_;
  }

  /// @brief List of models
  std::map<int64_t, std::shared_ptr<Model<iganet::real_t>>> models;
};

/// @brief Sessions structure
template <typename T> struct Sessions {
public:
  /// @brief Returns the requested session or throws an exception
  inline std::shared_ptr<Session<T>> getSession(std::string uuid) {
    auto it = sessions_.find(uuid);
    if (it == sessions_.end())
      throw InvalidSessionIdException();
    else {
      it->second->access();
      return it->second;
    }
  }

  /// @brief Returns a new session
  inline std::shared_ptr<Session<T>> createSession(std::string hash) {
    auto session =
        std::make_shared<iganet::webapp::Session<iganet::real_t>>(hash);
    sessions_[session->getUUID()] = session;
    return session;
  }

  /// @brief Returns the session and removes it from the list of sessions
  inline std::shared_ptr<Session<T>> removeSession(std::string uuid) {
    auto it = sessions_.find(uuid);
    if (it == sessions_.end())
      throw InvalidSessionIdException();
    else {
      auto session = it->second;
      sessions_.erase(it);
      return session;
    }
  }

  /// @brief Add path to model path
  inline static void addModelPath(const std::string &path) {
    models_.addModelPath(path);
  }

  /// @brief Add list of paths to model path
  inline static void addModelPath(const std::vector<std::string> &path) {
    models_.addModelPath(path);
  }

  /// @brief Returns a non-constant reference to the list of sessions
  inline static auto &getSessions() { return sessions_; }

  /// @brief Returns a non-constant reference to the list of models
  inline static auto &getModels() { return models_; }

private:
  /// @brief List of sessions shared between all sockets
  inline static std::map<std::string, std::shared_ptr<Session<T>>> sessions_;

  /// @brief List of models
  inline static ModelManager models_ =
      ModelManager(iganet::webapp::tokenize("webapps/models,models", ","));
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
  auto threads_option =
      op.add<popl::Value<int>>("t", "threads", "number of server threads", 1);

  op.parse(argc, argv);

  // Print auto-generated help message
  if (help_option->count() == 1) {
    std::cout << op << std::endl;
    return 0;
  } else if (help_option->count() == 2) {
    std::cout << op.help(popl::Attribute::advanced) << std::endl;
    return 0;
  } else if (help_option->count() > 2) {
    std::cout << op.help(popl::Attribute::expert) << std::endl;
    return 0;
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
  if (port_option->is_set())
    config["port"] = port_option->value();

  if (keyfile_option->is_set())
    config["keyFile"] = keyfile_option->value();

  if (certfile_option->is_set())
    config["certFile"] = certfile_option->value();

  if (passphrase_option->is_set())
    config["passphrase"] = passphrase_option->value();

  if (modelpath_option->is_set())
    config["modelPath"] = modelpath_option->value();

  if (threads_option->is_set())
    config["numThreads"] = threads_option->value();

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

  // Multi-threaded websocket application
  std::vector<std::thread *> threads(config.contains("numThreads")
                                         ? config["numThreads"].get<int>()
                                         : std::thread::hardware_concurrency());

  std::transform(
      threads.begin(), threads.end(), threads.begin(),
      [&config, &port_option](std::thread *) {
        return new std::thread([&config, &port_option]() {
          // Create WebSocket application
          try {
            uWS::SSLApp(
                {.key_file_name =
                     (config.contains("keyFile")
                          ? config["keyFile"].get<std::string>().c_str()
                          : std::filesystem::path(__FILE__)
                                .replace_filename("key.pem")
                                .c_str()),
                 .cert_file_name =
                     (config.contains("certFile")
                          ? config["certFile"].get<std::string>().c_str()
                          : std::filesystem::path(__FILE__)
                                .replace_filename("cert.pem")
                                .c_str()),
                 .passphrase =
                     (config.contains("passphrase")
                          ? config["passphrase"].get<std::string>().c_str()
                          : "")})
                .ws<PerSocketData>(
                    "/*",
                    {/* Settings */
                     .compression =
                         uWS::CompressOptions(uWS::DEDICATED_COMPRESSOR_4KB |
                                              uWS::DEDICATED_DECOMPRESSOR),
                     .maxPayloadLength =
                         (config.contains("maxPayloadLength")
                              ? config["maxPayloadLength"].get<unsigned int>()
                              : 100 * 1024 * 1024),
                     .idleTimeout =
                         (config.contains("idleTimeout")
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
                           ws->subscribe("broadcast");
#ifndef NDEBUG
                           std::stringstream msg;
                           msg << "[Thread " << std::this_thread::get_id()
                               << "] Connection has been opened\n";
                           std::clog << msg.str();
#endif
                         },
                     .message =
                         [](auto *ws, std::string_view message,
                            uWS::OpCode opCode) {
                           try {
                             // Tokenize request
                             auto request = nlohmann::json::parse(message);
                             auto tokens = iganet::webapp::tokenize(
                                 request["request"].get<std::string>());

                             // Prepare response
                             nlohmann::json response;
                             response["request"] = request["id"];
                             response["status"] =
                                 iganet::webapp::status::success;

#ifndef NDEBUG
                             std::stringstream msg;
                             msg << "[Thread " << std::this_thread::get_id()
                                 << "] ";
                             for (auto const &token : tokens)
                               msg << token << "/";
                             msg << std::endl;
                             std::clog << msg.str();
#endif

                             // Dispatch request
                             if (tokens[0] == "get") {
                               //
                               // request: get/*
                               //

                               try {

                                 if (tokens.size() == 2) {
                                   //
                                   // request: get/sessions OR get/<session-id>
                                   //

                                   if (tokens[1] == "sessions") {
                                     
                                     std::vector<std::string> ids;
                                     auto sessions = nlohmann::json::array();
                                     
                                     for (const auto &session :
                                            ws->getUserData()->getSessions()) {
                                       ids.push_back(session.first);

                                       auto json = nlohmann::json();
                                       auto creation_time =
                                         std::chrono::system_clock::to_time_t(
                                                                              session.second->getCreationTime());
                                       auto access_time =
                                         std::chrono::system_clock::to_time_t(
                                                                              session.second->getAccessTime());
                                       json["id"] = session.first;
                                       json["creationTime"] =
                                         std::ctime(&creation_time);
                                       json["accessTime"] =
                                         std::ctime(&access_time);
                                       json["hasHash"] =
                                         session.second->hasHash();

                                       json["nmodels"] =
                                         session.second->getModels().size();

                                       sessions.push_back(json);
                                     }
                                     
                                     response["data"]["ids"] = ids;
                                     response["data"]["sessions"] = sessions;
                                     ws->send(response.dump(), uWS::OpCode::TEXT,
                                              true);
                                   }
                                   
                                   else {
                                     // Get session
                                     auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                     // Get list of all active models in session
                                     std::vector<int64_t> ids;
                                     auto models = nlohmann::json::array();
                                     for (const auto &model :
                                            session->getModels()) {
                                       ids.push_back(model.first);
                                       models.push_back(model.second->getModel());
                                     }
                                     response["data"]["ids"] = ids;
                                     response["data"]["models"] = models;
                                     ws->send(response.dump(), uWS::OpCode::TEXT,
                                              true);
                                   }
                                 }

                                 else if (tokens.size() == 3) {
                                   //
                                   // request: get/<session-id>/<model-id>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                       session->getModel(stoi(tokens[2]));

                                   // Serialize model to JSON
                                   response["data"] =
                                       model->to_json("", "", "");
                                   response["data"]["model"] =
                                       model->getModel();
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);
                                 }

                                 else if (tokens.size() == 4) {
                                   //
                                   // request: get/<session-id>/<model-id>/<patch-id>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                       session->getModel(stoi(tokens[2]));

                                   // Serialize model patch to JSON
                                   response["data"] =
                                       model->to_json(tokens[3], "", "");
                                   response["data"]["model"] =
                                       model->getModel();
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);
                                 }

                                 else if (tokens.size() == 5) {
                                   //
                                   // request: get/<session-id>/<model-id>/<patch-id>/<component>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                       session->getModel(stoi(tokens[2]));

                                   // Serialize model component or attribute to
                                   // JSON
                                   response["data"] =
                                       model->to_json(tokens[3], tokens[4], "");
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);
                                 }

                                 else if (tokens.size() == 6) {
                                   //
                                   // request: get/<session-id>/<model-id>/<patch-id>/<component>/<attribute>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                       session->getModel(stoi(tokens[2]));

                                   // Serialize attribute of model component to
                                   // JSON
                                   response["data"] = model->to_json(
                                       tokens[3], tokens[4], tokens[5]);
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);
                                 }

                                 else
                                   throw std::runtime_error(
                                       "Invalid GET request");

                               } catch (...) {
                                 response["status"] =
                                     iganet::webapp::status::invalidGetRequest;
                                 response["reason"] =
                                     "Invalid GET request. Valid GET requests "
                                     "are "
                                     "\"get/sessions\", \"get/<session-id>\", "
                                     "\"get/<session-id>/<model-id>\", "
                                     "\"get/<session-id>/<model-id>/"
                                     "<patch-id>\", "
                                     "\"get/<session-id>/<model-id>/"
                                     "<patch-id>/<component>\", and "
                                     "\"get/<session-id>/<model-id>/"
                                     "<patch-id>/<component>/<attribute>\"";
                                 ws->send(response.dump(), uWS::OpCode::TEXT,
                                          true);
                               }

                             } // GET

                             else if (tokens[0] == "put") {
                               //
                               // request: put/*
                               //

                               try {

                                 if (tokens.size() == 5) {
                                   //
                                   // request: put/<session-id>/<model-id>/<patch-id>/<attribute>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                       session->getModel(stoi(tokens[2]));

                                   // Update model attribute
                                   response["data"] = model->updateAttribute(
                                       tokens[3], "", tokens[4], request);
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);

                                   // Broadcast model update
                                   nlohmann::json broadcast;
                                   broadcast["id"] = session->getUUID();
                                   broadcast["request"] = "update/model";
                                   broadcast["data"]["id"] = stoi(tokens[2]);
                                   broadcast["data"]["patch"] = tokens[3];
                                   broadcast["data"]["component"] = "";
                                   broadcast["data"]["attribute"] = tokens[4];
                                   ws->publish(session->getUUID(),
                                               broadcast.dump(),
                                               uWS::OpCode::TEXT);
                                 }

                                 else if (tokens.size() == 6) {
                                   //
                                   // request: put/<session-id>/<model-id>/<patch-id>/<component>/<attribute>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                       session->getModel(stoi(tokens[2]));

                                   // Update model attribute
                                   response["data"] = model->updateAttribute(
                                       tokens[3], tokens[4], tokens[5],
                                       request);
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);

                                   // Broadcast model update
                                   nlohmann::json broadcast;
                                   broadcast["id"] = session->getUUID();
                                   broadcast["request"] = "update/model";
                                   broadcast["data"]["id"] = stoi(tokens[2]);
                                   broadcast["data"]["patch"] = tokens[3];
                                   broadcast["data"]["component"] = tokens[4];
                                   broadcast["data"]["attribute"] = tokens[5];
                                   ws->publish(session->getUUID(),
                                               broadcast.dump(),
                                               uWS::OpCode::TEXT);
                                 }

                                 else
                                   throw std::runtime_error(
                                       "Invalid PUT request");

                               } catch (...) {
                                 response["status"] =
                                     iganet::webapp::status::invalidPutRequest;
                                 response["reason"] =
                                     "Invalid PUT request. Valid PUT requests "
                                     "are "
                                     "\"put/<session-id>/<model-id>/"
                                     "<patch-id>/<attribute>\", and "
                                     "\"put/<session-id>/<model-id>/"
                                     "<patch-id>/<component>/<attribute>\"";
                                 ws->send(response.dump(), uWS::OpCode::TEXT,
                                          true);
                               }

                             } // PUT

                             else if (tokens[0] == "create") {
                               //
                               // request: create/*
                               //

                               try {

                                 if (tokens.size() == 2 &&
                                     tokens[1] == "session") {
                                   //
                                   // request: create/session
                                   //

                                   // Get password hash
                                   std::string hash("");
                                   if (request.contains("data"))
                                     if (request["data"].contains("hash"))
                                       hash = request["data"]["hash"]
                                                  .get<std::string>();

                                   // Create a new session
                                   auto session =
                                       ws->getUserData()->createSession(hash);
                                   std::string uuid = session->getUUID();

                                   response["data"]["id"] = uuid;
                                   response["data"]["models"] =
                                       ws->getUserData()
                                           ->getModels()
                                           .getModels();
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);

                                   // Subscribe to new session
                                   ws->subscribe(uuid);

                                   // Broadcast creation of a new session
                                   nlohmann::json broadcast;
                                   broadcast["id"] = uuid;
                                   broadcast["request"] = "create/session";
                                   broadcast["data"]["id"] = uuid;
                                   ws->publish("broadcast", broadcast.dump(),
                                               uWS::OpCode::TEXT);
                                 }

                                 else if (tokens.size() == 3) {
                                   //
                                   // request: create/<session-id>/<model-type>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get new model's id
                                   int64_t id = (session->getModels().size() > 0
                                                     ? session->getModels()
                                                               .crbegin()
                                                               ->first +
                                                           1
                                                     : 0);

                                   // Create a new model
                                   session->models[id] =
                                       ws->getUserData()->getModels().create(
                                           tokens[2], request);
                                   response["data"]["id"] = std::to_string(id);
                                   response["data"]["model"] =
                                       session->models[id]->getModel();
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);

                                   // Broadcast model creation
                                   nlohmann::json broadcast;
                                   broadcast["id"] = session->getUUID();
                                   broadcast["request"] = "create/model";
                                   broadcast["data"]["id"] = id;
                                   broadcast["data"]["model"] =
                                       session->models[id]->getModel();
                                   ws->publish(session->getUUID(),
                                               broadcast.dump(),
                                               uWS::OpCode::TEXT);
                                 }

                                 else
                                   throw std::runtime_error(
                                       "Invalid CREATE request");

                               } catch (...) {
                                 response["status"] = iganet::webapp::status::
                                     invalidCreateRequest;
                                 response["reason"] =
                                     "Invalid CREATE request. Valid CREATE "
                                     "requests "
                                     "are \"create/session\" and "
                                     "\"create/<session-id>/<model-type>\"";
                                 ws->send(response.dump(), uWS::OpCode::TEXT,
                                          true);
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
                                       ws->getUserData()->removeSession(
                                           tokens[1]);
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);

                                   // Broadcast removal of session
                                   nlohmann::json broadcast;
                                   broadcast["id"] = session->getUUID();
                                   broadcast["request"] = "remove/session";
                                   broadcast["data"]["id"] = session->getUUID();
                                   ws->publish(session->getUUID(),
                                               broadcast.dump(),
                                               uWS::OpCode::TEXT);
                                 }

                                 else if (tokens.size() == 3) {
                                   //
                                   // request:
                                   // remove/<session-id>/<model-id>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Remove model
                                   auto model =
                                       session->removeModel(stoi(tokens[2]));
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);

                                   // Broadcast model removal
                                   nlohmann::json broadcast;
                                   broadcast["id"] = session->getUUID();
                                   broadcast["request"] = "remove/model";
                                   broadcast["data"]["id"] = stoi(tokens[2]);
                                   ws->publish(session->getUUID(),
                                               broadcast.dump(),
                                               uWS::OpCode::TEXT);
                                 }

                                 else if (tokens.size() == 4) {
                                   //
                                   // request:
                                   // remove/<session-id>/<model-id>/<patch-id>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                     session->getModel(stoi(tokens[2]));

                                   // Remove an existing model
                                   if (auto m = std::dynamic_pointer_cast<
                                           iganet::ModelRemovePatch>(model))
                                     m->removePatch(request);
                                   else {
                                     response["status"] = iganet::webapp::
                                         status::invalidRemovePatchRequest;
                                     response["reason"] =
                                         "Invalid REMOVE request. Valid REMOVE "
                                         "requests are "
                                         "are "
                                         "\"remove/<session-id>\", "
                                         "\"remove/<session-id>/<model-id>\" and "
                                         "\"remove/<session-id>/<model-id>/<patch-id>\"";
                                   }
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);                                                                      

                                   // Broadcast patch removal
                                   nlohmann::json broadcast;
                                   broadcast["id"] = session->getUUID();
                                   broadcast["request"] = "remove/patch";
                                   broadcast["data"]["id"] = stoi(tokens[2]);
                                   broadcast["data"]["patch"] = tokens[3];
                                   ws->publish(session->getUUID(),
                                               broadcast.dump(),
                                               uWS::OpCode::TEXT);
                                 }

                                 else
                                   throw std::runtime_error(
                                       "Invalid REMOVE request");

                               } catch (...) {
                                 response["status"] = iganet::webapp::status::
                                     invalidRemoveRequest;
                                 response["reason"] =
                                     "Invalid REMOVE request. Valid REMOVE "
                                     "requests "
                                     "are "
                                     "\"remove/<session-id>\", "
                                     "\"remove/<session-id>/<model-id>\" and "
                                     "\"remove/<session-id>/<model-id>/<patch-id>\"";
                                 ws->send(response.dump(), uWS::OpCode::TEXT,
                                          true);
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

                                   // Get password hash
                                   std::string hash("");
                                   if (request.contains("data"))
                                     if (request["data"].contains("hash"))
                                       hash = request["data"]["hash"]
                                                  .get<std::string>();

                                   if (!session->checkHash(hash))
                                     throw std::runtime_error(
                                         "Invalid CONNECT request. Invalid "
                                         "ession password.");

                                   // Connect to an existing session
                                   response["data"]["id"] = session->getUUID();
                                   response["data"]["models"] =
                                       ws->getUserData()
                                           ->getModels()
                                           .getModels();
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);

                                   // Subscribe to existing session
                                   ws->subscribe(session->getUUID());
                                 }

                                 else
                                   throw std::runtime_error(
                                       "Invalid CONNECT request");

                               } catch (...) {
                                 response["status"] = iganet::webapp::status::
                                     invalidConnectRequest;
                                 response["reason"] =
                                     "Invalid CONNECT request. Valid CONNECT "
                                     "requests "
                                     "are \"connect/<session-id>\"";
                                 ws->send(response.dump(), uWS::OpCode::TEXT,
                                          true);
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
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);

                                   // Unsubscribe from existing session
                                   ws->unsubscribe(session->getUUID());
                                 }

                                 else
                                   throw std::runtime_error(
                                       "Invalid DISCONNECT request");

                               } catch (...) {
                                 response["status"] = iganet::webapp::status::
                                     invalidDisconnectRequest;
                                 response["reason"] =
                                     "Invalid DISCONNECT request. Valid "
                                     "DISCONNECT "
                                     "requests are "
                                     "\"diconnect/<session-id>\"";
                                 ws->send(response.dump(), uWS::OpCode::TEXT,
                                          true);
                               }

                             } // DISCONNECT

                             else if (tokens[0] == "eval") {
                               //
                               // request: eval/*
                               //

                               try {

                                 if (tokens.size() == 5) {
                                   //
                                   // request: eval/<session-id>/<model-id>/<patch-id>/<component>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                       session->getModel(stoi(tokens[2]));

                                   // Evaluate an existing model
                                   if (auto m = std::dynamic_pointer_cast<
                                           iganet::ModelEval>(model))
                                     response["data"] =
                                         m->eval(tokens[3], tokens[4], request);
                                   else {
                                     response["status"] = iganet::webapp::
                                         status::invalidEvalRequest;
                                     response["reason"] =
                                         "Invalid EVAL request. Valid EVAL "
                                         "requests "
                                         "are "
                                         "\"eval/<session-id>/<model-id>/"
                                         "<component>\"";
                                   }
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);
                                 }

                                 else
                                   throw std::runtime_error(
                                       "Invalid EVAL request");

                               } catch (...) {
                                 response["status"] =
                                     iganet::webapp::status::invalidEvalRequest;
                                 response["reason"] =
                                     "Invalid EVAL request. Valid EVAL "
                                     "requests are "
                                     "\"eval/<session-id>/<model-id>/"
                                     "<patch-id>/<component>\"";
                                 ws->send(response.dump(), uWS::OpCode::TEXT,
                                          true);
                               }

                             } // EVAL

                             else if (tokens[0] == "load") {
                               //
                               // request: load/*
                               //

                               try {

                                 if (tokens.size() == 2) {
                                   //
                                   // request: load/<session-id>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get binary data
                                   auto instances =
                                       request["data"]["instances"];

                                   // Create vector if ids and array of models
                                   std::vector<int64_t> ids;
                                   auto models = nlohmann::json::array();

                                   // Loop over all instances
                                   for (const auto &instance : instances) {

                                     // Get new model's id
                                     int64_t id =
                                         (session->getModels().size() > 0
                                              ? session->getModels()
                                                        .crbegin()
                                                        ->first +
                                                    1
                                              : 0);

                                     nlohmann::json request;
                                     request["data"]["binary"] = instance;

                                     // Create a new model from binary
                                     // data stream
                                     session->models[id] =
                                         ws->getUserData()->getModels().load(
                                             request);
                                     ids.push_back(id);
                                     models.push_back(
                                         session->models[id]->getModel());

                                     // Broadcast creation of a new model
                                     nlohmann::json broadcast;
                                     broadcast["id"] = session->getUUID();
                                     broadcast["request"] = "create/model";
                                     broadcast["data"]["id"] = id;
                                     broadcast["data"]["model"] =
                                         session->models[id]->getModel();
                                     ws->publish(session->getUUID(),
                                                 broadcast.dump(),
                                                 uWS::OpCode::TEXT);
                                   }

                                   response["data"]["ids"] = ids;
                                   response["data"]["models"] = models;
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);
                                 }

                                 else
                                   throw std::runtime_error(
                                       "Invalid LOAD request");

                               } catch (...) {
                                 response["status"] =
                                     iganet::webapp::status::invalidLoadRequest;
                                 response["reason"] = "Invalid LOAD request. "
                                                      "Valid LOAD requests are "
                                                      "\"load/session\" and "
                                                      "\"load/<session-id>\"";
                                 ws->send(response.dump(), uWS::OpCode::TEXT,
                                          true);
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
                                   for (const auto &model :
                                        session->getModels()) {
                                     if (auto m = std::dynamic_pointer_cast<
                                             iganet::ModelSerialize>(
                                             model.second)) {
                                       models.push_back(m->save());
                                     }
                                   }
                                   response["data"] = models;
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);
                                 }

                                 else if (tokens.size() == 3) {
                                   //
                                   // request: save/<session-id>/<model-id>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                       session->getModel(stoi(tokens[2]));

                                   // Save model
                                   if (auto m = std::dynamic_pointer_cast<
                                           iganet::ModelSerialize>(model))
                                     response["data"] = m->save();
                                   else {
                                     response["status"] = iganet::webapp::
                                         status::invalidSaveRequest;
                                     response["reason"] =
                                         "Invalid SAVE request. Valid SAVE "
                                         "requests "
                                         "are "
                                         "\"save/<session-id>\" and "
                                         "\"save/<session-id>/"
                                         "<model-id>\"";
                                   }
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);
                                 }

                                 else
                                   throw std::runtime_error(
                                       "Invalid SAVE request");

                               } catch (...) {
                                 response["status"] =
                                     iganet::webapp::status::invalidSaveRequest;
                                 response["reason"] =
                                     "Invalid SAVE request. Valid SAVE "
                                     "requests are "
                                     "\"save/<session-id>\" and "
                                     "\"save/<session-id>/<model-id>\"";
                                 ws->send(response.dump(), uWS::OpCode::TEXT,
                                          true);
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
                                   for (const auto &model :
                                        session->getModels()) {
                                     if (auto m = std::dynamic_pointer_cast<
                                             iganet::ModelXML>(model.second)) {
                                       m->importXML("", "", request, model.first);
                                     } else {
                                       response["status"] = iganet::webapp::
                                           status::invalidImportRequest;
                                       response["reason"] =
                                           "Invalid IMPORTXML request. Valid "
                                           "IMPORTXML "
                                           "requests are "
                                           "\"importxml/<session-id>\", "
                                           "\"importxml/<session-id>/"
                                           "<model-id>\", "
                                           "\"importxml/<session-id>/"
                                           "<model-id>/"
                                           "<component>\", "
                                           "\"importxml/<session-id>/"
                                           "<model-id>/"
                                           "<patch-id>\", and "
                                           "\"importxml/<session-id>/"
                                           "<model-id>/"
                                           "<patch-id>/<component>\"";
                                       ws->send(response.dump(),
                                                uWS::OpCode::TEXT, true);
                                       break;
                                     }
                                   }
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);

                                   // Broadcast model updates
                                   std::vector<int64_t> ids;
                                   for (const auto &model :
                                        session->getModels())
                                     ids.push_back(model.first);

                                   // Broadcast model update
                                   nlohmann::json broadcast;
                                   broadcast["id"] = session->getUUID();
                                   broadcast["request"] = "update/model";
                                   broadcast["data"]["ids"] = ids;
                                   ws->publish(session->getUUID(),
                                               broadcast.dump(),
                                               uWS::OpCode::TEXT);
                                 }

                                 else if (tokens.size() == 3) {
                                   //
                                   // request: importxml/<session-id>/<model-id>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                       session->getModel(stoi(tokens[2]));

                                   // Import an existing model from XML
                                   if (auto m = std::dynamic_pointer_cast<
                                           iganet::ModelXML>(model))
                                     m->importXML("", "", request, -1);
                                   else {
                                     response["status"] = iganet::webapp::
                                       status::invalidImportRequest;
                                     response["reason"] =
                                           "Invalid IMPORTXML request. Valid "
                                           "IMPORTXML "
                                           "requests are "
                                           "\"importxml/<session-id>\", "
                                           "\"importxml/<session-id>/"
                                           "<model-id>\", "
                                           "\"importxml/<session-id>/"
                                           "<model-id>/"
                                           "<component>\", "
                                           "\"importxml/<session-id>/"
                                           "<model-id>/"
                                           "<patch-id>\", and "
                                           "\"importxml/<session-id>/"
                                           "<model-id>/"
                                           "<patch-id>/<component>\"";
                                   }
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);

                                   // Broadcast model update
                                   nlohmann::json broadcast;
                                   broadcast["id"] = session->getUUID();
                                   broadcast["request"] = "update/model";
                                   broadcast["data"]["id"] = stoi(tokens[2]);
                                   ws->publish(session->getUUID(),
                                               broadcast.dump(),
                                               uWS::OpCode::TEXT);
                                 }

                                 else if (tokens.size() == 4) {
                                   //
                                   // request: importxml/<session-id>/<model-id>/<component> OR
                                   //          importxml/<session-id>/<model-id>/<patch-id>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                       session->getModel(stoi(tokens[2]));

                                   // Import an existing model component from
                                   // XML
                                   if (auto m = std::dynamic_pointer_cast<
                                       iganet::ModelXML>(model)) {
                                     try {
                                       (void)stoi(tokens[3]);
                                       m->importXML(tokens[3], "", request, -1);  
                                     } catch (...) {
                                       m->importXML("", tokens[3], request, -1);  
                                     }
                                   }
                                     
                                   else {
                                     response["status"] = iganet::webapp::
                                       status::invalidImportRequest;
                                     response["reason"] =
                                           "Invalid IMPORTXML request. Valid "
                                           "IMPORTXML "
                                           "requests are "
                                           "\"importxml/<session-id>\", "
                                           "\"importxml/<session-id>/"
                                           "<model-id>\", "
                                           "\"importxml/<session-id>/"
                                           "<model-id>/"
                                           "<component>\", "
                                           "\"importxml/<session-id>/"
                                           "<model-id>/"
                                           "<patch-id>\", and "
                                           "\"importxml/<session-id>/"
                                           "<model-id>/"
                                           "<patch-id>/<component>\"";
                                   }
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);

                                   // Broadcast model update
                                   nlohmann::json broadcast;
                                   broadcast["id"] = session->getUUID();
                                   broadcast["request"] = "update/model";
                                   broadcast["data"]["id"] = stoi(tokens[2]);
                                   ws->publish(session->getUUID(),
                                               broadcast.dump(),
                                               uWS::OpCode::TEXT);
                                 }

                                 else if (tokens.size() == 5) {
                                   //
                                   // request: importxml/<session-id>/<model-id>/<patch-id>/<component>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                       session->getModel(stoi(tokens[2]));

                                   // Import an existing model component from
                                   // XML
                                   if (auto m = std::dynamic_pointer_cast<
                                       iganet::ModelXML>(model))
                                     m->importXML(tokens[3], tokens[4], request, -1);  
                                     
                                   else {
                                     response["status"] = iganet::webapp::
                                       status::invalidImportRequest;
                                     response["reason"] =
                                           "Invalid IMPORTXML request. Valid "
                                           "IMPORTXML "
                                           "requests are "
                                           "\"importxml/<session-id>\", "
                                           "\"importxml/<session-id>/"
                                           "<model-id>\", "
                                           "\"importxml/<session-id>/"
                                           "<model-id>/"
                                           "<component>\", "
                                           "\"importxml/<session-id>/"
                                           "<model-id>/"
                                           "<patch-id>\", and "
                                           "\"importxml/<session-id>/"
                                           "<model-id>/"
                                           "<patch-id>/<component>\"";
                                   }
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);

                                   // Broadcast model update
                                   nlohmann::json broadcast;
                                   broadcast["id"] = session->getUUID();
                                   broadcast["request"] = "update/model";
                                   broadcast["data"]["id"] = stoi(tokens[2]);
                                   ws->publish(session->getUUID(),
                                               broadcast.dump(),
                                               uWS::OpCode::TEXT);
                                 }

                                 else
                                   throw std::runtime_error(
                                       "Invalid IMPORTXML request");

                               } catch (...) {
                                 response["status"] = iganet::webapp::status::
                                     invalidImportRequest;
                                 response["reason"] =
                                     "Invalid IMPORTXML request. Valid "
                                     "IMPORTXML "
                                     "IMPORTXML "
                                     "requests are \"importxml/<session-id>\", "
                                     "\"importxml/<session-id>/"
                                     "<model-id>\" and "
                                     "\"importxml/<session-id>/"
                                     "<model-id>/"
                                     "<component>\"";
                                 ws->send(response.dump(), uWS::OpCode::TEXT,
                                          true);
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

                                   for (const auto &model :
                                        session->getModels()) {
                                     if (auto m = std::dynamic_pointer_cast<
                                             iganet::ModelXML>(model.second))
                                       xml = m->exportXML("", "", xml, model.first);
                                     else
                                       throw std::runtime_error(
                                           "Invalid EXPORTXML request");
                                   }
                                   std::ostringstream oss;
                                   doc.save(oss);

                                   response["data"]["xml"] = oss.str();
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);
                                 }

                                 else if (tokens.size() == 3) {
                                   //
                                   // request: exportxml/<session-id>/<model-id>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                       session->getModel(stoi(tokens[2]));

                                   // Export an existing model to XML
                                   if (auto m = std::dynamic_pointer_cast<
                                           iganet::ModelXML>(model))
                                     response["data"]["xml"] =
                                       m->exportXML("", "", stoi(tokens[2]));
                                   else
                                     throw std::runtime_error(
                                         "Invalid EXPORTXML request");

                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);
                                 }

                                 else if (tokens.size() == 4) {
                                   //
                                   // request: exportxml/<session-id>/<model-id>/<component> OR
                                   //          exportxml/<session-id>/<model-id>/<patch-id>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                       session->getModel(stoi(tokens[2]));

                                   // Export an existing model to XML
                                   if (auto m = std::dynamic_pointer_cast<
                                       iganet::ModelXML>(model))
                                     try {
                                       (void)stoi(tokens[3]);
                                       response["data"]["xml"] = m->exportXML(tokens[3], "", stoi(tokens[2]));
                                     } catch (...) {
                                       response["data"]["xml"] = m->exportXML("", tokens[3], stoi(tokens[2]));
                                     }
                                   else
                                     throw std::runtime_error(
                                         "Invalid EXPORTXML request");

                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);
                                 }

                                 else if (tokens.size() == 5) {
                                   //
                                   // request: exportxml/<session-id>/<model-id>/<patch-id>/<component>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                       session->getModel(stoi(tokens[2]));

                                   // Export an existing model to XML
                                   if (auto m = std::dynamic_pointer_cast<
                                           iganet::ModelXML>(model))
                                     response["data"]["xml"] = m->exportXML(
                                                                            tokens[3], tokens[4], stoi(tokens[2]));
                                   else
                                     throw std::runtime_error(
                                         "Invalid EXPORTXML request");

                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);
                                 }

                                 else
                                   throw std::runtime_error(
                                       "Invalid EXPORTXML request");

                               } catch (...) {
                                 response["status"] = iganet::webapp::status::
                                     invalidExportRequest;
                                 response["reason"] =
                                     "Invalid EXPORTXML request. Valid "
                                     "EXPORTXML "
                                     "EXPORTXML "
                                     "requests are \"exportxml/<session-id>\", "
                                     "\"exportxml/<session-id>/"
                                     "<model-id>\" and "
                                     "and "
                                     "\"exportxml/<session-id>/"
                                     "<model-id>/"
                                     "<component>\"";
                                 ws->send(response.dump(), uWS::OpCode::TEXT,
                                          true);
                               }

                             } // EXPORTXML

                             else if (tokens[0] == "refine") {
                               //
                               // request: refine/*
                               //

                               try {

                                 if (tokens.size() == 3) {
                                   //
                                   // request: refine/<session-id>/<model-id>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                       session->getModel(stoi(tokens[2]));

                                   // Refine an existing model
                                   if (auto m = std::dynamic_pointer_cast<
                                           iganet::ModelRefine>(model))
                                     m->refine(request);
                                   else {
                                     response["status"] = iganet::webapp::
                                         status::invalidRefineRequest;
                                     response["reason"] =
                                         "Invalid REFINE request. Valid REFINE "
                                         "requests are "
                                         "\"refine/<session-id>/"
                                         "<model-id>\"";
                                   }
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);

                                   // Broadcast model refinement
                                   nlohmann::json broadcast;
                                   broadcast["id"] = session->getUUID();
                                   broadcast["request"] = "refine/model";
                                   broadcast["data"]["id"] = stoi(tokens[2]);
                                   ws->publish(session->getUUID(),
                                               broadcast.dump(),
                                               uWS::OpCode::TEXT);
                                 }

                                 else
                                   throw std::runtime_error(
                                       "Invalid REFINE request");

                               } catch (...) {
                                 response["status"] = iganet::webapp::status::
                                     invalidRefineRequest;
                                 response["reason"] =
                                     "Invalid REFINE request. Valid REFINE "
                                     "requests "
                                     "are "
                                     "\"refine/<session-id>/<model-id>\"";
                                 ws->send(response.dump(), uWS::OpCode::TEXT,
                                          true);
                               }

                             } // REFINE

                             else if (tokens[0] == "elevate") {
                               //
                               // request: elevate/*
                               //

                               try {

                                 if (tokens.size() == 3) {
                                   //
                                   // request: elevate/<session-id>/<model-id>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                       session->getModel(stoi(tokens[2]));

                                   // Degree elevate an existing model
                                   if (auto m = std::dynamic_pointer_cast<
                                           iganet::ModelElevate>(model))
                                     m->elevate(request);
                                   else {
                                     response["status"] = iganet::webapp::
                                         status::invalidElevateRequest;
                                     response["reason"] =
                                         "Invalid ELEVATE request. Valid "
                                         "ELEVATE "
                                         "requests are "
                                         "\"elevate/<session-id>/"
                                         "<model-id>\"";
                                   }
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);

                                   // Broadcast model degree elevation
                                   nlohmann::json broadcast;
                                   broadcast["id"] = session->getUUID();
                                   broadcast["request"] = "elevate/model";
                                   broadcast["data"]["id"] = stoi(tokens[2]);
                                   ws->publish(session->getUUID(),
                                               broadcast.dump(),
                                               uWS::OpCode::TEXT);
                                 }

                                 else
                                   throw std::runtime_error(
                                       "Invalid ELEVATE request");

                               } catch (...) {
                                 response["status"] = iganet::webapp::status::
                                     invalidElevateRequest;
                                 response["reason"] =
                                     "Invalid ELEVATE request. Valid ELEVATE "
                                     "requests "
                                     "are "
                                     "\"elevate/<session-id>/"
                                     "<model-id>\"";
                                 ws->send(response.dump(), uWS::OpCode::TEXT,
                                          true);
                               }

                             } // ELEVATE

                             else if (tokens[0] == "increase") {
                               //
                               // request: increase/*
                               //

                               try {

                                 if (tokens.size() == 3) {
                                   //
                                   // request: increase/<session-id>/<model-id>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                       session->getModel(stoi(tokens[2]));

                                   // Degree increase an existing model
                                   if (auto m = std::dynamic_pointer_cast<
                                           iganet::ModelIncrease>(model))
                                     m->increase(request);
                                   else {
                                     response["status"] = iganet::webapp::
                                         status::invalidIncreaseRequest;
                                     response["reason"] =
                                         "Invalid INCREASE request. Valid "
                                         "INCREASE "
                                         "requests are "
                                         "\"increase/<session-id>/"
                                         "<model-id>\"";
                                   }
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);

                                   // Broadcast model degree increase
                                   nlohmann::json broadcast;
                                   broadcast["id"] = session->getUUID();
                                   broadcast["request"] = "increase/model";
                                   broadcast["data"]["id"] = stoi(tokens[2]);
                                   ws->publish(session->getUUID(),
                                               broadcast.dump(),
                                               uWS::OpCode::TEXT);
                                 }

                                 else
                                   throw std::runtime_error(
                                       "Invalid INCREASE request");

                               } catch (...) {
                                 response["status"] = iganet::webapp::status::
                                     invalidElevateRequest;
                                 response["reason"] =
                                     "Invalid INCREASE request. Valid INCREASE "
                                     "requests "
                                     "are "
                                     "\"increase/<session-id>/"
                                     "<model-id>\"";
                                 ws->send(response.dump(), uWS::OpCode::TEXT,
                                          true);
                               }

                             } // INCREASE

                             else if (tokens[0] == "reparameterize") {
                               //
                               // request: reparameterize/*
                               //

                               try {

                                 if (tokens.size() == 3) {
                                   //
                                   // request: reparameterize/<session-id>/<model-id>
                                   //

                                   // Get session
                                   auto session =
                                       ws->getUserData()->getSession(tokens[1]);

                                   // Get model
                                   auto model =
                                       session->getModel(stoi(tokens[2]));

                                   // Reparameterize an existing model
                                   if (auto m = std::dynamic_pointer_cast<
                                           iganet::ModelReparameterize>(model))
                                     m->reparameterize(request);
                                   else {
                                     response["status"] = iganet::webapp::
                                         status::invalidReparameterizeRequest;
                                     response["reason"] =
                                         "Invalid REPARAMETERIZE request. "
                                         "Valid REPARAMETERIZE "
                                         "requests are "
                                         "\"reparameterize/<session-id>/"
                                         "<model-id>\"";
                                   }
                                   ws->send(response.dump(), uWS::OpCode::TEXT,
                                            true);

                                   // Broadcast model reparameterization
                                   nlohmann::json broadcast;
                                   broadcast["id"] = session->getUUID();
                                   broadcast["request"] =
                                       "reparameterize/model";
                                   broadcast["data"]["id"] = stoi(tokens[2]);
                                   ws->publish(session->getUUID(),
                                               broadcast.dump(),
                                               uWS::OpCode::TEXT);
                                 }

                                 else
                                   throw std::runtime_error(
                                       "Invalid REPARAMETERIZE request");

                               } catch (...) {
                                 response["status"] = iganet::webapp::status::
                                     invalidReparameterizeRequest;
                                 response["reason"] =
                                     "Invalid REPARAMETERIZE request. Valid "
                                     "REPARAMETERIZE "
                                     "requests "
                                     "are "
                                     "\"reparameterize/<session-id>/"
                                     "<model-id>\"";
                                 ws->send(response.dump(), uWS::OpCode::TEXT,
                                          true);
                               }

                             } // REPARAMETERIZE                             

                             else {
                               response["status"] =
                                   iganet::webapp::status::invalidRequest;
                               response["reason"] = "Invalid request";
                               ws->send(response.dump(), uWS::OpCode::TEXT,
                                        true);
                             }
                           } catch (std::exception &e) {
                             nlohmann::json response;
                             try {
                               auto request = nlohmann::json::parse(message);
                               response["request"] = request["id"];
                               response["status"] =
                                   iganet::webapp::status::invalidRequest;
                               response["reason"] = e.what();
                               ws->send(response.dump(), uWS::OpCode::TEXT,
                                        true);
                             } catch (...) {
                               response["request"] = "unknown";
                               response["status"] =
                                   iganet::webapp::status::invalidRequest;
                               response["reason"] = "Invalid request";
                               ws->send(response.dump(), uWS::OpCode::TEXT,
                                        true);
                             }
                           }
                         },
                     .drain =
                         [](auto *ws) {
                           /* Check ws->getBufferedAmount() here */
                         },
                     .ping =
                         [](auto *ws, std::string_view) {
                           /* You don't need to handle this one, we automatically respond to pings as per standard */
                         },
                     .pong =
                         [](auto *ws, std::string_view) {
                           /* You don't need to handle this one, we automatically respond to pings as per standard */
                         },
                     .close =
                         [](auto *ws, int code, std::string_view message) {
                           /* You may access ws->getUserData() here */
                           ws->unsubscribe("broadcast");
#ifndef NDEBUG
                           std::stringstream msg;
                           msg << "[Thread " << std::this_thread::get_id()
                               << "] Connection has been closed\n";
                           std::clog << msg.str();
#endif
                         }})
                .listen(port_option->value(),
                        [&port_option](auto *listen_socket) {
                          if (listen_socket) {
                            std::stringstream msg;
                            msg << "[Thread " << std::this_thread::get_id()
                                << "] Listening on port "
                                << port_option->value() << std::endl;
                            std::clog << msg.str();
                          } else {
                            std::stringstream msg;
                            msg << "[Thread " << std::this_thread::get_id()
                                << "] Failed to listen on port "
                                << port_option->value() << std::endl;
                            std::clog << msg.str();
                          }
                        })
                .run();
          } catch (std::exception &e) {
            std::cerr << e.what();
          }
        });
      });

  std::for_each(threads.begin(), threads.end(),
                [](std::thread *t) { t->join(); });

  return 0;
}
