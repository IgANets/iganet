/**
   @file webapps/models/BSplineModel.hpp

   @brief BSpline model

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <model.hpp>

namespace iganet {

namespace webapp {

/// @brief Enumerator for specifying the degree of B-splines
enum class degree {
  constant = 0,  /*!<  constant B-Spline basis functions */
  linear = 1,    /*!<    linear B-Spline basis functions */
  quadratic = 2, /*!< quadratic B-Spline basis functions */
  cubic = 3,     /*!<     cubic B-Spline basis functions */
  quartic = 4,   /*!<   quartic B-Spline basis functions */
  quintic = 5    /*!<   quintic B-Spline basis functions */
};

/// @brief B-spline model
template <class BSpline_t>
class BSplineModel : public Model,
                     public ModelEval,
                     public ModelReparameterize,
                     public ModelRefine,
                     public ModelSerialize,
                     public ModelXML,
                     public BSpline_t {
private:
  /// @brief Global offset vector
  torch::Tensor offset_;

  /// @brief Global rotation vector
  torch::Tensor rotation_;

  /// @brief "fake" solution vector
  BSpline_t solution_;

public:
  /// @brief Default constructor
  BSplineModel()
      : offset_(torch::zeros({3}, Options<typename BSpline_t::value_type>{})),
        rotation_(
            torch::zeros({3}, Options<typename BSpline_t::value_type>{})) {}

  /// @brief Constructor for equidistant knot vectors
  BSplineModel(const std::array<int64_t, BSpline_t::parDim()> ncoeffs,
               enum iganet::init init = iganet::init::zeros)
      : BSpline_t(ncoeffs, init), solution_(ncoeffs, init),
        offset_(torch::zeros({3}, Options<typename BSpline_t::value_type>{})),
        rotation_(
            torch::zeros({3}, Options<typename BSpline_t::value_type>{})) {
    if constexpr (BSpline_t::parDim() == 1)
      solution_.transform([](const std::array<typename BSpline_t::value_type, 1>
                                 xi) {
        return std::array<typename BSpline_t::value_type, BSpline_t::geoDim()>{
            static_cast<iganet::real_t>(std::sin(M_PI * xi[0])), 0.0, 0.0};
      });

    else if constexpr (BSpline_t::parDim() == 2)
      solution_.transform([](const std::array<typename BSpline_t::value_type, 2>
                                 xi) {
        return std::array<typename BSpline_t::value_type, BSpline_t::geoDim()>{
            static_cast<iganet::real_t>(std::sin(M_PI * xi[0]) *
                                        std::sin(M_PI * xi[1])),
            0.0, 0.0};
      });

    else if constexpr (BSpline_t::parDim() == 3)
      solution_.transform([](const std::array<typename BSpline_t::value_type, 3>
                                 xi) {
        return std::array<typename BSpline_t::value_type, BSpline_t::geoDim()>{
            static_cast<iganet::real_t>(std::sin(M_PI * xi[0]) *
                                        std::sin(M_PI * xi[1]) *
                                        std::sin(M_PI * xi[2])),
            0.0, 0.0};
      });
  }

  /// @brief Destructor
  ~BSplineModel() {}

  /// @brief Returns the model's name
  std::string getName() const override {
    if constexpr (BSpline_t::parDim() == 1)
      return "BSplineCurve";
    else if constexpr (BSpline_t::parDim() == 2)
      return "BSplineSurface";
    else if constexpr (BSpline_t::parDim() == 3)
      return "BSplineVolume";
    else if constexpr (BSpline_t::parDim() == 4)
      return "BSplineHyperVolume";
    else
      return "invalidName";
  }

  /// @brief Returns the model's description
  std::string getDescription() const override {
    if constexpr (BSpline_t::parDim() == 1)
      return "B-spline curve";
    else if constexpr (BSpline_t::parDim() == 2)
      return "B-spline surface";
    else if constexpr (BSpline_t::parDim() == 3)
      return "B-spline volume";
    else if constexpr (BSpline_t::parDim() == 4)
      return "B-spline hypervolume";
    else
      return "invalidDescription";
  }

  /// @brief Returns the model's options
  nlohmann::json getOptions() const override {
    if constexpr (BSpline_t::parDim() == 1)
      return R"([{
             "name" : "degree",
             "description" : "Polynomial degree of the B-spline",
             "type" : "select",
             "value" : ["constant", "linear", "quadratic", "cubic", "quartic", "quintic"],
             "default" : 2,
             "uiid" : 0},{
             "name" : "ncoeffs",
             "description" : "Number of coefficients per parametric dimension",
             "type" : ["int"],
             "value" : [3],
             "default" : [3],
             "uiid" : 1},{
             "name" : "init",
             "description" : "Initialization of the coefficients",
             "type" : "select",
             "value" : ["zeros", "ones", "linear", "random", "greville"],
             "default" : 4,
             "uiid" : 2},{
             "name" : "nonuniform",
             "description" : "Create non-uniform B-spline",
             "type" : "select",
             "value" : ["false", "true"],
             "default" : 0,
             "uiid" : 3}])"_json;

    else if constexpr (BSpline_t::parDim() == 2)
      return R"([{
             "name" : "degree",
             "description" : "Polynomial degree of the B-spline",
             "type" : "select",
             "value" : ["constant", "linear", "quadratic", "cubic", "quartic", "quintic"],
             "default" : 2,
             "uiid" : 0},{
             "name" : "ncoeffs",
             "description" : "Number of coefficients per parametric dimension",
             "type" : ["int","int"],
             "value" : [3,3],
             "default" : [3,3],
             "uiid" : 1},{
             "name" : "init",
             "description" : "Initialization of the coefficients",
             "type" : "select",
             "value" : ["zeros", "ones", "linear", "random", "greville"],
             "default" : 4,
             "uiid" : 2},{
             "name" : "nonuniform",
             "description" : "Create non-uniform B-spline",
             "type" : "select",
             "value" : ["false", "true"],
             "default" : 0,
             "uiid" : 3}])"_json;

    else if constexpr (BSpline_t::parDim() == 3)
      return R"([{
             "name" : "degree",
             "description" : "Polynomial degree of the B-spline",
             "type" : "select",
             "value" : ["constant", "linear", "quadratic", "cubic", "quartic", "quintic"],
             "default" : 2,
             "uiid" : 0},{
             "name" : "ncoeffs",
             "description" : "Number of coefficients per parametric dimension",
             "type" : ["int","int","int"],
             "value" : [3,3,3],
             "default" : [3,3,3],
             "uiid" : 1},{
             "name" : "init",
             "description" : "Initialization of the coefficients",
             "type" : "select",
             "value" : ["zeros", "ones", "linear", "random", "greville"],
             "default" : 4,
             "uiid" : 2},{
             "name" : "nonuniform",
             "description" : "Create non-uniform B-spline",
             "type" : "select",
             "value" : ["false", "true"],
             "default" : 0,
             "uiid" : 3}])"_json;

    else if constexpr (BSpline_t::parDim() == 4)
      return R"([{
             "name" : "degree",
             "description" : "Polynomial degree of the B-spline",
             "type" : "select",
             "value" : ["constant", "linear", "quadratic", "cubic", "quartic", "quintic"],
             "default" : 2,
             "uiid" : 0},{
             "name" : "ncoeffs",
             "description" : "Number of coefficients per parametric dimension",
             "type" : [int,int,int,int],
             "value" : [3,3,3,3],
             "default" : [3,3,3,3],
             "uiid" : 1},{
             "name" : "init",
             "description" : "Initialization of the coefficients",
             "type" : "select",
             "value" : ["zeros", "ones", "linear", "random", "greville"],
             "default" : 4,
             "uiid" : 2},{
             "name" : "nonuniform",
             "description" : "Create non-uniform B-spline",
             "type" : "select",
             "value" : ["false", "true"],
             "default" : 0,
             "uiid" : 3}])"_json;

    else
      return R"({ INVALID REQUEST })"_json;
  }

  /// @brief Returns the model's inputs
  nlohmann::json getInputs() const override {
    return R"([{
              "name" : "geometry",
              "description" : "Geometry",
              "type" : 2}])"_json;
  }

  /// @brief Returns the model's outputs
  nlohmann::json getOutputs() const override {
    if constexpr (BSpline_t::geoDim() == 1)
      return R"([{
                "name" : "ValueFieldMagnitude",
                "description" : "Magnitude of the B-spline values",
                "type" : 1}])"_json;
    else
      return R"([{
                "name" : "ValueFieldMagnitude",
                "description" : "Magnitude of the B-spline values",
                "type" : 1},{
                "name" : "ValueField",
                "description" : "B-spline values",
                "type" : 2}])"_json;
  }

  /// @brief Serializes the model to JSON
  nlohmann::json to_json(const std::string &component,
                         const std::string &attribute) const override {

    if (component == "geometry" ||
        component == "" /* might be removed in the future */) {
      if (attribute != "") {
        nlohmann::json json;
        if (attribute == "degrees")
          json["degrees"] = this->degrees();
        else if (attribute == "geoDim")
          json["geoDim"] = this->geoDim();
        else if (attribute == "parDim")
          json["parDim"] = this->parDim();
        else if (attribute == "ncoeffs")
          json["ncoeffs"] = this->ncoeffs();
        else if (attribute == "nknots")
          json["nknots"] = this->nknots();
        else if (attribute == "coeffs")
          json["coeffs"] = this->coeffs_to_json();
        else if (attribute == "knots")
          json["knots"] = this->knots_to_json();

        return json;

      } else {
        auto json = BSpline_t::to_json();
        json.update(Model::to_json("transform", ""), true);

        return json;
      }
    }

    else if (component == "solution") {
      if (attribute != "") {
        nlohmann::json json;
        if (attribute == "degrees")
          json["degrees"] = solution_.degrees();
        else if (attribute == "geoDim")
          json["geoDim"] = solution_.geoDim();
        else if (attribute == "parDim")
          json["parDim"] = solution_.parDim();
        else if (attribute == "ncoeffs")
          json["ncoeffs"] = solution_.ncoeffs();
        else if (attribute == "nknots")
          json["nknots"] = solution_.nknots();
        else if (attribute == "coeffs")
          json["coeffs"] = solution_.coeffs_to_json();
        else if (attribute == "knots")
          json["knots"] = solution_.knots_to_json();

        return json;
      } else

        return solution_.to_json();
    }

    else
      return Model::to_json(component, attribute);
  }

  /// @brief Updates the attributes of the model
  nlohmann::json updateAttribute(const std::string &component,
                                 const std::string &attribute,
                                 const nlohmann::json &json) override {
    if (attribute == "coeffs") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("indices") || !json["data"].contains("coeffs"))
        throw InvalidModelAttributeException();

      auto indices = json["data"]["indices"].get<std::vector<int64_t>>();
      auto coeffs_cpu =
          utils::to_tensorAccessor<typename BSpline_t::value_type, 1>(
              BSpline_t::coeffs(), torch::kCPU);

      switch (BSpline_t::geoDim()) {
      case (1): {
        auto coords =
            json["data"]["coeffs"]
                .get<std::vector<std::tuple<typename BSpline_t::value_type>>>();
        auto xAccessor = std::get<1>(coeffs_cpu)[0];

        for (const auto &[index, coord] : iganet::utils::zip(indices, coords)) {
          if (index < 0 || index >= BSpline_t::ncumcoeffs())
            throw IndexOutOfBoundsException();
          xAccessor[index] = std::get<0>(coord);
        }
        break;
      }
      case (2): {
        auto coords =
            json["data"]["coeffs"]
                .get<std::vector<std::tuple<typename BSpline_t::value_type,
                                            typename BSpline_t::value_type>>>();
        auto xAccessor = std::get<1>(coeffs_cpu)[0];
        auto yAccessor = std::get<1>(coeffs_cpu)[1];

        for (const auto &[index, coord] : iganet::utils::zip(indices, coords)) {
          if (index < 0 || index >= BSpline_t::ncumcoeffs())
            throw IndexOutOfBoundsException();

          xAccessor[index] = std::get<0>(coord);
          yAccessor[index] = std::get<1>(coord);
        }
        break;
      }
      case (3): {
        auto coords =
            json["data"]["coeffs"]
                .get<std::vector<std::tuple<typename BSpline_t::value_type,
                                            typename BSpline_t::value_type,
                                            typename BSpline_t::value_type>>>();
        auto xAccessor = std::get<1>(coeffs_cpu)[0];
        auto yAccessor = std::get<1>(coeffs_cpu)[1];
        auto zAccessor = std::get<1>(coeffs_cpu)[2];

        for (const auto &[index, coord] : iganet::utils::zip(indices, coords)) {
          if (index < 0 || index >= BSpline_t::ncumcoeffs())
            throw IndexOutOfBoundsException();

          xAccessor[index] = std::get<0>(coord);
          yAccessor[index] = std::get<1>(coord);
          zAccessor[index] = std::get<2>(coord);
        }
        break;
      }
      case (4): {
        auto coords =
            json["data"]["coeffs"]
                .get<std::vector<std::tuple<typename BSpline_t::value_type,
                                            typename BSpline_t::value_type,
                                            typename BSpline_t::value_type,
                                            typename BSpline_t::value_type>>>();
        auto xAccessor = std::get<1>(coeffs_cpu)[0];
        auto yAccessor = std::get<1>(coeffs_cpu)[1];
        auto zAccessor = std::get<1>(coeffs_cpu)[2];
        auto tAccessor = std::get<1>(coeffs_cpu)[3];

        for (const auto &[index, coord] : iganet::utils::zip(indices, coords)) {
          if (index < 0 || index >= BSpline_t::ncumcoeffs())
            throw IndexOutOfBoundsException();

          xAccessor[index] = std::get<0>(coord);
          yAccessor[index] = std::get<1>(coord);
          zAccessor[index] = std::get<2>(coord);
          tAccessor[index] = std::get<3>(coord);
        }
        break;
      }
      default:
        throw InvalidModelAttributeException();
      }
      return "{}";
    } else
      return Model::updateAttribute(component, attribute, json);
  }

  /// @brief Evaluates the model
  nlohmann::json eval(const std::string &component,
                      const nlohmann::json &json) const override {

    if constexpr (BSpline_t::parDim() == 1) {

      std::array<int64_t, 1> res({25});
      if (json.contains("data"))
        if (json["data"].contains("resolution"))
          res = json["data"]["resolution"].get<std::array<int64_t, 1>>();

      utils::TensorArray1 xi = {torch::linspace(0, 1, res[0])};

      if (component == "ValueFieldMagnitude") {
        return nlohmann::json::array().emplace_back(
            utils::to_json<iganet::real_t, 1>(*(solution_.eval(xi)[0])));
      } else if (component == "ValueField") {
        auto values = BSpline_t::eval(xi);
        auto result = nlohmann::json::array();
        for (short_t dim = 0; dim < BSpline_t::geoDim(); ++dim)
          result.emplace_back(
              utils::to_json<iganet::real_t, 1>(*(values[dim])));
        return result;
      } else
        return R"({ INVALID REQUEST })"_json;
    }

    else if constexpr (BSpline_t::parDim() == 2) {

      std::array<int64_t, 2> res({25, 25});
      if (json.contains("data"))
        if (json["data"].contains("resolution"))
          res = json["data"]["resolution"].get<std::array<int64_t, 2>>();

      utils::TensorArray2 xi = utils::to_array<2>(torch::meshgrid(
          {torch::linspace(0, 1, res[0],
                           Options<typename BSpline_t::value_type>{}),
           torch::linspace(0, 1, res[1],
                           Options<typename BSpline_t::value_type>{})},
          "xy"));

      if (component == "ValueFieldMagnitude") {
        return nlohmann::json::array().emplace_back(
            utils::to_json<iganet::real_t, 2>(*(solution_.eval(xi)[0])));
      } else if (component == "ValueField") {
        auto values = BSpline_t::eval(xi);
        auto result = nlohmann::json::array();
        for (short_t dim = 0; dim < BSpline_t::geoDim(); ++dim)
          result.emplace_back(
              utils::to_json<iganet::real_t, 2>(*(values[dim])));
        return result;
      } else
        return R"({ INVALID REQUEST })"_json;
    }

    else if constexpr (BSpline_t::parDim() == 3) {

      std::array<int64_t, 3> res({25, 25, 25});
      if (json.contains("data"))
        if (json["data"].contains("resolution"))
          res = json["data"]["resolution"].get<std::array<int64_t, 3>>();

      utils::TensorArray3 xi = utils::to_array<3>(torch::meshgrid(
          {torch::linspace(0, 1, res[0],
                           Options<typename BSpline_t::value_type>{}),
           torch::linspace(0, 1, res[1],
                           Options<typename BSpline_t::value_type>{}),
           torch::linspace(0, 1, res[2],
                           Options<typename BSpline_t::value_type>{})},
          "xy"));

      if (component == "ValueFieldMagnitude") {
        return nlohmann::json::array().emplace_back(
            utils::to_json<iganet::real_t, 3>(*(solution_.eval(xi)[0])));
      } else if (component == "ValueField") {
        auto values = BSpline_t::eval(xi);
        auto result = nlohmann::json::array();
        for (short_t dim = 0; dim < BSpline_t::geoDim(); ++dim)
          result.emplace_back(
              utils::to_json<iganet::real_t, 3>(*(values[dim])));
        return result;
      } else
        return R"({ INVALID REQUEST })"_json;
    }

    else if constexpr (BSpline_t::parDim() == 4) {

      std::array<int64_t, 4> res({25, 25, 25, 25});
      if (json.contains("data"))
        if (json["data"].contains("resolution"))
          res = json["data"]["resolution"].get<std::array<int64_t, 4>>();

      utils::TensorArray4 xi = utils::to_array<4>(torch::meshgrid(
          {torch::linspace(0, 1, res[0],
                           Options<typename BSpline_t::value_type>{}),
           torch::linspace(0, 1, res[1],
                           Options<typename BSpline_t::value_type>{}),
           torch::linspace(0, 1, res[2],
                           Options<typename BSpline_t::value_type>{}),
           torch::linspace(0, 1, res[3],
                           Options<typename BSpline_t::value_type>{})},
          "xy"));

      if (component == "ValueFieldMagnitude") {
        return nlohmann::json::array().emplace_back(
            utils::to_json<iganet::real_t, 4>(*(solution_.eval(xi)[0])));
      } else if (component == "ValueField") {
        auto values = BSpline_t::eval(xi);
        auto result = nlohmann::json::array();
        for (short_t dim = 0; dim < BSpline_t::geoDim(); ++dim)
          result.emplace_back(
              utils::to_json<iganet::real_t, 4>(*(values[dim])));
        return result;
      } else
        return R"({ INVALID REQUEST })"_json;
    }
  }

  /// @brief Refines the model
  void refine(const nlohmann::json &json = NULL) override {
    int num = 1, dim = -1;

    if (json.contains("data")) {
      if (json["data"].contains("num"))
        num = json["data"]["num"].get<int>();

      if (json["data"].contains("dim"))
        dim = json["data"]["dim"].get<int>();
    }

    BSpline_t::uniform_refine(num, dim);

    solution_.uniform_refine(num, dim);
    if constexpr (BSpline_t::parDim() == 1)
      solution_.transform([](const std::array<typename BSpline_t::value_type, 1>
                                 xi) {
        return std::array<typename BSpline_t::value_type, BSpline_t::geoDim()>{
            static_cast<iganet::real_t>(std::sin(M_PI * xi[0])), 0.0, 0.0};
      });

    else if constexpr (BSpline_t::parDim() == 2)
      solution_.transform([](const std::array<typename BSpline_t::value_type, 2>
                                 xi) {
        return std::array<typename BSpline_t::value_type, BSpline_t::geoDim()>{
            static_cast<iganet::real_t>(std::sin(M_PI * xi[0]) *
                                        std::sin(M_PI * xi[1])),
            0.0, 0.0};
      });

    else if constexpr (BSpline_t::parDim() == 3)
      solution_.transform([](const std::array<typename BSpline_t::value_type, 3>
                                 xi) {
        return std::array<typename BSpline_t::value_type, BSpline_t::geoDim()>{
            static_cast<iganet::real_t>(std::sin(M_PI * xi[0]) *
                                        std::sin(M_PI * xi[1]) *
                                        std::sin(M_PI * xi[2])),
            0.0, 0.0};
      });
  }

  /// @brief Reparameterize the model
  void reparameterize(const nlohmann::json &json = NULL) override {

    // gismo::gsBarrierPatch<d, T> opt(geo_, false);
    // opt.options().setInt("ParamMethod", 1);
    // opt.compute();

    // geo_ = opt.result();
  }

  /// @brief Loads model from LibTorch file
  void load(const nlohmann::json &json) override {

    if (json.contains("data")) {
      if (json["data"].contains("binary")) {

        // get binary vector from JSON object
        std::vector<std::uint8_t> binary = json["data"]["binary"];

        // recover input archive from binary vector
        torch::serialize::InputArchive archive;
        archive.load_from(reinterpret_cast<const char *>(binary.data()),
                          binary.size());

        archive.read("transform", transform_);
        BSpline_t::read(archive, "geometry");
        solution_.read(archive, "solution");

        return;
      }
    }

    throw InvalidModelException();
  }

  /// @brief Saves model to LibTorch file
  nlohmann::json save() const override {

    // serialize model to output archive
    torch::serialize::OutputArchive archive;
    archive.write("model",
                  static_cast<int64_t>(std::hash<std::string>{}(getName())));
    archive.write("nonuniform", static_cast<bool>(BSpline_t::is_nonuniform()));
    archive.write("transform", transform_);

    BSpline_t::write(archive, "geometry");
    solution_.write(archive, "solution");

    // store output archive in binary vector
    std::vector<std::uint8_t> binary;

    archive.save_to(
        [&binary](const void *data, size_t size) mutable -> std::size_t {
          auto data_ = reinterpret_cast<const std::uint8_t *>(data);

          for (std::size_t i = 0; i < size; ++i)
            binary.push_back(data_[i]);

          return size;
        });

    // attach binary vector to JSON object
    nlohmann::json json;
    json["binary"] = binary;

    return json;
  }

  /// @brief Imports the model from XML (as JSON object)
  void importXML(const nlohmann::json &json, const std::string &component,
                 int id) override {

    if (json.contains("data")) {
      if (json["data"].contains("xml")) {

        std::string xml = json["data"]["xml"].get<std::string>();

        pugi::xml_document doc;
        pugi::xml_parse_result result =
            doc.load_buffer(xml.c_str(), xml.size());

        if (pugi::xml_node root = doc.child("xml"))
          importXML(root, component, id);
        else
          throw std::runtime_error("No \"xml\" node in XML object");

        return;
      }
    }

    throw std::runtime_error("No XML node in JSON object");
  }

  /// @brief Imports the model from XML (as XML object)
  void importXML(const pugi::xml_node &xml, const std::string &component,
                 int id) override {

    if (component.empty()) {
      BSpline_t::from_xml(xml, id, "geometry");
      solution_.from_xml(xml, id, "solution");
      iganet::utils::from_xml<iganet::real_t, 2>(xml, transform_, "Matrix", id,
                                                 "transform", false);
    } else {
      if (component == "geometry") {
        BSpline_t::from_xml(xml, id, "geometry");
        iganet::utils::from_xml<iganet::real_t, 2>(xml, transform_, "Matrix",
                                                   id, "transform", false);
      } else if (component == "solution")
        solution_.from_xml(xml, id, "solution");
      else
        throw std::runtime_error("Unsupported component");
    }
  }

  /// @brief Exports the model to XML (as JSON object)
  nlohmann::json exportXML(const std::string &component, int id) override {

    // serialize to XML
    pugi::xml_document doc;
    pugi::xml_node xml = doc.append_child("xml");
    xml = exportXML(xml, component, id);

    // serialize to JSON
    std::ostringstream oss;
    doc.save(oss);

    return oss.str();
  }

  /// @brief Exports the model to XML (as XML object)
  pugi::xml_node &exportXML(pugi::xml_node &xml, const std::string &component,
                            int id) override {

    if (component.empty()) {
      BSpline_t::to_xml(xml, id, "geometry");
      solution_.to_xml(xml, id, "solution");
      iganet::utils::to_xml<iganet::real_t, 2>(transform_, xml, "Matrix", id,
                                               "transform");
    } else {
      if (component == "geometry") {
        BSpline_t::to_xml(xml, id, "geometry");
        iganet::utils::to_xml<iganet::real_t, 2>(transform_, xml, "Matrix", id,
                                                 "transform");
      } else if (component == "solution")
        solution_.to_xml(xml, id, "solution");
      else
        throw std::runtime_error("Unsupported component");
    }
    return xml;
  }
};

} // namespace webapp
} // namespace iganet
