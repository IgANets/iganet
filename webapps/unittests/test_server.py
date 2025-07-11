import os
import unittest
import xml.etree.ElementTree as ET

from wsiganet import *


class TestSession(unittest.TestCase):
    def setUp(self):
        self.protocol = os.environ.get("PROTOCOL", "ws")
        self.hostname = os.environ.get("HOSTNAME", "localhost")
        self.port = os.environ.get("PORT", "9001")
        self.certfile = os.environ.get("CERTFILE", "cert.pem")
        self.keyfile = os.environ.get("KEYFILE", "key.pem")
        self.password = os.environ.get("PASSWORD", "")

        # Establish connection
        context = ssl._create_unverified_context()
        self.ws = create_connection(
            self.protocol + "://" + self.hostname + ":" + self.port,
            sslopt={
                "certfile": self.certfile,
                "keyfile": self.keyfile,
                "password": self.password,
                "cert_reqs": ssl.CERT_NONE,
                "check_hostname": False,
                "context": context,
                "verify": False,
            },
        )

        # Get list of sessions
        self.session_ids = get_sessions(self.ws)

        # Create a new session
        self.session_id, _ = create_session(self.ws)

    def tearDown(self):
        # Remove session
        remove_session(self.ws, self.session_id)

        # Check that list of sessions has not changed
        self.assertEqual(self.session_ids, get_sessions(self.ws))

        # Close connection
        self.ws.close()

    def test_create_remove(self):
        # Check that list of sessions has changed
        self.assertNotEqual(self.session_ids, get_sessions(self.ws))

    def test_connect_disconnect(self):
        # Check that list of sessions has changed
        self.assertNotEqual(self.session_ids, get_sessions(self.ws))

        # Disconnect from session
        disconnect_session(self.ws, self.session_id)

        # Check that list of sessions has changed
        self.assertNotEqual(self.session_ids, get_sessions(self.ws))

        # Reconnect to session
        connect_session(self.ws, self.session_id)

    def test_broadcast(self):
        # Establish connection
        ws = create_connection(
            self.protocol + "://" + self.hostname + ":" + self.port
        )

        # Connect to existing session
        connect_session(ws, self.session_id)

        # Create a new BSpline surface
        model, _ = create_BSplineSurface(self.ws, self.session_id)

        # Receive broadcast message
        self.assertTrue(ws.recv())

        # Disconnect from session
        disconnect_session(self.ws, self.session_id)

        # Close connection
        ws.close()

    def test_exportxml(self):
        # Create three new BSpline objects
        model0, _ = create_BSplineCurve(self.ws, self.session_id)
        model1, _ = create_BSplineSurface(self.ws, self.session_id)
        model2, _ = create_BSplineVolume(self.ws, self.session_id)

        data = export_session_xml(self.ws, self.session_id)
        xml1 = ET.fromstring(data["xml"])

        xml2 = ET.parse(
            os.path.join(
                os.path.dirname(__file__), "../filedata/xml/Session.xml"
            )
        )

        self.assertEqual(ET.tostring(xml1), ET.tostring(xml2.getroot()))

        # Remove model instances
        remove_model(self.ws, self.session_id, model0)
        remove_model(self.ws, self.session_id, model1)
        remove_model(self.ws, self.session_id, model2)

    def test_importxml(self):
        # Create three new BSpline objects
        model0, data0 = create_BSplineCurve(self.ws, self.session_id)
        model1, data1 = create_BSplineSurface(self.ws, self.session_id)
        model2, data2 = create_BSplineVolume(self.ws, self.session_id)

        # Change coefficients
        put_model_attribute(
            self.ws,
            self.session_id,
            model0,
            data0["inputs"][0]["name"],
            "coeffs",
            {"indices": [0, 3], "coeffs": [[0.0, 0.0, 0.0], [0.5, 1.0, 0.5]]},
        )
        put_model_attribute(
            self.ws,
            self.session_id,
            model1,
            data1["inputs"][0]["name"],
            "coeffs",
            {"indices": [0, 3], "coeffs": [[0.0, 0.0, 0.0], [0.5, 1.0, 0.5]]},
        )
        put_model_attribute(
            self.ws,
            self.session_id,
            model2,
            data2["inputs"][0]["name"],
            "coeffs",
            {"indices": [0, 3], "coeffs": [[0.0, 0.0, 0.0], [0.5, 1.0, 0.5]]},
        )

        # Export session from XML
        data = export_session_xml(self.ws, self.session_id)
        xml1 = ET.fromstring(data["xml"])

        xml2 = ET.parse(
            os.path.join(
                os.path.dirname(__file__), "../filedata/xml/Session.xml"
            )
        )

        self.assertNotEqual(ET.tostring(xml1), ET.tostring(xml2.getroot()))

        # Import session from XML
        result = import_session_xml(
            self.ws, self.session_id, {"xml": str(ET.tostring(xml2.getroot()))}
        )

        # Export session from XML
        data = export_session_xml(self.ws, self.session_id)
        xml1 = ET.fromstring(data["xml"])

        self.assertEqual(ET.tostring(xml1), ET.tostring(xml2.getroot()))

        # Remove model instances
        remove_model(self.ws, self.session_id, model0)
        remove_model(self.ws, self.session_id, model1)
        remove_model(self.ws, self.session_id, model2)

    def test_save(self):
        import filecmp

        # Create three new BSpline objects
        model0, data0 = create_BSplineCurve(self.ws, self.session_id)
        model1, data1 = create_BSplineSurface(self.ws, self.session_id)
        model2, data2 = create_BSplineVolume(self.ws, self.session_id)

        # Save session to binary data
        data = save_session(self.ws, self.session_id)

        # Create XML root
        xml = ET.Element("Session")

        for index in range(0, len(data)):
            # Add to XML root
            model = ET.SubElement(
                xml,
                "Model",
                part=str(index),
                file="Session." + str(index) + ".pt",
            )

            # Open binary reference file for writing
            with open("Session_tmp." + str(index) + ".pt", "wb") as file:
                for byte in data[index]["binary"]:
                    file.write(byte.to_bytes(1, byteorder="big"))

            self.assertTrue(
                filecmp.cmp(
                    "Session_tmp." + str(index) + ".pt",
                    os.path.join(
                        os.path.dirname(__file__),
                        "../filedata/pytorch/Session." + str(index) + ".pt",
                    ),
                )
            )

            # Remove model file
            os.remove("Session_tmp." + str(index) + ".pt")

        # Write session file
        ET.ElementTree(xml).write("Session_tmp.ptc")

        xml1 = ET.parse("Session_tmp.ptc")
        xml2 = ET.parse(
            os.path.join(
                os.path.dirname(__file__), "../filedata/pytorch/Session.ptc"
            )
        )

        self.assertEqual(
            ET.tostring(xml1.getroot()), ET.tostring(xml2.getroot())
        )

        os.remove("Session_tmp.ptc")

    def test_load(self):
        import filecmp

        # Open and read session file
        xml = ET.parse(
            os.path.join(
                os.path.dirname(__file__), "../filedata/pytorch/Session.ptc"
            )
        )

        binary = []
        # Load binary files
        for model in xml.getroot():
            with open(
                os.path.join(
                    os.path.dirname(__file__),
                    "../filedata/pytorch/" + model.attrib["file"],
                ),
                "rb",
            ) as file:
                data = file.read()
                binary.append([int(byte) for byte in data])

        # Load session from binary data
        session_id, _ = load_session(self.ws, {"binary": binary})

        # Get list of model instances
        models = get_models(self.ws, session_id)

        # Remove model instances
        for model in models:
            remove_model(self.ws, session_id, str(model))

        # Remove session
        remove_session(self.ws, session_id)


class TestBSplineSurface(unittest.TestCase):
    def setUp(self):
        self.protocol = os.environ.get("PROTOCOL", "ws")
        self.hostname = os.environ.get("HOSTNAME", "localhost")
        self.port = os.environ.get("PORT", "9001")

        # Establish connection
        self.ws = create_connection(
            self.protocol + "://" + self.hostname + ":" + self.port
        )

        # Get list of sessions
        self.session_ids = get_sessions(self.ws)

        # Create a new session
        self.session_id, _ = create_session(self.ws)

        # Get list of model instances
        self.models = get_models(self.ws, self.session_id)

        # Create a new BSpline surface
        self.model, self.data = create_BSplineSurface(
            self.ws, self.session_id, degree=1, init=4, ncoeffs=[3, 2]
        )

        # Get component name
        self.component = self.data["inputs"][0]["name"]

    def tearDown(self):
        # Remove model instance
        remove_model(self.ws, self.session_id, self.model)

        # Check that list of model instances has not changed
        self.assertEqual(self.models, get_models(self.ws, self.session_id))

        # Remove session
        remove_session(self.ws, self.session_id)

        # Check that list of sessions has not changed
        self.assertEqual(self.session_ids, get_sessions(self.ws))

        # Close connection
        self.ws.close()

    def test_create_remove(self):
        # Check that list of model instances has changed
        self.assertNotEqual(self.models, get_models(self.ws, self.session_id))

    def test_get_component(self):
        # Get component
        component = get_model_component(
            self.ws, self.session_id, self.model, self.component
        )

        self.assertTrue(component)

    def test_get_attributes(self):
        # Check degrees
        self.assertEqual(
            get_model_attribute(
                self.ws, self.session_id, self.model, self.component, "degrees"
            )["degrees"],
            [1, 1],
        )

        # Check number of coefficients
        self.assertEqual(
            get_model_attribute(
                self.ws, self.session_id, self.model, self.component, "ncoeffs"
            )["ncoeffs"],
            [3, 2],
        )

        # Check number of knots
        self.assertEqual(
            get_model_attribute(
                self.ws, self.session_id, self.model, self.component, "nknots"
            )["nknots"],
            [5, 4],
        )

        # Check coefficients
        self.assertEqual(
            get_model_attribute(
                self.ws, self.session_id, self.model, self.component, "coeffs"
            )["coeffs"],
            [
                [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        )

        # Check knots
        self.assertEqual(
            get_model_attribute(
                self.ws, self.session_id, self.model, self.component, "knots"
            )["knots"],
            [[0.0, 0.0, 0.5, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        )

    def test_put_attributes(self):
        # Check coefficients
        self.assertEqual(
            get_model_attribute(
                self.ws, self.session_id, self.model, self.component, "coeffs"
            )["coeffs"],
            [
                [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        )

        # Change coefficients
        put_model_attribute(
            self.ws,
            self.session_id,
            self.model,
            self.component,
            "coeffs",
            {"indices": [0, 3], "coeffs": [[0.0, 0.0, 0.0], [0.5, 1.0, 0.5]]},
        )

        # Check updated coefficients
        self.assertEqual(
            get_model_attribute(
                self.ws, self.session_id, self.model, self.component, "coeffs"
            )["coeffs"],
            [
                [0.0, 0.5, 1.0, 0.5, 0.5, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.5, 1.0, 1.0],
            ],
        )

    def test_exportxml(self):
        # Export model to XML
        data = export_model_xml(self.ws, self.session_id, self.model)
        xml1 = ET.fromstring(data["xml"])
        xml2 = ET.parse(
            os.path.join(
                os.path.dirname(__file__), "../filedata/xml/BSplineSurface.xml"
            )
        )

        self.assertEqual(ET.tostring(xml1), ET.tostring(xml2.getroot()))

    def test_exportxml_component(self):
        # Export model component to XML
        data = export_model_component_xml(
            self.ws, self.session_id, self.model, "geometry"
        )
        xml1 = ET.fromstring(data["xml"])
        xml2 = ET.parse(
            os.path.join(
                os.path.dirname(__file__),
                "../filedata/xml/BSplineSurfaceComponent.xml",
            )
        )

        self.assertEqual(ET.tostring(xml1), ET.tostring(xml2.getroot()))

    def test_importxml(self):
        # Change coefficients
        put_model_attribute(
            self.ws,
            self.session_id,
            self.model,
            self.component,
            "coeffs",
            {"indices": [0, 3], "coeffs": [[0.0, 0.0, 0.0], [0.5, 1.0, 0.5]]},
        )

        # Export model to XML
        data = export_model_xml(self.ws, self.session_id, self.model)
        xml1 = ET.fromstring(data["xml"])
        xml2 = ET.parse(
            os.path.join(
                os.path.dirname(__file__), "../filedata/xml/BSplineSurface.xml"
            )
        )

        self.assertNotEqual(ET.tostring(xml1), ET.tostring(xml2.getroot()))

        # Import model from XML
        result = import_model_xml(
            self.ws,
            self.session_id,
            self.model,
            {"xml": str(ET.tostring(xml2.getroot()))},
        )

        # Export model to XML
        data = export_model_xml(self.ws, self.session_id, self.model)
        xml1 = ET.fromstring(data["xml"])

        self.assertEqual(ET.tostring(xml1), ET.tostring(xml2.getroot()))

    def test_importxml_component(self):
        # Change coefficients
        put_model_attribute(
            self.ws,
            self.session_id,
            self.model,
            self.component,
            "coeffs",
            {"indices": [0, 3], "coeffs": [[0.0, 0.0, 0.0], [0.5, 1.0, 0.5]]},
        )

        # Export model to XML
        data = export_model_xml(self.ws, self.session_id, self.model)
        xml1 = ET.fromstring(data["xml"])
        xml2 = ET.parse(
            os.path.join(
                os.path.dirname(__file__),
                "../filedata/xml/BSplineSurfaceComponent.xml",
            )
        )

        self.assertNotEqual(ET.tostring(xml1), ET.tostring(xml2.getroot()))

        # Import model from XML
        result = import_model_component_xml(
            self.ws,
            self.session_id,
            self.model,
            "geometry",
            {"xml": str(ET.tostring(xml2.getroot()))},
        )

        # Export model to XML
        data = export_model_component_xml(
            self.ws, self.session_id, self.model, "geometry"
        )
        xml1 = ET.fromstring(data["xml"])

        self.assertEqual(ET.tostring(xml1), ET.tostring(xml2.getroot()))

    def test_save(self):
        import filecmp

        # Save model to binary data
        data = save_model(self.ws, self.session_id, self.model)

        # Open binary reference file for writing
        with open("BSplineSurface_tmp.pt", "wb") as file:
            for byte in data["binary"]:
                file.write(byte.to_bytes(1, byteorder="big"))

        self.assertTrue(
            filecmp.cmp(
                "BSplineSurface_tmp.pt",
                os.path.join(
                    os.path.dirname(__file__),
                    "../filedata/pytorch/BSplineSurface.pt",
                ),
            )
        )

        os.remove("BSplineSurface_tmp.pt")

    def test_load(self):
        import filecmp

        # Open binary file for reading
        file = open(
            os.path.join(
                os.path.dirname(__file__),
                "../filedata/pytorch/BSplineSurface.pt",
            ),
            "rb",
        )
        data = file.read()
        file.close()

        # Load model from binary data
        binary = [int(byte) for byte in data]
        model, _ = load_model(self.ws, self.session_id, {"binary": binary})

        # Save model to binary data
        data = save_model(self.ws, self.session_id, model)

        # Open binary reference file for writing
        with open("BSplineSurface_tmp.pt", "wb") as file:
            for byte in data["binary"]:
                file.write(byte.to_bytes(1, byteorder="big"))

        self.assertTrue(
            filecmp.cmp(
                "BSplineSurface_tmp.pt",
                os.path.join(
                    os.path.dirname(__file__),
                    "../filedata/pytorch/BSplineSurface.pt",
                ),
            )
        )

        os.remove("BSplineSurface_tmp.pt")

        remove_model(self.ws, self.session_id, model)


if __name__ == "__main__":
    unittest.main()
