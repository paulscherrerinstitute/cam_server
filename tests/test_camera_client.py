import base64
import os
import unittest
from multiprocessing import Process
from time import sleep

import numpy
from bsread import source, SUB

from cam_server import CamClient
from cam_server.start_camera_server import start_camera_server
from cam_server.utils import get_host_port_from_stream_address
from tests import test_cleanup, get_simulated_camera


class CameraClientTest(unittest.TestCase):
    def setUp(self):
        self.host = "0.0.0.0"
        self.port = 8888

        test_base_dir = os.path.split(os.path.abspath(__file__))[0]
        self.config_folder = os.path.join(test_base_dir, "camera_config/")
        self.user_scripts_folder = os.path.join(test_base_dir, "user_scripts/")

        self.process = Process(target=start_camera_server, args=(self.host, self.port, self.config_folder, self.user_scripts_folder))
        self.process.start()

        # Give it some time to start.
        sleep(1.0)

        server_address = "http://%s:%s" % (self.host, self.port)
        self.client = CamClient(server_address)

    def tearDown(self):
        test_cleanup([self.client], [self.process], [os.path.join(self.config_folder, "simulation_temp.json")])


    def test_client(self):
        server_info = self.client.get_server_info()
        self.assertIsNot(server_info["active_instances"],
                         "There should be no running instances.")

        expected_cameras = set(["camera_example_1", "camera_example_2", "camera_example_3", "camera_example_4",
                                "simulation", "simulation2"])

        #self.assertSetEqual(set(self.client.get_cameras()), expected_cameras, "Not getting all expected cameras")
        for camera in  expected_cameras:
            self.assertIn(camera, set(self.client.get_cameras()), "Not getting expected camera: " + camera)


        camera_stream_address = self.client.get_instance_stream("simulation")

        self.assertTrue(bool(camera_stream_address), "Camera stream address cannot be empty.")
        self.assertTrue(self.client.is_instance_running("simulation"), "Simulation camera not present in server info.")

        # Check if we can connect to the stream and receive data (in less than 2 seconds).
        host, port = get_host_port_from_stream_address(camera_stream_address)
        with source(host=host, port=port, receive_timeout=2000, mode=SUB) as stream:
            data = stream.receive()
            self.assertIsNotNone(data, "Received data was none.")

            required_fields = set(["image", "timestamp", "width", "height", "x_axis", "y_axis"])
            self.assertSetEqual(required_fields, set(data.data.data.keys()), "Required fields missing.")

            image = data.data.data["image"].value
            x_size, y_size = get_simulated_camera().get_geometry()
            self.assertListEqual(list(image.shape), [y_size, x_size],
                                 "Original and received image are not the same.")

            self.assertEqual(data.data.data["width"].value, x_size, "Width not correct.")
            self.assertEqual(data.data.data["height"].value, y_size, "Height not correct.")

        # Stop the simulation instance.
        self.client.stop_instance("simulation")

        self.assertTrue(not self.client.is_instance_running("simulation"), "Camera simulation did not stop.")

        self.client.get_instance_stream("simulation")

        self.assertTrue(self.client.is_instance_running("simulation"), "Camera simulation did not start.")

        self.client.stop_all_instances()

        self.assertTrue(not self.client.is_instance_running("simulation"), "Camera simulation did not stop.")

        example_1_config = self.client.get_camera_config("camera_example_1")

        self.assertTrue(bool(example_1_config), "Cannot retrieve config.")

        # Change the name to reflect tha camera.
        example_1_config["name"] = "testing_camera"

        self.client.set_camera_config("testing_camera", example_1_config)

        testing_camera_config = self.client.get_camera_config("testing_camera")

        self.assertDictEqual(example_1_config, testing_camera_config, "Saved and loaded configs are not the same.")

        geometry = self.client.get_camera_geometry("simulation")
        simulated_camera = get_simulated_camera()
        size_x, size_y = simulated_camera.get_geometry()
        self.assertListEqual(geometry, [size_x, size_y],
                             'The geometry of the simulated camera is not correct.')

        self.assertTrue("testing_camera" in self.client.get_cameras(), "Testing camera should be present.")

        self.client.delete_camera_config("testing_camera")

        self.assertTrue("testing_camera" not in self.client.get_cameras(), "Testing camera should not be present.")

        # Test if it fails quickly enough.
        with self.assertRaisesRegex(ValueError, "Camera with prefix EPICS_example_1 is offline"):
            self.client.get_instance_stream("camera_example_1")

        self.assertTrue(self.client.is_camera_online("simulation"), "Simulation should be always online")

        self.assertFalse(self.client.is_camera_online("camera_example_1"), "Epics not working in this tests.")

        self.client.set_camera_config("simulation_temp", self.client.get_camera_config("simulation"))

        stream_address = self.client.get_instance_stream("simulation_temp")
        camera_host, camera_port = get_host_port_from_stream_address(stream_address)
        sim_x, sim_y = get_simulated_camera().get_geometry()

        instance_info = self.client.get_server_info()["active_instances"]["simulation_temp"]
        self.assertTrue("last_start_time" in instance_info)
        self.assertTrue("statistics" in instance_info)

        # Collect from the pipeline.
        with source(host=camera_host, port=camera_port, mode=SUB) as stream:
            data = stream.receive()

            x_size = data.data.data["width"].value
            y_size = data.data.data["height"].value
            self.assertEqual(x_size, sim_x)
            self.assertEqual(y_size, sim_y)

            x_axis_1 = data.data.data["x_axis"].value
            y_axis_1 = data.data.data["y_axis"].value

            self.assertEqual(x_axis_1.shape[0], sim_x)
            self.assertEqual(y_axis_1.shape[0], sim_y)

        camera_config = self.client.get_camera_config("simulation_temp")
        camera_config["rotate"] = 1
        self.client.set_camera_config("simulation_temp", camera_config)
        sleep(0.5)

        # Collect from the pipeline.
        with source(host=camera_host, port=camera_port, mode=SUB) as stream:
            data = stream.receive()

            x_size = data.data.data["width"].value
            y_size = data.data.data["height"].value

            # We rotate the image for 90 degrees - X and Y size should be inverted.
            self.assertEqual(x_size, sim_y)
            self.assertEqual(y_size, sim_x)

            x_axis_2 = data.data.data["x_axis"].value
            y_axis_2 = data.data.data["y_axis"].value

            # We rotate the image for 90 degrees - X and Y size should be inverted.
            self.assertEqual(x_axis_2.shape[0], sim_y)
            self.assertEqual(y_axis_2.shape[0], sim_x)

        self.client.delete_camera_config("simulation_temp")

        image = self.client.get_camera_image("simulation")
        self.assertGreater(len(image.content), 0)

        image = self.client.get_camera_image_bytes("simulation")
        dtype = image["dtype"]
        shape = image["shape"]
        bytes = base64.b64decode(image["bytes"].encode())

        x_size, y_size = get_simulated_camera().get_geometry()
        self.assertEqual(shape, [y_size, x_size])

        image_array = numpy.frombuffer(bytes, dtype=dtype).reshape(shape)
        self.assertIsNotNone(image_array)

        self.client.stop_all_instances()


if __name__ == '__main__':
    unittest.main()
