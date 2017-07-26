import os
import signal
import unittest

from multiprocessing import Process
from time import sleep

from bsread import source

from cam_server import CamClient
from cam_server.start import start_camera_server
from cam_server.utils import get_host_port_from_stream_address


class CameraClientTest(unittest.TestCase):
    def setUp(self):
        self.host = "0.0.0.0"
        self.port = 8888

        test_base_dir = os.path.split(os.path.abspath(__file__))[0]
        config_folder = os.path.join(test_base_dir, "test_camera_config/")

        self.process = Process(target=start_camera_server, args=(self.host, self.port, config_folder))
        self.process.start()

        # Give it some time to start.
        sleep(0.5)

        server_address = "http://%s:%s" % (self.host, self.port)
        self.client = CamClient(server_address)

    def tearDown(self):
        os.kill(self.process.pid, signal.SIGINT)
        # Wait for the server to die.
        sleep(1)

    def test_client(self):
        server_info = self.client.get_server_info()

        self.assertIsNot(server_info["active_instances"],
                         "There should be no running instances.")

        expected_cameras = ["example_1", "example_2", "example_3", "example_4", "simulation"]

        self.assertListEqual(self.client.get_cameras(), expected_cameras, "Not getting all expected cameras")

        camera_stream_address = self.client.get_camera_stream("simulation")

        self.assertTrue(bool(camera_stream_address), "Camera stream address cannot be empty.")

        self.assertTrue("simulation" in self.client.get_server_info()["active_instances"],
                        "Simulation camera not present in server info.")

        # Check if we can connect to the stream and receive data (in less than 2 seconds).
        host, port = get_host_port_from_stream_address(camera_stream_address)
        with source(host=host, port=port, receive_timeout=2000) as stream:
            data = stream.receive()
            self.assertIsNotNone(data, "Received data was none.")

        # Stop the simulation instance.
        self.client.stop_camera("simulation")

        self.assertTrue("simulation" not in self.client.get_server_info()["active_instances"],
                        "Camera simulation did not stop.")

        self.client.get_camera_stream("simulation")

        self.assertTrue("simulation" in self.client.get_server_info()["active_instances"],
                        "Camera simulation did not start.")

        self.client.stop_all_cameras()

        self.assertTrue("simulation" not in self.client.get_server_info()["active_instances"],
                        "Camera simulation did not stop.")


if __name__ == '__main__':
    unittest.main()
