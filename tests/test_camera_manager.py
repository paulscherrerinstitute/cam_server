import imghdr
import unittest
from time import sleep

from PIL import Image
from io import BytesIO

from cam_server import config
from cam_server.camera.configuration import CameraConfigManager
from cam_server.camera.management import CameraInstanceManager
from cam_server.pipeline.data_processing.functions import get_png_from_image
from tests.helpers.factory import get_test_instance_manager, MockConfigStorage


class CameraTest(unittest.TestCase):
    simulation_camera = "simulation"

    def setUp(self):
        self.instance_manager = get_test_instance_manager()

    def tearDown(self):
        self.instance_manager.stop_all_instances()
        sleep(1)

    def test_get_stream(self):

        self.assertListEqual([self.simulation_camera], self.instance_manager.get_camera_list(),
                             "Missing simulation camera.")

        # Shorten the time to make the tests run faster.
        old_timeout = config.MFLOW_NO_CLIENTS_TIMEOUT
        config.MFLOW_NO_CLIENTS_TIMEOUT = 1

        stream_address = self.instance_manager.get_camera_stream("simulation")
        stream_address_copy = self.instance_manager.get_camera_stream("simulation")

        # We should get the same stream both times.
        self.assertEqual(stream_address, stream_address_copy,
                         "Got 2 stream addresses, instead of the same one twice.")

        n_active_instances = len(self.instance_manager.get_info()["active_instances"])
        self.assertTrue(n_active_instances == 1, "Number of active instances is not correct.")

        # Lets wait for the stream to dies.
        sleep(config.MFLOW_NO_CLIENTS_TIMEOUT + 1)

        # The simulation camera should disconnect, since the no client timeout has passed.
        n_active_instances = len(self.instance_manager.get_info()["active_instances"])
        self.assertTrue(n_active_instances == 0, "All instances should be dead by now.")

        # Restore the old timeout.
        config.MFLOW_NO_CLIENTS_TIMEOUT = old_timeout

    def test_multiple_stream_requests(self):

        # Shorten the time to make the tests run faster.
        old_timeout = config.MFLOW_NO_CLIENTS_TIMEOUT
        config.MFLOW_NO_CLIENTS_TIMEOUT = 0.5

        stream_address = self.instance_manager.get_camera_stream(self.simulation_camera)

        for _ in range(5):
            # The camera stream address should always be the same.
            self.assertEqual(stream_address, self.instance_manager.get_camera_stream(self.simulation_camera))

        for _ in range(5):
            # The camera stream address should be the same even after restarting the camera instance.
            self.assertEqual(stream_address, self.instance_manager.get_camera_stream(self.simulation_camera))

            sleep(config.MFLOW_NO_CLIENTS_TIMEOUT + 1.5)

            n_active_instances = len(self.instance_manager.get_info()["active_instances"])
            self.assertTrue(n_active_instances == 0, "All instances should be dead by now.")

        # Restore the old timeout.
        config.MFLOW_NO_CLIENTS_TIMEOUT = old_timeout

    def test_stop_stream(self):

        old_timeout = config.MFLOW_NO_CLIENTS_TIMEOUT
        # We need a high timeout to be sure the stream does not disconnect by itself.
        config.MFLOW_NO_CLIENTS_TIMEOUT = 30

        self.instance_manager.get_camera_stream(self.simulation_camera)
        self.assertTrue(self.simulation_camera in self.instance_manager.get_info()["active_instances"],
                        "Simulation camera instance is not running.")

        self.instance_manager.stop_instance(self.simulation_camera)
        self.assertTrue(self.simulation_camera not in self.instance_manager.get_info()["active_instances"],
                        "Simulation camera instance should not be running.")

        # Revert back.
        config.MFLOW_NO_CLIENTS_TIMEOUT = old_timeout

    def test_camera_image(self):

        camera = self.instance_manager.config_manager.load_camera(self.simulation_camera)

        # Retrieve a single image from the camera.
        camera.connect()
        image_raw_bytes = camera.get_image()
        camera.disconnect()

        image = get_png_from_image(image_raw_bytes)

        self.assertEqual("png", imghdr.test_png(image, None), "Image is not a valid PNG.")

        camera_size = camera.get_geometry()
        png_size = Image.open(BytesIO(image)).size

        self.assertEqual(camera_size, png_size, "Camera and image size are not the same.")

    def test_camera_settings_change(self):
        pass

    def test_custom_hostname(self):
        config_manager = CameraConfigManager(config_provider=MockConfigStorage())
        camera_instance_manager = CameraInstanceManager(config_manager, hostname="custom_cam_hostname")

        stream_address = camera_instance_manager.get_camera_stream("simulation")
        self.assertTrue(stream_address.startswith("tcp://custom_cam_hostname"))

        camera_instance_manager.stop_all_instances()

if __name__ == '__main__':
    unittest.main()
