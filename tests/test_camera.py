import imghdr
import unittest
from time import sleep

from PIL import Image
from io import BytesIO

from cam_server import config
from cam_server.pipeline.data_processing.functions import get_png_from_image
from tests.helpers.camera import get_test_instance_manager


class CameraTest(unittest.TestCase):
    simulation_camera = "simulation"

    def test_get_stream(self):

        instance_manager = get_test_instance_manager()

        self.assertListEqual([self.simulation_camera], instance_manager.get_camera_list(),
                             "Missing simulation camera.")

        # Shorten the time to make the tests run faster.
        old_timeout = config.MFLOW_NO_CLIENTS_TIMEOUT
        config.MFLOW_NO_CLIENTS_TIMEOUT = 1

        stream_address = instance_manager.get_camera_stream("simulation")
        stream_address_copy = instance_manager.get_camera_stream("simulation")

        # We should get the same stream both times.
        self.assertEqual(stream_address, stream_address_copy,
                         "Got 2 stream addresses, instead of the same one twice.")

        n_active_instances = len(instance_manager.get_info()["active_instances"])
        self.assertTrue(n_active_instances == 1, "Number of active instances is not correct.")

        # Lets wait for the stream to dies.
        sleep(config.MFLOW_NO_CLIENTS_TIMEOUT + 1.5)

        # The simulation camera should disconnect, since the no client timeout has passed.
        n_active_instances = len(instance_manager.get_info()["active_instances"])
        self.assertTrue(n_active_instances == 0, "All instances should be dead by now.")

        # Restore the old timeout.
        config.MFLOW_NO_CLIENTS_TIMEOUT = old_timeout

    def test_stop_stream(self):
        instance_manager = get_test_instance_manager()

        old_timeout = config.MFLOW_NO_CLIENTS_TIMEOUT
        # We need a high timeout to be sure the stream does not disconnect by itself.
        config.MFLOW_NO_CLIENTS_TIMEOUT = 30

        instance_manager.get_camera_stream(self.simulation_camera)
        self.assertTrue(self.simulation_camera in instance_manager.get_info()["active_instances"],
                        "Simulation camera instance is not running.")

        instance_manager.stop_instance(self.simulation_camera)
        self.assertTrue(self.simulation_camera not in instance_manager.get_info()["active_instances"],
                        "Simulation camera instance should not be running.")

        # Revert back.
        config.MFLOW_NO_CLIENTS_TIMEOUT = old_timeout

    def test_camera_image(self):
        instance_manager = get_test_instance_manager()

        camera = instance_manager.config_manager.load_camera(self.simulation_camera)

        # Retrieve a single image from the camera.
        camera.connect()
        image_raw_bytes = camera.get_image()
        camera.disconnect()

        image = get_png_from_image(image_raw_bytes)

        self.assertEqual("png", imghdr.test_png(image, None), "Image is not a valid PNG.")

        camera_size = camera.get_geometry()
        png_size = Image.open(BytesIO(image)).size

        self.assertEqual(camera_size, png_size, "Camera and image size are not the same.")

if __name__ == '__main__':
    unittest.main()
