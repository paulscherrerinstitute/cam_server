import unittest
from time import sleep

from cam_server import config
from cam_server.camera.configuration import CameraConfigManager
from cam_server.camera.management import CameraInstanceManager
from tests.helpers.configuration import MockConfigStorage


class CameraTest(unittest.TestCase):

    def test_get_stream(self):
        simulation_camera = "simulation"

        config_manager = CameraConfigManager(config_provider=MockConfigStorage())
        camera_instance_manager = CameraInstanceManager(config_manager)

        self.assertListEqual([simulation_camera], camera_instance_manager.get_camera_list(),
                             "Missing simulation camera.")

        # Shorten the time to make the tests run faster.
        old_timeout = config.MFLOW_NO_CLIENTS_TIMEOUT
        config.MFLOW_NO_CLIENTS_TIMEOUT = 1

        stream_address = camera_instance_manager.get_camera_stream("simulation")
        stream_address_copy = camera_instance_manager.get_camera_stream("simulation")

        self.assertEqual(stream_address, stream_address_copy,
                         "Got 2 stream addresses, instead of the same one twice.")

        sleep(config.MFLOW_NO_CLIENTS_TIMEOUT + 0.5)

        # Restore the old timeout.
        config.MFLOW_NO_CLIENTS_TIMEOUT = old_timeout


if __name__ == '__main__':
    unittest.main()
