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

        # We should get the same stream both times.
        self.assertEqual(stream_address, stream_address_copy,
                         "Got 2 stream addresses, instead of the same one twice.")

        n_active_instances = len(camera_instance_manager.get_info()["active_instances"])
        self.assertTrue(n_active_instances == 1, "Number of active instances is not correct.")

        # Lets wait for the stream to dies.
        sleep(config.MFLOW_NO_CLIENTS_TIMEOUT + 1.5)

        # The simulation camera should disconnect, since the no client timeout has passed.
        n_active_instances = len(camera_instance_manager.get_info()["active_instances"])
        self.assertTrue(n_active_instances == 0, "All instances should be dead by now.")

        # Restore the old timeout.
        config.MFLOW_NO_CLIENTS_TIMEOUT = old_timeout


if __name__ == '__main__':
    unittest.main()
