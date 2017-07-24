import unittest

from tests.helpers.camera import get_test_instance_manager


class CameraConfigTest(unittest.TestCase):

    def test_simulation_camera(self):
        instance_manager = get_test_instance_manager()

        cameras = instance_manager.get_camera_list()
        self.assertListEqual(cameras, ["simulation"], "Simulation camera missing in config.")

        width, height = instance_manager.config_manager.get_camera_geometry("simulation")
        self.assertTrue(width > 0 and height > 0, "Width and height need to be positive numbers.")

        configuration = instance_manager.config_manager.get_camera_config("simulation")
        self.assertIsNotNone(configuration, "Configuration for the simulated camera should exist.")

    def test_get_camera_config(self):
        pass

    def test_set_camera_config(self):
        # TODO: Test the config.
        pass
