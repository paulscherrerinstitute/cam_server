import unittest

from tests.helpers.camera import get_test_instance_manager


class CameraConfigTest(unittest.TestCase):

    def test_get_geometry(self):
        width, height = get_test_instance_manager().config_manager.get_camera_geometry("simulation")
        self.assertTrue(width > 0 and height > 0, "Width and height need to be positive numbers.")

    def test_get_camera_config(self):
        # TODO: Test the config.
        pass

    def test_set_camera_config(self):
        # TODO: Test the config.
        pass
