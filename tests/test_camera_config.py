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

    def test_set_get_camera_config(self):
        instance_manager = get_test_instance_manager()

        # Verify that the config does not exist yet.
        with self.assertRaisesRegex(ValueError, "Unable to load config"):
            instance_manager.config_manager.get_camera_config("test_camera_1")

        # Check if default values work as expected.
        instance_manager.config_manager.save_camera_config("test_camera_1", {})

        camera = instance_manager.config_manager.load_camera("test_camera_1")
        self.assertIsNotNone(camera, "Retrieved camera config works.")

        camera_config = {"prefix": "EPICS_PREFIX",
                         "mirror_x": True,
                         "mirror_y": True,
                         "rotate": 3}

        # Overwrite existing and create new config with same values. The result should be the same.
        instance_manager.config_manager.save_camera_config("test_camera_1", camera_config)
        instance_manager.config_manager.save_camera_config("test_camera_2", camera_config)

        camera_config_1 = instance_manager.config_manager.get_camera_config("test_camera_1")
        camera_config_2 = instance_manager.config_manager.get_camera_config("test_camera_2")

        self.assertDictEqual(camera_config_1.parameters, camera_config_2.parameters, "Configs should be identical.")

        # You should never be able to overwrite the simulation camera.
        with self.assertRaisesRegex(ValueError, "Cannot save config for simulation camera."):
            instance_manager.config_manager.save_camera_config("simulation", camera_config)

if __name__ == '__main__':
    unittest.main()
