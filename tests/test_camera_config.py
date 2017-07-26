import unittest

from cam_server.camera.receiver import CameraSimulation
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

        with self.assertRaisesRegex(ValueError, "Cannot save config for simulation camera."):
            instance_manager.config_manager.save_camera_config("simulation", {})

        with self.assertRaisesRegex(ValueError, "Cannot delete simulation camera."):
            instance_manager.config_manager.delete_camera_config("simulation")

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

        instance_manager.config_manager.delete_camera_config("test_camera_2")

        self.assertListEqual(sorted(instance_manager.config_manager.get_camera_list()),
                             sorted(["simulation", "test_camera_1"]),
                             "Test camera was not deleted successfully.")

        # You should never be able to overwrite the simulation camera.
        with self.assertRaisesRegex(ValueError, "Cannot save config for simulation camera."):
            instance_manager.config_manager.save_camera_config("simulation", camera_config)

    def test_load_camera(self):
        expected_config_example_1 = {
            "name": "example_1",
            "prefix": "EPICS_example_1",
            "mirror_x": False,
            "mirror_y": True,
            "rotate": 1
        }

        instance_manager = get_test_instance_manager()
        instance_manager.config_manager.save_camera_config("example_1", expected_config_example_1)

        camera_config = instance_manager.config_manager.get_camera_config("example_1")

        self.assertDictEqual(camera_config.to_dict(), expected_config_example_1,
                             "CameraConfig not as expected")

        camera = instance_manager.config_manager.load_camera("example_1")

        self.assertDictEqual(camera.camera_config.to_dict(), expected_config_example_1,
                             "Camera not as expected")

        simulated_camera = instance_manager.config_manager.load_camera("simulation")

        self.assertTrue(isinstance(simulated_camera, CameraSimulation),
                        "The 'simulation' configuration did not return the camera simulation.")


if __name__ == '__main__':
    unittest.main()
