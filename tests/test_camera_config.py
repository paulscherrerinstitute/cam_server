import unittest

from cam_server.camera.configuration import CameraConfig
from cam_server.camera.receiver import CameraSimulation
from cam_server.utils import update_camera_config
from tests.helpers.factory import get_test_instance_manager


class CameraConfigTest(unittest.TestCase):
    def test_simulation_camera(self):
        instance_manager = get_test_instance_manager()

        cameras = instance_manager.get_camera_list()
        self.assertListEqual(cameras, ["simulation"], "Simulation camera missing in config.")

        width, height = instance_manager.config_manager.get_camera_geometry("simulation")
        self.assertTrue(width > 0 and height > 0, "Width and height need to be positive numbers.")

        configuration = instance_manager.config_manager.get_camera_config("simulation")
        self.assertIsNotNone(configuration, "Configuration for the simulated camera should exist.")

        with self.assertRaisesRegex(ValueError, "Cannot delete simulation camera."):
            instance_manager.config_manager.delete_camera_config("simulation")

    def test_set_get_camera_config(self):
        instance_manager = get_test_instance_manager()

        # Verify that the config does not exist yet.
        with self.assertRaisesRegex(ValueError, "Config 'test_camera_1' does not exist."):
            instance_manager.config_manager.get_camera_config("test_camera_1")

        # Check if default values work as expected.
        with self.assertRaisesRegex(ValueError, "Config object cannot be empty. Config: {}"):
            instance_manager.config_manager.save_camera_config("test_camera_1", {})

        with self.assertRaisesRegex(ValueError, "The following mandatory attributes were not found in the "):
            instance_manager.config_manager.save_camera_config("test_camera_1", {"prefix": "test"})

        instance_manager.config_manager.save_camera_config("test_camera_1", {"prefix": "test",
                                                                             "mirror_x": False,
                                                                             "mirror_y": False,
                                                                             "rotate": 0,
                                                                             "camera_calibration": None})
        camera = instance_manager.config_manager.load_camera("test_camera_1")
        self.assertIsNotNone(camera, "Retrieved camera config works.")

        camera_config = {"prefix": "EPICS_PREFIX",
                         "mirror_x": True,
                         "mirror_y": True,
                         "rotate": 3,
                         "camera_calibration": None}

        # Overwrite existing and create new config with same values. The result should be the same.
        instance_manager.config_manager.save_camera_config("test_camera_1", camera_config)
        instance_manager.config_manager.save_camera_config("test_camera_2", camera_config)

        camera_config_1 = instance_manager.config_manager.get_camera_config("test_camera_1")
        camera_config_2 = instance_manager.config_manager.get_camera_config("test_camera_2")

        self.assertDictEqual(camera_config_1.parameters, camera_config_2.parameters, "Configs should be identical.")

        self.assertIsNone(camera_config_1.parameters["camera_calibration"])
        self.assertIsNone(camera_config_2.parameters["camera_calibration"])

        instance_manager.config_manager.delete_camera_config("test_camera_2")

        self.assertListEqual(sorted(instance_manager.config_manager.get_camera_list()),
                             sorted(["simulation", "test_camera_1"]),
                             "Test camera was not deleted successfully.")

        # Simulation camera cannot be saved - config should not match.
        simulation_config_before = instance_manager.config_manager.get_camera_config("simulation").get_configuration()
        instance_manager.config_manager.save_camera_config("simulation", camera_config)
        simulation_config_after = instance_manager.config_manager.get_camera_config("simulation").get_configuration()

        self.assertDictEqual(simulation_config_before, simulation_config_after)

    def test_load_camera(self):
        expected_config_example_1 = {
            "name": "camera_example_1",
            "prefix": "EPICS_example_1",
            "mirror_x": False,
            "mirror_y": True,
            "rotate": 1,
            "camera_calibration": None
        }

        instance_manager = get_test_instance_manager()
        instance_manager.config_manager.save_camera_config("example_1", expected_config_example_1)

        camera_config = instance_manager.config_manager.get_camera_config("example_1")

        self.assertDictEqual(camera_config.get_configuration(), expected_config_example_1,
                             "CameraConfig not as expected")

        camera = instance_manager.config_manager.load_camera("example_1")

        self.assertDictEqual(camera.camera_config.get_configuration(), expected_config_example_1,
                             "Camera not as expected")

        simulated_camera = instance_manager.config_manager.load_camera("simulation")

        self.assertTrue(isinstance(simulated_camera, CameraSimulation),
                        "The 'simulation' configuration did not return the camera simulation.")

    def test_delete_camera_config(self):
        instance_manager = get_test_instance_manager()

        with self.assertRaisesRegex(ValueError, "Config 'example_test' does not exist."):
            instance_manager.config_manager.delete_camera_config("example_test")

        example_test = {
            "name": "camera_example_1",
            "prefix": "EPICS_example_1",
            "mirror_x": False,
            "mirror_y": True,
            "rotate": 1,
            "camera_calibration": None
        }

        instance_manager.config_manager.save_camera_config("different_name", example_test)
        instance_manager.config_manager.get_camera_config("different_name")
        instance_manager.config_manager.delete_camera_config("different_name")

        with self.assertRaisesRegex(ValueError, "Config 'different_name' does not exist."):
            instance_manager.config_manager.get_camera_config("different_name")

    def test_default_config(self):
        configuration = {
            "prefix": "simulation"
        }

        configuration = CameraConfig("simulation", configuration)
        complete_config = configuration.get_configuration()

        self.assertIsNone(complete_config["camera_calibration"])
        self.assertFalse(complete_config["mirror_x"])
        self.assertFalse(complete_config["mirror_y"])
        self.assertEqual(complete_config["rotate"], 0)

        configuration = {
            "prefix": "simulation",
            "camera_calibration": {}
        }

        configuration = CameraConfig("simulation", configuration)
        complete_config = configuration.get_configuration()

        self.assertSetEqual(set(complete_config["camera_calibration"].keys()),
                            set(CameraConfig.DEFAULT_CAMERA_CALIBRATION.keys()),
                            "Missing keys in camera calibration.")

    def test_update_camera_config(self):
        config = {
            "prefix": "simulation",
            "camera_calibration": {
                "reference_marker": [0, 0, 100, 100],
                "reference_marker_width": 100.0,
                "reference_marker_height": 100.0,
                "angle_horizontal": 0.0,
                "angle_vertical": 0.0
            }
        }

        updated_config = update_camera_config(config, {"camera_calibration": {"angle_horizontal": 10}})

        self.assertEqual(updated_config["camera_calibration"]["angle_horizontal"], 10)

        updated_config = update_camera_config(updated_config, {"camera_calibration": None})
        self.assertIsNone(updated_config["camera_calibration"])


if __name__ == '__main__':
    unittest.main()
