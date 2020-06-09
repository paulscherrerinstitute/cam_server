import unittest

from cam_server.camera.configuration import CameraConfig
from cam_server.camera.source.simulation import CameraSimulation
from cam_server.utils import update_camera_config
from tests.helpers.factory import get_test_instance_manager


class CameraConfigTest(unittest.TestCase):
    def test_set_get_camera_config(self):
        instance_manager = get_test_instance_manager()

        # Verify that the config does not exist yet.
        with self.assertRaisesRegex(ValueError, "Config 'test_camera_1' does not exist."):
            instance_manager.config_manager.get_camera_config("test_camera_1")

        # Check if default values work as expected.
        with self.assertRaisesRegex(ValueError, "Config object cannot be empty. Config: {}"):
            instance_manager.config_manager.save_camera_config("test_camera_1", {})

        with self.assertRaisesRegex(ValueError, "not specified in configuration"):
            instance_manager.config_manager.save_camera_config("test_camera_1", {"test": "test"})

        instance_manager.config_manager.save_camera_config("test_camera_1", {"source": "test",
                                                                             "source_type": "epics",
                                                                             "mirror_x": False,
                                                                             "mirror_y": False,
                                                                             "rotate": 0,
                                                                             "camera_calibration": None})
        camera = instance_manager.config_manager.load_camera("test_camera_1")
        self.assertIsNotNone(camera, "Retrieved camera config works.")

        camera_config = {"source": "EPICS_PREFIX",
                         "source_type": "epics",
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

        self.assertListEqual(instance_manager.config_manager.get_camera_list(), ["test_camera_1"],
                             "Test camera was not deleted successfully.")

    def test_load_camera(self):
        expected_config_example_1 = {
            "name": "camera_example_1",
            "source": "EPICS_example_1",
            "source_type": "epics",
            "mirror_x": False,
            "mirror_y": True,
            "rotate": 1,
            "roi": None,
            "image_background":None,
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

        simulation_config = {
            "name": "simulation",
            "source": "",
            "source_type": "simulation",
            "mirror_x": False,
            "mirror_y": False,
            "rotate": 0,
            "camera_calibration": None
        }

        instance_manager.config_manager.save_camera_config("simulation", simulation_config)

        simulated_camera = instance_manager.config_manager.load_camera("simulation")

        self.assertTrue(isinstance(simulated_camera, CameraSimulation),
                        "The 'simulation' configuration did not return the camera simulation.")

    def test_delete_camera_config(self):
        instance_manager = get_test_instance_manager()

        with self.assertRaisesRegex(ValueError, "Config 'example_test' does not exist."):
            instance_manager.config_manager.delete_camera_config("example_test")

        example_test = {
            "name": "camera_example_1",
            "source": "EPICS_example_1",
            "source_type": "epics",
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
            "source": "simulation"
        }

        configuration = CameraConfig("simulation", configuration)
        complete_config = configuration.get_configuration()

        self.assertIsNone(complete_config["camera_calibration"])
        self.assertFalse(complete_config["mirror_x"])
        self.assertFalse(complete_config["mirror_y"])
        self.assertEqual(complete_config["rotate"], 0)
        self.assertEqual(complete_config["source_type"], "epics")

        configuration = {
            "source": "simulation",
            "camera_calibration": {}
        }

        configuration = CameraConfig("simulation", configuration)
        complete_config = configuration.get_configuration()

        self.assertSetEqual(set(complete_config["camera_calibration"].keys()),
                            set(CameraConfig.DEFAULT_CAMERA_CALIBRATION.keys()),
                            "Missing keys in camera calibration.")

    def test_update_camera_config(self):
        config = {
            "source": "simulation",
            "source_type": "simulation",
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

    def test_invalid_source_type(self):
        configuration = {
            "source": "simulation",
            "source_type": "invalid"
        }

        with self.assertRaisesRegex(ValueError, "Invalid source_type "):
            configuration = CameraConfig("simulation", configuration)

    def test_save_simulation_new_frame_rate(self):
        instance_manager = get_test_instance_manager()

        configuration = {
            "source": "test_with_frame_rate",
            "source_type": "simulation",
            "frame_rate": 1,
            'camera_calibration': None,
            'mirror_x': False,
            'rotate': 0,
            'mirror_y': False
        }

        instance_manager.config_manager.save_camera_config("test_with_frame_rate", configuration)

        camera = instance_manager.config_manager.load_camera("test_with_frame_rate")

        self.assertEqual(camera.frame_rate, configuration["frame_rate"])

        instance_manager.config_manager.delete_camera_config("test_with_frame_rate")


if __name__ == '__main__':
    unittest.main()
