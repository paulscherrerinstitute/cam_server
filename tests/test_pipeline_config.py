import unittest
from shutil import rmtree

from time import sleep

import numpy
import os

from copy import deepcopy

from cam_server.pipeline.configuration import BackgroundImageManager, PipelineConfig, UserScriptsManager
from cam_server.utils import update_pipeline_config
from tests import require_folder
from tests.helpers.factory import get_test_pipeline_manager


class PipelineConfigTest(unittest.TestCase):

    def tearDown(self):
        if os.path.isdir("user_scripts/"):
            rmtree("user_scripts/")

    def test_get_set_delete_pipeline_config(self):
        instance_manager = get_test_pipeline_manager()

        pipelines = instance_manager.config_manager.get_pipeline_list()
        self.assertIsNot(pipelines, "Pipelines should be empty at the beginning.")

        example_pipeline_config = {
            "camera_name": "simulation"
        }

        instance_manager.config_manager.save_pipeline_config("simulation_pipeline", example_pipeline_config)
        self.assertListEqual(["simulation_pipeline"], instance_manager.config_manager.get_pipeline_list(),
                             "Pipeline not added.")

        expected_config = deepcopy(PipelineConfig.DEFAULT_CONFIGURATION)
        expected_config.update(example_pipeline_config)

        self.assertDictEqual(expected_config,
                             instance_manager.config_manager.get_pipeline_config("simulation_pipeline"),
                             "Saved and retrieved pipeline configs are not the same.")

        instance_manager.config_manager.delete_pipeline_config("simulation_pipeline")
        self.assertIsNot(instance_manager.config_manager.get_pipeline_list(),
                         "Pipeline config should be empty.")

        with self.assertRaisesRegex(ValueError, "Pipeline 'non_existing_pipeline' does not exist."):
            instance_manager.config_manager.delete_pipeline_config("non_existing_pipeline")

    def test_load_pipeline(self):
        instance_manager = get_test_pipeline_manager()

        example_pipeline_config = {
            "camera_name": "simulation"
        }

        instance_manager.config_manager.save_pipeline_config("pipeline_simulation", example_pipeline_config)

        expected_config = deepcopy(PipelineConfig.DEFAULT_CONFIGURATION)
        expected_config.update(example_pipeline_config)

        pipeline = instance_manager.config_manager.load_pipeline("pipeline_simulation")
        self.assertDictEqual(pipeline.get_configuration(), expected_config,
                             "Saved and loaded pipelines are not the same.")

    def test_invalid_config(self):
        instance_manager = get_test_pipeline_manager()

        invalid_pipeline_config = {
            # Wrong attribute name - should be "camera_name".
            "camera": "simulation"
        }

        with self.assertRaisesRegex(ValueError, "Camera name not specified in configuration."):
            instance_manager.config_manager.save_pipeline_config("invalid_pipeline", invalid_pipeline_config)

    def test_background_provider(self):
        background_manager = BackgroundImageManager("background_config/")

        shape = (960, 1280)
        image = numpy.zeros(shape=shape, dtype="f8")

        return_name = background_manager.save_background("test_background", image, append_timestamp=False)
        expected_file = "background_config/test_background.npy"

        self.assertEqual(return_name, "test_background", "Saved and returned backgrounds are not the same.")

        self.assertTrue(os.path.exists(expected_file),
                        "The background is not in the expected location.")

        loaded_image = background_manager.get_background("test_background")

        self.assertTrue(numpy.array_equal(image, loaded_image), "Loaded background not same as saved.")

        os.remove(expected_file)

        return_name = background_manager.save_background("test_background", image, append_timestamp=True)
        self.assertNotEqual(return_name, "test_background", "The names should not be equal.")
        expected_file = "background_config/" + return_name + ".npy"

        os.remove(expected_file)

    def test_get_latest_background(self):
        background_manager = BackgroundImageManager("background_config/")

        shape = (960, 1280)
        image = numpy.zeros(shape=shape, dtype="f8")

        return_name_1 = background_manager.save_background("test_background", image)
        sleep(0.1)
        return_name_2 = background_manager.save_background("test_background", image)
        sleep(0.1)
        return_name_3 = background_manager.save_background("test_background", image)

        self.assertNotEqual(return_name_1, return_name_2)
        self.assertNotEqual(return_name_2, return_name_3)

        return_name_latest = background_manager.get_latest_background_id("test_background")

        self.assertEqual(return_name_3, return_name_latest, "Return background id is not correct.")

        with self.assertRaisesRegex(ValueError, "No background matches for the specified prefix 'not_here'."):
            background_manager.get_latest_background_id("not_here")

    def test_default_config(self):
        configuration = {
            "camera_name": "simulation",
            "image_background": "test_background",
            "image_good_region": {

            }
        }

        configuration = PipelineConfig("test", configuration)
        complete_config = configuration.get_configuration()

        self.assertIsNone(complete_config["image_slices"], "This section should be None.")

        self.assertSetEqual(set(complete_config["image_good_region"].keys()),
                            set(["threshold", "gfscale"]))

        configuration = {
            "camera_name": "simulation",
            "image_good_region": {},
            "image_slices": {}
        }

        configuration = PipelineConfig("test", configuration)
        complete_config = configuration.get_configuration()

        self.assertSetEqual(set(complete_config["image_good_region"].keys()),
                            set(["threshold", "gfscale"]),
                            "Missing keys in camera calibration.")

        self.assertSetEqual(set(complete_config["image_slices"].keys()),
                            set(["number_of_slices", "scale", "orientation"]),
                            "Missing keys in camera calibration.")

    def test_update_pipeline_config(self):
        configuration = {
            "camera_name": "simulation"
        }

        old_config = PipelineConfig("test", configuration).get_configuration()

        config_updates = {"image_good_region": {}}
        updated_config = update_pipeline_config(old_config, config_updates)

        self.assertIsNotNone(updated_config["image_good_region"])

        configuration = {
            "camera_name": "simulation",
            "image_slices": {
                "number_of_slices": 98
            }
        }

        old_config = PipelineConfig("test", configuration).get_configuration()

        config_updates = {
            "image_slices": {
                "scale": 99
            }
        }

        updated_config = update_pipeline_config(old_config, config_updates)
        self.assertEqual(updated_config["image_slices"]["scale"], 99, "New value not updated.")
        self.assertEqual(updated_config["image_slices"]["number_of_slices"], 98, "Old value changed.")

        configuration = {
            "camera_name": "name 1",
            "image_background": "test",
            "image_threshold": 1,
            "image_region_of_interest": [0, 1, 2, 3],
            "image_good_region": {
                "threshold": 0.3,
                "gfscale": 1.8
            },
            "image_slices": {
                "number_of_slices": 1,
                "scale": 2
            }
        }

        old_config = PipelineConfig("test", configuration).get_configuration()

        config_updates = {
            "camera_name": "name 2",
            "image_background_enable": True,
            "image_background": "test_2",
            "image_threshold": 2,
            "image_region_of_interest": [3, 4, 5, 6],
            "image_good_region": {
                "threshold": 0.9,
                "gfscale": 3.6
            },
            "image_slices": {
                "number_of_slices": 6,
                "scale": 7
            },
            "pipeline_type": "processing"
        }

        updated_config = update_pipeline_config(old_config, config_updates)

        self.assertDictEqual(updated_config["image_good_region"], config_updates["image_good_region"],
                             "Everything was not updated properly.")

        self.assertDictEqual(updated_config["image_slices"], config_updates["image_slices"],
                             "Everything was not updated properly.")

        self.assertDictEqual(updated_config, config_updates,
                             "Everything was not updated properly.")

        # Try removing element by element from the config - negative update.

        updated_config = update_pipeline_config(updated_config, {"camera_name": None})
        self.assertEqual(updated_config["camera_name"], None)

        updated_config = update_pipeline_config(updated_config, {"image_background": None})
        self.assertEqual(updated_config["image_background"], None)

        updated_config = update_pipeline_config(updated_config, {"image_region_of_interest": None})
        self.assertEqual(updated_config["image_region_of_interest"], None)

        updated_config = update_pipeline_config(updated_config, {"image_good_region": {"threshold": None}})
        self.assertEqual(updated_config["image_good_region"]["threshold"], None)
        self.assertEqual(updated_config["image_good_region"]["gfscale"], 3.6)

        updated_config = update_pipeline_config(updated_config, {"image_good_region": None})
        self.assertEqual(updated_config["image_good_region"], None)

        updated_config = update_pipeline_config(updated_config, {"image_slices": {"number_of_slices": None}})
        self.assertEqual(updated_config["image_slices"]["number_of_slices"], None)
        self.assertEqual(updated_config["image_slices"]["scale"], 7)

        updated_config = update_pipeline_config(updated_config, {"image_slices": None})
        self.assertEqual(updated_config["image_slices"], None)

        updated_config = update_pipeline_config(updated_config, {"image_threshold": None})
        self.assertEqual(updated_config["image_threshold"], None)

        updated_config = update_pipeline_config(updated_config, {"image_background_enable": None})
        self.assertEqual(updated_config["image_background_enable"], None)

        updated_config = update_pipeline_config(updated_config, {"pipeline_type": None})
        self.assertEqual(updated_config["pipeline_type"], None)

        self.assertTrue(all(value is None for value in updated_config.values()))

    def test_expand_pipeline_config(self):
        configuration = {"camera_name": "simulation",
                         "image_slices": {
                             "number_of_slices": None,
                             "scale": 7}
                         }

        expanded_configuration = PipelineConfig.expand_config(configuration)
        self.assertEqual(expanded_configuration["image_slices"]["number_of_slices"],
                         PipelineConfig.DEFAULT_IMAGE_SLICES["number_of_slices"],
                         "Default not applied.")

        self.assertEqual(expanded_configuration["pipeline_type"],
                         PipelineConfig.DEFAULT_CONFIGURATION["pipeline_type"],
                         "Default not applied.")

        configuration = {"camera_name": "simulation",
                         "image_slices": None}

        expanded_configuration = PipelineConfig.expand_config(configuration)

        self.assertIsNone(expanded_configuration["image_slices"],
                          "It should still be None.")

        configuration = {"camera_name": "simulation",
                         "image_slices": {}}

        expanded_configuration = PipelineConfig.expand_config(configuration)

        self.assertDictEqual(expanded_configuration["image_slices"], PipelineConfig.DEFAULT_IMAGE_SLICES,
                             "Image slices defaults not applied.")

        configuration = {"camera_name": "simulation",
                         "image_background_enable": None}

        expanded_configuration = PipelineConfig.expand_config(configuration)

        self.assertEqual(expanded_configuration["image_background_enable"],
                         PipelineConfig.DEFAULT_CONFIGURATION["image_background_enable"],
                         "Default value not applied. It should not be None.")

    def test_invalid_number_of_slices(self):
        configuration = {"camera_name": "simulation",
                         "image_slices": {
                             "number_of_slices": 1.2}
                         }

        expanded_configuration = PipelineConfig.expand_config(configuration)
        with self.assertRaisesRegex(ValueError, "number_of_slices must be an integer"):
            PipelineConfig.validate_pipeline_config(expanded_configuration)

    def test_invalid_pipeline_type(self):
        configuration = {"camera_name": "simulation"}
        expanded_configuration = PipelineConfig.expand_config(configuration)
        PipelineConfig.validate_pipeline_config(expanded_configuration)

        expanded_configuration["pipeline_type"] = "invalid"

        with self.assertRaisesRegex(ValueError, "pipeline_type 'invalid' not present in mapping. Available:"):
            PipelineConfig.validate_pipeline_config(expanded_configuration)

    def test_scripts_manager(self):
        require_folder("user_scripts/")

        scripts_manager = UserScriptsManager("user_scripts/")

        script_name = "Test.py"
        script_content = "print('Hello world')"
        scripts_manager.save_script(script_name, script_content )
        try:
            scripts = scripts_manager.get_scripts()
            self.assertIn(script_name, scripts)

            ret  = scripts_manager.get_script(script_name)
            self.assertEqual(ret, script_content)
        finally:
            os.remove(os.path.join(scripts_manager.scripts_folder, "Test.py"),)


if __name__ == '__main__':
    unittest.main()
