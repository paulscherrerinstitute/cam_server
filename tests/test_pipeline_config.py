import unittest

import numpy
import os
from cam_server.pipeline.configuration import BackgroundImageManager
from tests.helpers.factory import get_test_pipeline_manager


class PipelineConfigTest(unittest.TestCase):

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

        self.assertDictEqual(example_pipeline_config,
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

        pipeline = instance_manager.config_manager.load_pipeline("pipeline_simulation")
        self.assertDictEqual(pipeline.get_parameters(), example_pipeline_config,
                             "Saved and loaded pipelines are not the same.")

    def test_invalid_config(self):
        instance_manager = get_test_pipeline_manager()

        invalid_pipeline_config = {
            # Wrong attribute name - should be "camera_name".
            "camera": "simulation"
        }

        with self.assertRaisesRegex(ValueError, "The following mandatory attributes were not "):
            instance_manager.config_manager.save_pipeline_config("invalid_pipeline", invalid_pipeline_config)

    def test_background_provider(self):
        background_manager = BackgroundImageManager("background_config/")

        shape = (960, 1280)
        image = numpy.zeros(shape=shape, dtype="f8")

        background_manager.save_background("test_background", image)
        expected_file = "background_config/test_background.npy"

        self.assertTrue(os.path.exists(expected_file),
                        "The background is not in the expected location.")

        loaded_image = background_manager.get_background("test_background")

        self.assertTrue(numpy.array_equal(image, loaded_image), "Loaded background not same as saved.")

        os.remove(expected_file)


if __name__ == '__main__':
    unittest.main()
