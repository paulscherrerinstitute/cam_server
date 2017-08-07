import json
import unittest

import time

import numpy
from cam_server.camera.receiver import CameraSimulation
from cam_server.pipeline.configuration import PipelineConfig
from cam_server.pipeline.data_processing.processor import process_image
from tests.helpers.factory import MockBackgroundManager


class PipelineProcessingTest(unittest.TestCase):
    def test_noop_pipeline(self):
        pipeline_config = PipelineConfig("test_pipeline")

        simulated_camera = CameraSimulation()
        image = simulated_camera.get_image()
        x_axis, y_axis = simulated_camera.get_x_y_axis()
        parameters = pipeline_config.get_configuration()

        result = process_image(image=image,
                               timestamp=time.time(),
                               x_axis=x_axis,
                               y_axis=y_axis,
                               parameters=parameters)

        required_fields_in_result = ['x_center_of_mass', 'x_axis', 'y_axis', 'x_profile', 'y_fit_standard_deviation',
                                     'y_rms', 'timestamp', 'y_profile', 'image', 'max_value', 'x_fit_offset',
                                     'x_fit_gauss_function', 'y_center_of_mass', 'min_value', 'y_fit_mean',
                                     'x_fit_mean', 'x_rms', 'y_fit_amplitude', 'x_fit_amplitude',
                                     'y_fit_gauss_function', 'x_fit_standard_deviation', 'y_fit_offset',
                                     "processing_parameters"]

        self.assertSetEqual(set(required_fields_in_result), set(result.keys()),
                            "Not all required keys are present in the result")

        self.assertTrue(numpy.array_equal(result["image"], image),
                        "The input and output image are not the same, but the pipeline should not change it.")

        self.assertDictEqual(parameters, json.loads(result["processing_parameters"]),
                             "The passed and the received processing parameters are not the same.")

    def test_image_background(self):

        pipeline_parameters = {
            "camera_name": "simulation",
            "image_background": "white_background"
        }

        simulated_camera = CameraSimulation()
        image = simulated_camera.get_image()
        x_axis, y_axis = simulated_camera.get_x_y_axis()

        background_provider = MockBackgroundManager()
        x_size, y_size = simulated_camera.get_geometry()
        background_provider.save_background("white_background", numpy.zeros(shape=(y_size, x_size)))

        pipeline_config = PipelineConfig("test_pipeline", pipeline_parameters)
        parameters = pipeline_config.get_configuration()
        image_background_array = background_provider.get_background(parameters.get("image_background"))

        result = process_image(image=image,
                               timestamp=time.time(),
                               x_axis=x_axis,
                               y_axis=y_axis,
                               parameters=parameters,
                               image_background_array=image_background_array)

        self.assertTrue(numpy.array_equal(result["image"], image),
                        "A zero background should not change the image.")

        max_value_in_image = result["max_value"]

        pipeline_parameters = {
            "camera_name": "simulation",
            "image_background": "max_background"
        }

        max_background = numpy.zeros(shape=(y_size, x_size))
        max_background.fill(max_value_in_image)
        background_provider.save_background("max_background", max_background)

        pipeline_config = PipelineConfig("test_pipeline", pipeline_parameters)
        parameters = pipeline_config.get_configuration()
        image_background_array = background_provider.get_background(parameters.get("image_background"))

        expected_image = numpy.zeros(shape=(y_size, x_size))

        result = process_image(image=image,
                               timestamp=time.time(),
                               x_axis=x_axis,
                               y_axis=y_axis,
                               parameters=parameters,
                               image_background_array=image_background_array)

        self.assertTrue(numpy.array_equal(result["image"], expected_image),
                        "The image should be all zeros - negative numbers are not allowed.")

    def test_image_threshold(self):
        simulated_camera = CameraSimulation()
        image = simulated_camera.get_image()
        x_axis, y_axis = simulated_camera.get_x_y_axis()
        x_size, y_size = simulated_camera.get_geometry()

        pipeline_parameters = {
            "camera_name": "simulation",
            "image_threshold": 9999999
        }

        pipeline_config = PipelineConfig("test_pipeline", pipeline_parameters)
        parameters = pipeline_config.get_configuration()

        result = process_image(image=image,
                               timestamp=time.time(),
                               x_axis=x_axis,
                               y_axis=y_axis,
                               parameters=parameters)

        expected_image = numpy.zeros(shape=(y_size, x_size))
        self.assertTrue(numpy.array_equal(result["image"], expected_image),
                        "An image of zeros should have been produced.")

        pipeline_parameters = {
            "camera_name": "simulation",
            "image_threshold": 0
        }

        pipeline_config = PipelineConfig("test_pipeline", pipeline_parameters)
        parameters = pipeline_config.get_configuration()

        result = process_image(image=image,
                               timestamp=time.time(),
                               x_axis=x_axis,
                               y_axis=y_axis,
                               parameters=parameters)

        self.assertTrue(numpy.array_equal(result["image"], image),
                        "The image should be the same as the original image.")

    def test_region_of_interest(self):
        # TODO: Write tests.
        pass

    def test_good_region(self):
        # TODO: Write tests.
        pass

    def test_slices(self):
        # TODO: Write tests.
        pass


if __name__ == '__main__':
    unittest.main()
