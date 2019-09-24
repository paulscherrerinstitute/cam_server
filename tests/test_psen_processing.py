import time
import unittest

from cam_server.camera.configuration import CameraConfig
from cam_server.camera.source.simulation import CameraSimulation
from cam_server.pipeline.configuration import PipelineConfig
from cam_server.pipeline.data_processing.psen import process_image


class PsenProcessingTest(unittest.TestCase):
    def test_no_roi(self):
        pipeline_config = PipelineConfig("test_pipeline")

        simulated_camera = CameraSimulation(CameraConfig("simulation"))
        image = simulated_camera.get_image()
        x_axis, y_axis = simulated_camera.get_x_y_axis()
        parameters = pipeline_config.get_configuration()
        camera_name = simulated_camera.get_name()

        result = process_image(image=image,
                               pulse_id=0,
                               timestamp=time.time(),
                               x_axis=x_axis,
                               y_axis=y_axis,
                               parameters=parameters)

        required_fields_in_result = [camera_name + ".processing_parameters"]

        self.assertSetEqual(set(required_fields_in_result), set(result.keys()),
                            "Not all required keys are present in the result")

        # self.assertTrue(camera_name + '.roi_signal_x_profile' not in result)
        # self.assertTrue(camera_name + '.edge_position' not in result)
        # self.assertTrue(camera_name + '.cross_correlation_amplitude' not in result)
        # self.assertTrue(camera_name + '.roi_background_x_profile' not in result)

    def test_background_roi(self):
        pipeline_config = PipelineConfig("test_pipeline")

        simulated_camera = CameraSimulation(CameraConfig("simulation"))
        image = simulated_camera.get_image()
        x_axis, y_axis = simulated_camera.get_x_y_axis()
        parameters = pipeline_config.get_configuration()
        camera_name = simulated_camera.get_name()

        parameters["roi_background"] = [0, 200, 0, 200]

        result = process_image(image=image,
                               pulse_id=0,
                               timestamp=time.time(),
                               x_axis=x_axis,
                               y_axis=y_axis,
                               parameters=parameters)

        required_fields_in_result = [camera_name + ".processing_parameters",
                                     camera_name + '.roi_background_x_profile']

        self.assertSetEqual(set(required_fields_in_result), set(result.keys()),
                            "Not all required keys are present in the result")

        # self.assertTrue(camera_name + '.roi_signal_x_profile' not in result)
        # self.assertTrue(camera_name + '.edge_position' not in result)
        # self.assertTrue(camera_name + '.cross_correlation_amplitude' not in result)

    def test_signal_roi(self):
        pipeline_config = PipelineConfig("test_pipeline")

        simulated_camera = CameraSimulation(CameraConfig("simulation"))
        image = simulated_camera.get_image()
        x_axis, y_axis = simulated_camera.get_x_y_axis()
        parameters = pipeline_config.get_configuration()
        camera_name = simulated_camera.get_name()

        parameters["roi_signal"] = [0, 200, 0, 200]

        result = process_image(image=image,
                               pulse_id=0,
                               timestamp=time.time(),
                               x_axis=x_axis,
                               y_axis=y_axis,
                               parameters=parameters)

        required_fields_in_result = [camera_name + ".processing_parameters",
                                     camera_name + '.roi_signal_x_profile',
                                     # camera_name + '.edge_position',
                                     # camera_name + '.cross_correlation_amplitude'
                                     ]

        self.assertSetEqual(set(required_fields_in_result), set(result.keys()),
                            "Not all required keys are present in the result")

        self.assertTrue(camera_name + '.roi_background_x_profile' not in result)

    def test_both_rois(self):
        pipeline_config = PipelineConfig("test_pipeline")

        simulated_camera = CameraSimulation(CameraConfig("simulation"))
        image = simulated_camera.get_image()
        x_axis, y_axis = simulated_camera.get_x_y_axis()
        parameters = pipeline_config.get_configuration()
        camera_name = simulated_camera.get_name()

        parameters["roi_signal"] = [0, 200, 0, 200]
        parameters["roi_background"] = [0, 200, 0, 200]

        for i in range(10):
            result = process_image(image=image,
                                   pulse_id=i,
                                   timestamp=time.time(),
                                   x_axis=x_axis,
                                   y_axis=y_axis,
                                   parameters=parameters)

        required_fields_in_result = [camera_name + ".processing_parameters",
                                     camera_name + '.roi_signal_x_profile',
                                     # camera_name + '.edge_position',
                                     # camera_name + '.cross_correlation_amplitude',
                                     camera_name + '.roi_background_x_profile'
                                     ]

        self.assertSetEqual(set(required_fields_in_result), set(result.keys()),
                            "Not all required keys are present in the result")

    def test_roi_configuration(self):
        pass


if __name__ == '__main__':
    unittest.main()
