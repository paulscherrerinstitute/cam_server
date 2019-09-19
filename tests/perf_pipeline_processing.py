import time
import unittest

import numpy

from cam_server.camera.configuration import CameraConfig
from cam_server.camera.source.simulation import CameraSimulation
from cam_server.pipeline.data_processing import functions
from cam_server.pipeline.data_processing.processor import process_image


class PipelinePerformanceTest(unittest.TestCase):

    def test_process_image_performance(self):
        # Profile only if LineProfiler present.
        # To install: conda install line_profiler
        try:
            from line_profiler import LineProfiler
        except ImportError:
            return

        simulated_camera = CameraSimulation(CameraConfig("simulation"), size_x=2048, size_y=2048)
        x_axis, y_axis = simulated_camera.get_x_y_axis()
        x_size, y_size = simulated_camera.get_geometry()
        image_background_array = numpy.zeros(shape=(y_size, x_size), dtype="uint16") + 3

        parameters = {

            "image_threshold": 1,
            "image_region_of_interest": [0, 2048, 0, 2048],

            "image_good_region": {
                "threshold": 0.3,
                "gfscale": 1.8
            },

            "image_slices": {
                "number_of_slices": 5,
                "scale": 1.0,
                "orientation": "horizontal"
            }
        }

        profile = LineProfiler(process_image)
        process_image_wrapper = profile(process_image)

        n_iterations = 300

        print("Generating images.")

        images = []
        for _ in range(n_iterations):
            images.append(simulated_camera.get_image())

        print("Processing images.")

        start_time = time.time()
        for image in images:
            process_image_wrapper(image=image,
                                  pulse_id=0,
                                  timestamp=time.time(),
                                  x_axis=x_axis,
                                  y_axis=y_axis,
                                  parameters=parameters,
                                  image_background_array=image_background_array)
        end_time = time.time()

        time_difference = end_time - start_time
        rate = n_iterations / time_difference

        print("Processing rate: ", rate)

        profile.print_stats()

    def test_single_function(self):
        # Profile only if LineProfiler present.
        # To install: conda install line_profiler
        try:
            from line_profiler import LineProfiler
        except ImportError:
            return

        function_to_perf = functions.subtract_background
        n_iterations = 200
        n_tests = 5

        simulated_camera = CameraSimulation(CameraConfig("simulation"), size_x=2048, size_y=2048)

        for _ in range(n_tests):

            profile = LineProfiler()
            wrapped_function = profile(function_to_perf)

            images = []
            backgrounds = []

            for _ in range(n_iterations):
                images.append(simulated_camera.get_image())
                backgrounds.append(simulated_camera.get_image())

            for index in range(n_iterations):
                wrapped_function(images[index], backgrounds[index])

            profile.print_stats()


if __name__ == '__main__':
    unittest.main()
