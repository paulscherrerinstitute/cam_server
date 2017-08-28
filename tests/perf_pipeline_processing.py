import unittest

import numpy
import time

from cam_server.camera.configuration import CameraConfig
from cam_server.camera.receiver import CameraSimulation
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

        simulated_camera = CameraSimulation(CameraConfig("simulation"))
        image = simulated_camera.get_image()
        x_axis, y_axis = simulated_camera.get_x_y_axis()
        x_size, y_size = simulated_camera.get_geometry()
        image_background_array = numpy.zeros(shape=(y_size, x_size))

        parameters = {

            "image_threshold": 0.5,
            "image_region_of_interest": [0, 100, 0, 100],

            "image_good_region": {
                "threshold": 0.9,
                "gfscale": 3
            },

            "image_slices": {
                "number_of_slices": 1,
                "scale": 1.0
            }
        }

        profile = LineProfiler(process_image)
        profile.add_function(functions.gauss_fit)
        profile.add_function(functions._gauss_fit)
        profile.add_function(functions._gauss_function)

        process_image_wrapper = profile(process_image)

        n_iterations = 100

        for _ in range(n_iterations):
            process_image_wrapper(image=image,
                                  timestamp=time.time(),
                                  x_axis=x_axis,
                                  y_axis=y_axis,
                                  parameters=parameters,
                                  image_background_array=image_background_array)

        profile.print_stats()


if __name__ == '__main__':
    unittest.main()
