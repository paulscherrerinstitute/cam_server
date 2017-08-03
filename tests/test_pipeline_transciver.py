import os
import signal
import unittest

from multiprocessing import Process, Event
from threading import Thread
from time import sleep

import multiprocessing

import numpy
from bsread import SUB, source
from cam_server import CamClient
from cam_server.pipeline.configuration import PipelineConfig
from cam_server.pipeline.transceiver import receive_process_send
from cam_server.start_camera_server import start_camera_server
from tests.helpers.factory import MockBackgroundManager


class PipelineTransceiverTest(unittest.TestCase):
    def setUp(self):
        self.host = "0.0.0.0"
        self.port = 8888

        test_base_dir = os.path.split(os.path.abspath(__file__))[0]
        self.config_folder = os.path.join(test_base_dir, "camera_config/")

        self.process = Process(target=start_camera_server, args=(self.host, self.port, self.config_folder))
        self.process.start()

        # Give it some time to start.
        sleep(0.5)

        self.rest_api_endpoint = "http://%s:%s" % (self.host, self.port)
        self.client = CamClient(self.rest_api_endpoint)

    def tearDown(self):
        self.client.stop_all_cameras()
        try:
            os.kill(self.process.pid, signal.SIGINT)
        except:
            pass
        try:
            os.remove(os.path.join(self.config_folder, "testing_camera.json"))
        except:
            pass
        # Wait for the server to die.
        sleep(1)

    def test_pipeline_with_simulation_camera(self):

        manager = multiprocessing.Manager()
        stop_event = multiprocessing.Event()
        statistics = manager.Namespace()
        parameter_queue = multiprocessing.Queue()

        pipeline_config = PipelineConfig("test_pipeline")

        def send():
            receive_process_send(stop_event, statistics, parameter_queue, self.client,
                                 pipeline_config, 12000, MockBackgroundManager())

        thread = Thread(target=send)
        thread.start()

        with source(host="127.0.0.1", port=12000, mode=SUB) as stream:
            data = stream.receive()
            self.assertIsNotNone(data, "Received None message.")

            required_keys = set(['x_fit_standard_deviation', 'x_axis', 'y_fit_standard_deviation', 'x_center_of_mass',
                                 'x_fit_amplitude', 'y_fit_mean', 'processing_parameters', 'timestamp',
                                 'y_fit_gauss_function', 'y_profile', 'y_center_of_mass', 'x_fit_gauss_function',
                                 'x_rms', 'y_rms', 'y_fit_amplitude', 'image', 'y_axis', 'min_value', 'x_fit_mean',
                                 'max_value', 'x_fit_offset', 'x_profile', 'y_fit_offset'])

            self.assertSetEqual(required_keys, set(data.data.data.keys()),
                                "Missing required keys in pipeline output bsread message.")

        thread.join()

    def test_pipeline_background_manager(self):
        manager = multiprocessing.Manager()
        stop_event = multiprocessing.Event()
        statistics = manager.Namespace()
        parameter_queue = multiprocessing.Queue()

        pipeline_config = PipelineConfig("test_pipeline", parameters={
            "camera_name": "simulation",
            "image_background": "full_background"
        })

        background_manager = MockBackgroundManager()

        with self.assertRaises(Exception):
            receive_process_send(stop_event, statistics, parameter_queue, self.client, pipeline_config,
                                 12000, background_manager)

        simulated_camera_shape = (960, 1280)

        background_array = numpy.zeros(shape=simulated_camera_shape)
        background_array.fill(99999)
        background_manager.save_background("full_background", background_array)

        def send():
            receive_process_send(stop_event, statistics, parameter_queue, self.client, pipeline_config,
                                 12000, background_manager)

        thread = Thread(target=send)
        thread.start()

        with source(host="127.0.0.1", port=12000, mode=SUB) as stream:
            data = stream.receive()

            self.assertIsNotNone(data, "Received None message.")
            self.assertTrue(numpy.array_equal(data.data.data["image"].value, numpy.zeros(shape=simulated_camera_shape)))

        thread.join()

    def test_pipeline_camera_calibration(self):
        manager = multiprocessing.Manager()
        stop_event = multiprocessing.Event()
        statistics = manager.Namespace()
        parameter_queue = multiprocessing.Queue()

        # This is the "neutral" calibration - it should not change the image.
        calibration_config = PipelineConfig("test_pipeline", parameters={
            "camera_name": "simulation",
            "calibration": {
                "reference_marker": [0, 0, 100, 100],
                "reference_marker_width": 100.0,
                "reference_marker_height": 100.0,
                "angle_horizontal": 0.0,
                "angle_vertical": 0.0
            }
        })

        def send():
            receive_process_send(stop_event, statistics, parameter_queue, self.client, calibration_config,
                                 12001, MockBackgroundManager())

        thread = Thread(target=send)
        thread.start()

        with source(host="127.0.0.1", port=12001, mode=SUB) as stream:
            data = stream.receive()
            x_axis = data.data.data["x_axis"].value
            y_axis = data.data.data["y_axis"].value

            self.assertIsNotNone(x_axis, "x_axis is None.")
            self.assertIsNotNone(y_axis, "y_axis is None.")

        thread.join()

        # TODO: How to test if the calibration was correct?


if __name__ == '__main__':
    unittest.main()
