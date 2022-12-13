import multiprocessing
import os
import unittest
from multiprocessing import Process
from threading import Thread
from time import sleep

import numpy
from bsread import SUB, source, PULL

from cam_server import CamClient, config
from cam_server.pipeline.configuration import PipelineConfig
from cam_server.pipeline.types.processing import run as processing_pipeline
from cam_server.pipeline.types.store import run as store_pipeline
from cam_server.start_camera_server import start_camera_server
from tests import test_cleanup, get_simulated_camera
from tests.helpers.factory import MockBackgroundManager


class PipelineTransceiverTest(unittest.TestCase):
    def setUp(self):
        self.host = "127.0.0.1"
        self.port = 8888

        test_base_dir = os.path.split(os.path.abspath(__file__))[0]
        self.config_folder = os.path.join(test_base_dir, "camera_config/")
        self.user_scripts_folder = os.path.join(test_base_dir, "user_scripts/")

        self.process = Process(target=start_camera_server, args=(self.host, self.port, self.config_folder, self.user_scripts_folder))
        self.process.start()

        # Give it some time to start.
        sleep(1.0)

        self.rest_api_endpoint = "http://%s:%s" % (self.host, self.port)
        self.client = CamClient(self.rest_api_endpoint)

    def tearDown(self):
        test_cleanup([self.client], [self.process], [os.path.join(self.config_folder, "simulation_temp.json")])

    def test_pipeline_with_simulation_camera(self):
        manager = multiprocessing.Manager()
        stop_event = multiprocessing.Event()
        statistics = manager.Namespace()
        parameter_queue = multiprocessing.Queue()

        pipeline_config = PipelineConfig("test_pipeline")

        def send():
            processing_pipeline(stop_event, statistics, parameter_queue, self.client,
                                pipeline_config, 12000, MockBackgroundManager())

        thread = Thread(target=send)
        thread.start()

        with source(host="127.0.0.1", port=12000, mode=SUB, receive_timeout=3000) as stream:
            data = stream.receive()
            self.assertIsNotNone(data, "Received None message.")

            required_keys = set(['x_fit_standard_deviation', 'x_axis', 'y_fit_standard_deviation', 'x_center_of_mass',
                                 'x_fit_amplitude', 'y_fit_mean', 'processing_parameters', 'timestamp',
                                 'y_fit_gauss_function', 'y_profile', 'y_center_of_mass', 'x_fit_gauss_function',
                                 'x_rms', 'y_rms', 'y_fit_amplitude', 'image', 'y_axis', 'min_value', 'x_fit_mean',
                                 'max_value', 'x_fit_offset', 'x_profile', 'y_fit_offset', 'width', 'height',
                                 'intensity', "x_fwhm", "y_fwhm"])

            self.assertSetEqual(required_keys, set(data.data.data.keys()),
                                "Missing required keys in pipeline output bsread message.")

        stop_event.set()
        thread.join()

    def test_system_exit(self):
        manager = multiprocessing.Manager()
        stop_event = multiprocessing.Event()
        statistics = manager.Namespace()
        parameter_queue = multiprocessing.Queue()
        pipeline_config = PipelineConfig("test_pipeline", parameters={
            "camera_name": "simulation",
            "image_background": "full_background",
            "image_background_enable": True,
            "image_threshold": 0
        })

        background_manager = MockBackgroundManager()

        with self.assertRaises(SystemExit):
            processing_pipeline(stop_event, statistics, parameter_queue, self.client, pipeline_config,
                                12001, background_manager)

    def test_pipeline_background_manager(self):
        manager = multiprocessing.Manager()
        stop_event = multiprocessing.Event()
        statistics = manager.Namespace()
        parameter_queue = multiprocessing.Queue()

        pipeline_config = PipelineConfig("test_pipeline", parameters={
            "camera_name": "simulation",
            "image_background": "full_background",
            "image_background_enable": True,
            "image_threshold": 0
        })

        background_manager = MockBackgroundManager()


        x_size, y_size = get_simulated_camera().get_geometry()
        simulated_camera_shape = (y_size, x_size)

        background_array = numpy.zeros(shape=simulated_camera_shape)
        background_array.fill(65535)
        background_manager.save_background("full_background", background_array, append_timestamp=False)

        def send():
            processing_pipeline(stop_event, statistics, parameter_queue, self.client, pipeline_config,
                                12001, background_manager)

        thread = Thread(target=send)
        thread.start()

        with source(host="127.0.0.1", port=12001, mode=SUB, receive_timeout = 3000) as stream:
            data = stream.receive()
            self.assertIsNotNone(data, "Received None message.")
            self.assertTrue(numpy.array_equal(data.data.data["image"].value, numpy.zeros(shape=simulated_camera_shape)))

        stop_event.set()
        thread.join()

    def test_rotate_camera(self):
        manager = multiprocessing.Manager()
        stop_event = multiprocessing.Event()
        statistics = manager.Namespace()
        parameter_queue = multiprocessing.Queue()

        self.client.set_camera_config("simulation_temp", self.client.get_camera_config("simulation"))

        pipeline_config = PipelineConfig("test_pipeline", parameters={
            "camera_name": "simulation_temp"
        })

        background_manager = MockBackgroundManager()

        x_size, y_size = get_simulated_camera().get_geometry()
        simulated_camera_shape = (y_size, x_size)

        def send():
            processing_pipeline(stop_event, statistics, parameter_queue, self.client, pipeline_config,
                                12002, background_manager)

        thread = Thread(target=send)
        thread.start()

        with source(host="127.0.0.1", port=12002, mode=SUB, receive_timeout = 3000) as stream:
            data = stream.receive()
            self.assertIsNotNone(data, "Received None message.")

            x_size = data.data.data["width"].value
            y_size = data.data.data["height"].value
            x_axis = data.data.data["x_axis"].value
            y_axis = data.data.data["y_axis"].value
            image_shape = data.data.data["image"].value.shape

            self.assertEqual(x_size, simulated_camera_shape[1])
            self.assertEqual(y_size, simulated_camera_shape[0])

            # Sanity checks.
            self.assertEqual(x_size, len(x_axis))
            self.assertEqual(y_size, len(y_axis))
            self.assertEqual(image_shape, (y_size, x_size))

            # Rotate the image by 90 degree.
            camera_config = self.client.get_camera_config("simulation_temp")
            camera_config["rotate"] = 1
            self.client.set_camera_config("simulation_temp", camera_config)

            # Make a few frames pass.
            for _ in range(5):
                data = stream.receive()

            self.assertIsNotNone(data, "Received None message.")

            x_size = data.data.data["width"].value
            y_size = data.data.data["height"].value
            x_axis = data.data.data["x_axis"].value
            y_axis = data.data.data["y_axis"].value
            image_shape = data.data.data["image"].value.shape

            # X and Y size should be inverted.
            self.assertEqual(x_size, simulated_camera_shape[0])
            self.assertEqual(y_size, simulated_camera_shape[1])

            # Sanity checks.
            self.assertEqual(x_size, len(x_axis))
            self.assertEqual(y_size, len(y_axis))
            self.assertEqual(image_shape, (y_size, x_size))

        self.client.delete_camera_config("simulation_temp")

        stop_event.set()
        thread.join()

    def test_store_pipeline_with_simulated_camera(self):
        manager = multiprocessing.Manager()
        stop_event = multiprocessing.Event()
        statistics = manager.Namespace()
        parameter_queue = multiprocessing.Queue()

        self.client.stop_all_instances()

        pipeline_config = PipelineConfig("test_pipeline")

        def send():
            store_pipeline(stop_event, statistics, parameter_queue, self.client,
                           pipeline_config, 12003, MockBackgroundManager())

        thread = Thread(target=send)
        thread.start()

        sleep(0.5)

        with source(host="127.0.0.1", port=12003, mode=PULL, receive_timeout = 3000) as stream:
            data = stream.receive()
            self.assertIsNotNone(data, "Receiving timeout.")

            required_keys = set(["simulation" + config.EPICS_PV_SUFFIX_IMAGE])

            self.assertSetEqual(required_keys, set(data.data.data.keys()),
                                "Missing required keys in pipeline output bsread message.")

        stop_event.set()
        thread.join()


    def test_transparent_pipeline(self):
        manager = multiprocessing.Manager()
        stop_event = multiprocessing.Event()
        statistics = manager.Namespace()
        parameter_queue = multiprocessing.Queue()

        pipeline_config = PipelineConfig("test_pipeline", parameters={"camera_name": "simulation",
                                                                      "function":"transparent"})

        def send():
            processing_pipeline(stop_event, statistics, parameter_queue, self.client,
                                pipeline_config, 12004, MockBackgroundManager())

        thread = Thread(target=send)
        thread.start()

        with source(host="127.0.0.1", port=12004, mode=SUB, receive_timeout = 3000) as stream:
            data = stream.receive()
            self.assertIsNotNone(data, "Received None message.")

            required_keys = set(["image", "timestamp", "width", "height", "x_axis", "y_axis", "processing_parameters"])

            self.assertSetEqual(required_keys, set(data.data.data.keys()),
                                "Missing required keys in pipeline output bsread message.")

        stop_event.set()
        thread.join(5.0)


if __name__ == '__main__':
    unittest.main()
