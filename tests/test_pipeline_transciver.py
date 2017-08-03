import os
import signal
import unittest

from multiprocessing import Process, Event
from threading import Thread
from time import sleep

import multiprocessing

from bsread import SUB, source
from cam_server import CamClient
from cam_server.pipeline.configuration import PipelineConfig
from cam_server.pipeline.transceiver import receive_process_send
from cam_server.start_cam_server import start_camera_server
from cam_server.utils import get_host_port_from_stream_address
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
            receive_process_send(stop_event, statistics, parameter_queue, self.rest_api_endpoint,
                                 pipeline_config, 12000, MockBackgroundManager())

        thread = Thread(target=send)
        thread.start()

        with source(host="127.0.0.1", port=12000, mode=SUB) as stream:
            data = stream.receive()
            self.assertIsNotNone(data, "Received None message.")

        thread.join()

    def test_pipeline_background_manager(self):
        manager = multiprocessing.Manager()
        stop_event = multiprocessing.Event()
        statistics = manager.Namespace()
        parameter_queue = multiprocessing.Queue()

        camera_stream_address = self.client.get_camera_stream("simulation")
        host, port = get_host_port_from_stream_address(camera_stream_address)

        pipeline_config = PipelineConfig("test_pipeline", parameters={
            "camera_name": "simulation",
            "image_background": "full_background"
        })

        receive_process_send(stop_event, statistics, parameter_queue, self.rest_api_endpoint, pipeline_config, 12000,
                             MockBackgroundManager())

        # camera_stream_address = self.client.get_camera_stream("simulation")
        # host, port = get_host_port_from_stream_address(camera_stream_address)
        #
        # def send():
        #     receive_process_send(stop_event, statistics, parameter_queue, host, port, pipeline_config, 12000,
        #                          MockBackgroundManager())
        #
        # thread = Thread(target=send)
        # thread.start()
        #
        # with source(host="127.0.0.1", port=12000, mode=SUB) as stream:
        #     data = stream.receive()
        #     self.assertIsNotNone(data, "Received None message.")
        #
        # thread.join()

if __name__ == '__main__':
    unittest.main()
