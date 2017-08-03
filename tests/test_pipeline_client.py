import os
import signal
import unittest

from multiprocessing import Process
from time import sleep

from cam_server import CamClient, PipelineClient
from cam_server.start_cam_server import start_camera_server
from cam_server.start_pipeline_server import start_pipeline_server


class PipelineClientTest(unittest.TestCase):
    def setUp(self):
        self.host = "0.0.0.0"
        self.cam_port = 8888
        self.pipeline_port = 8889

        test_base_dir = os.path.split(os.path.abspath(__file__))[0]
        cam_config_folder = os.path.join(test_base_dir, "camera_config/")
        pipeline_config_folder = os.path.join(test_base_dir, "pipeline_config/")
        background_config_folder = os.path.join(test_base_dir, "background_config/")

        cam_server_address = "http://%s:%s" % (self.host, self.cam_port)
        pipeline_server_address = "http://%s:%s" % (self.host, self.pipeline_port)

        self.cam_process = Process(target=start_camera_server, args=(self.host, self.cam_port,
                                                                     cam_config_folder))
        self.cam_process.start()

        self.pipeline_process = Process(target=start_pipeline_server, args=(self.host, self.pipeline_port,
                                                                            pipeline_config_folder,
                                                                            background_config_folder,
                                                                            cam_server_address))
        self.pipeline_process.start()

        # Give it some time to start.
        sleep(1)

        self.cam_client = CamClient(cam_server_address)
        self.pipeline_client = PipelineClient(pipeline_server_address)

    def tearDown(self):
        self.cam_client.stop_all_cameras()
        self.pipeline_client.stop_all_instances()

        os.kill(self.cam_process.pid, signal.SIGINT)
        os.kill(self.pipeline_process.pid, signal.SIGINT)
        try:
            os.remove(os.path.join(self.config_folder, "testing_camera.json"))
        except:
            pass
        # Wait for the server to die.
        sleep(1)

    def test_client(self):
        print(self.cam_client.get_cameras())
        print(self.pipeline_client.get_pipelines())


if __name__ == '__main__':
    unittest.main()
