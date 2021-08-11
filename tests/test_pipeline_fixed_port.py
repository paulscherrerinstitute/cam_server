
import os
import unittest
from multiprocessing import Process
from time import sleep

from cam_server import config
from cam_server import CamClient, PipelineClient, ProxyClient
from cam_server.start_camera_worker import start_camera_worker
from cam_server.start_camera_manager import start_camera_manager
from cam_server.start_pipeline_worker import start_pipeline_worker
from cam_server.start_pipeline_manager import start_pipeline_manager
from cam_server.utils import get_host_port_from_stream_address

from tests import test_cleanup, is_port_available, require_folder


class CameraClientProxyTest(unittest.TestCase):
    def setUp(self):
        test_base_dir = os.path.split(os.path.abspath(__file__))[0]
        self.cam_config_folder = os.path.join(test_base_dir, "camera_config/")
        self.pipeline_config_folder = os.path.join(test_base_dir, "pipeline_config/")
        self.background_config_folder = os.path.join(test_base_dir, "background_config/")
        self.user_scripts_folder = os.path.join(test_base_dir, "user_scripts/")
        self.temp_folder = os.path.join(test_base_dir, "temp/")

        require_folder(self.background_config_folder)
        require_folder(self.user_scripts_folder)
        require_folder(self.temp_folder)

        self.host = "localhost"
        self.cam_server_ports = [8880, 8881]
        self.cam_manager_port = 8888
        self.pipeline_server_ports = [8890, 8891]
        self.pipeline_manager_port = 8889

        self.cam_server_address = []
        self.process_camserver = []
        port_range = config.CAMERA_STREAM_PORT_RANGE
        for p in self.cam_server_ports:
            port_range = [port_range[0]+10000, port_range[1]+10000]
            self.cam_server_address.append("http://%s:%s" % (self.host, p))
            process = Process(target=start_camera_worker, args=(self.host, p, self.user_scripts_folder, None, port_range))
            self.process_camserver.append(process)
            process.start()
        self.cam_proxy_host = "0.0.0.0"

        self.process_camproxy = Process(target=start_camera_manager,
                                        args=(self.host, self.cam_manager_port, ",".join(self.cam_server_address),
                                              self.cam_config_folder, self.user_scripts_folder))
        self.process_camproxy.start()
        cam_server_proxy_address = "http://%s:%s" % (self.host, self.cam_manager_port)
        pipeline_server_proxy_address = "http://%s:%s" % (self.host, self.pipeline_manager_port)

        self.pipeline_server_address = []
        self.process_pipelineserver = []
        port_range = config.PIPELINE_STREAM_PORT_RANGE
        for p in self.pipeline_server_ports:
            port_range = [port_range[0]+10000, port_range[1]+10000]
            self.pipeline_server_address.append("http://%s:%s" % (self.host, p))
            process = Process(target=start_pipeline_worker, args=(self.host, p,
                                                        self.temp_folder,
                                                        self.temp_folder,
                                                        cam_server_proxy_address,
                                                        None,
                                                        port_range))
            self.process_pipelineserver.append(process)
            process.start()

        cfg = """{
          "http://localhost:8890": {
          "expanding": true
         }, "http://localhost:8891": {
          "cameras": [
           "simulation3"
          ],
          "expanding": false,
          "instances": [
           "simulation_sp:10123"
          ]
         }
        }"""

        self.pipeline_proxy_process = Process(target=start_pipeline_manager, args=(self.host, self.pipeline_manager_port,
                                                                    cfg,
                                                                    self.pipeline_config_folder,
                                                                    self.background_config_folder,
                                                                    config.DEFAULT_BACKGROUND_FILES_DAYS_TO_LIVE,
                                                                    self.user_scripts_folder,
                                                                    cam_server_proxy_address))
        self.pipeline_proxy_process.start()

        sleep(1.0)  # Give it some time to start.

        cam_server_address = "http://%s:%s" % (self.host, self.cam_manager_port)
        self.cam_client = CamClient(cam_server_address)
        pipeline_server_address = "http://%s:%s" % (self.host, self.pipeline_manager_port)
        self.pipeline_client = PipelineClient(pipeline_server_address)

        self.cam_proxy_client = ProxyClient(cam_server_address)
        self.pipeline_proxy_client = ProxyClient(pipeline_server_address)

    def tearDown(self):
        test_cleanup([self.cam_client, self.pipeline_client],
                     [self.pipeline_proxy_process, ] + self.process_pipelineserver +
                     [self.process_camproxy, ] + self.process_camserver,
                     [self.background_config_folder,
                      self.user_scripts_folder,
                      self.temp_folder])

    def test_manager(self):
        #Creating instances from config
        instance_id_1, instance_stream_1 = self.pipeline_client.create_instance_from_name("simulation_sp")
        pipeline_host_1, pipeline_port_1 = get_host_port_from_stream_address(instance_stream_1)
        self.assertEqual(pipeline_port_1, 10123)

        self.pipeline_client.stop_instance(instance_id_1)

        instance_id_2, instance_stream_2 = self.pipeline_client.create_instance_from_name("simulation_sp")
        pipeline_host_2, pipeline_port_2 = get_host_port_from_stream_address(instance_stream_2)
        self.assertEqual(pipeline_host_1, pipeline_host_2)
        self.assertEqual(pipeline_port_1, pipeline_port_2)


if __name__ == '__main__':
    unittest.main()
