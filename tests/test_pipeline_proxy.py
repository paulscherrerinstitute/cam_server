import base64
import os
import signal
import unittest
from multiprocessing import Process
from time import sleep

import numpy
from bsread import source, SUB

from cam_server import config
from cam_server import CamClient, PipelineClient
from cam_server.instance_management.proxy import ProxyClient
from cam_server.camera.configuration import CameraConfig
from cam_server.camera.source.simulation import CameraSimulation
from cam_server.start_camera_worker import start_camera_worker
from cam_server.start_camera_manager import start_camera_manager
from cam_server.start_pipeline_worker import start_pipeline_worker
from cam_server.start_pipeline_manager import start_pipeline_manager
from cam_server.utils import get_host_port_from_stream_address

from tests import test_cleanup, is_port_available


class CameraClientProxyTest(unittest.TestCase):
    def setUp(self):
        test_base_dir = os.path.split(os.path.abspath(__file__))[0]
        self.cam_config_folder = os.path.join(test_base_dir, "camera_config/")
        self.pipeline_config_folder = os.path.join(test_base_dir, "pipeline_config/")
        self.background_config_folder = os.path.join(test_base_dir, "background_config/")
        self.temp_folder = os.path.join(test_base_dir, "temp/")

        self.host = "0.0.0.0"
        self.cam_server_ports = [8880, 8881]
        self.cam_proxy_port = 8898
        self.pipeline_server_ports = [8890, 8891]
        self.pipeline_proxy_port = 8899

        self.cam_server_address = []
        self.process_camserver = []
        port_range = config.CAMERA_STREAM_PORT_RANGE
        for p in self.cam_server_ports:
            port_range = [port_range[0]+10000, port_range[1]+10000]
            self.cam_server_address.append("http://%s:%s" % (self.host, p))
            process = Process(target=start_camera_worker, args=(self.host, p, None, port_range))
            self.process_camserver.append(process)
            process.start()
        self.cam_proxy_host = "0.0.0.0"

        self.process_camproxy = Process(target=start_camera_manager,
                                        args=(self.host, self.cam_proxy_port, ",".join(self.cam_server_address),
                                              self.cam_config_folder))
        self.process_camproxy.start()
        cam_server_proxy_address = "http://%s:%s" % (self.host, self.cam_proxy_port)
        pipeline_server_proxy_address = "http://%s:%s" % (self.host, self.pipeline_proxy_port)

        self.pipeline_server_address = []
        self.process_pipelineserver = []
        port_range = config.PIPELINE_STREAM_PORT_RANGE
        for p in self.pipeline_server_ports:
            port_range = [port_range[0]+10000, port_range[1]+10000]
            self.pipeline_server_address.append("http://%s:%s" % (self.host, p))
            process = Process(target=start_pipeline_worker, args=(self.host, p,
                                                        self.temp_folder,
                                                        cam_server_proxy_address,
                                                        None,
                                                        port_range))
            self.process_pipelineserver.append(process)
            process.start()

        cfg = ",".join(self.pipeline_server_address)

        self.pipeline_proxy_process = Process(target=start_pipeline_manager, args=(self.host, self.pipeline_proxy_port,
                                                                    cfg,
                                                                    self.pipeline_config_folder,
                                                                    self.background_config_folder,
                                                                    cam_server_proxy_address))
        self.pipeline_proxy_process.start()

        sleep(1.0) # Give it some time to start.

        cam_server_address = "http://%s:%s" % (self.host, self.cam_proxy_port)
        self.cam_client = CamClient(cam_server_address)
        pipeline_server_address = "http://%s:%s" % (self.host, self.pipeline_proxy_port)
        self.pipeline_client = PipelineClient(pipeline_server_address)

        self.cam_proxy_client = ProxyClient(cam_server_address)
        self.pipeline_proxy_client = ProxyClient(pipeline_server_address)

    def tearDown(self):
        test_cleanup([self.cam_client, self.pipeline_client],
                     [self.pipeline_proxy_process, ] + self.process_pipelineserver +
                     [self.process_camproxy, ] + self.process_camserver ,
                     [])

    def test_manager(self):
        #Creating instances from name
        instance_id_1, instance_stream_1 = self.pipeline_client.create_instance_from_name("simulation_sp")
        instance_id_2, instance_stream_2 = self.pipeline_client.create_instance_from_name("simulation2_sp")
        print (instance_id_1, instance_stream_1)
        print(instance_id_2, instance_stream_2)
        # Check if streams are alive
        pipeline_host, pipeline_port = get_host_port_from_stream_address(instance_stream_1)
        with source(host=pipeline_host, port=pipeline_port, mode=SUB) as stream:
            data = stream.receive()
            for key in ["image", "width", "height"]:
                self.assertIn(key, data.data.data.keys())

        pipeline_host, pipeline_port = get_host_port_from_stream_address(instance_stream_2)
        with source(host=pipeline_host, port=pipeline_port, mode=SUB) as stream:
            data = stream.receive()
            for key in ["image", "width", "height"]:
                self.assertIn(key, data.data.data.keys())

        #check client
        server_info = self.cam_proxy_client.get_servers_info()
        status_info = self.cam_proxy_client.get_status_info()
        instance_info = self.cam_proxy_client.get_instances_info()
        #Check if camera streams are equally distributed
        self.assertEqual(server_info[self.cam_server_address[0]]["load"], 1)
        self.assertEqual(server_info[self.cam_server_address[1]]["load"], 1)


        server_info = self.pipeline_proxy_client.get_servers_info()
        status_info = self.pipeline_proxy_client.get_status_info()
        instance_info = self.pipeline_proxy_client.get_instances_info()
        print(server_info)
        print(instance_info)
        #Check if pipeline are equally distributed
        self.assertEqual(server_info[self.pipeline_server_address[0]]["load"], 1)
        self.assertEqual(server_info[self.pipeline_server_address[1]]["load"], 1)

        # Check if instance information is available  for each server instance
        for instance in server_info[self.pipeline_server_address[0]]["instances"]:
            self.assertIn(instance, instance_info)
        for instance in server_info[self.pipeline_server_address[1]]["instances"]:
            self.assertIn(instance, instance_info)

        #Test stopping instances
        self.pipeline_client.stop_instance(instance_id_1)
        self.pipeline_client.stop_instance(instance_id_2)
        server_info = self.pipeline_proxy_client.get_servers_info()
        self.assertEqual(server_info[self.pipeline_server_address[0]]["load"], 0)
        self.assertEqual(server_info[self.pipeline_server_address[1]]["load"], 0)


        #Creating instances from config
        instance_id_1, instance_stream_1 = self.pipeline_client.create_instance_from_config({"camera_name": "simulation"})
        instance_id_2, instance_stream_2 = self.pipeline_client.create_instance_from_config({"camera_name": "simulation2"})
        print (instance_id_1, instance_stream_1)
        print(instance_id_2, instance_stream_2)
        # Check if streams are alive
        pipeline_host, pipeline_port = get_host_port_from_stream_address(instance_stream_1)
        with source(host=pipeline_host, port=pipeline_port, mode=SUB) as stream:
            data = stream.receive()
            for key in ["image", "width", "height"]:
                self.assertIn(key, data.data.data.keys())

        pipeline_host, pipeline_port = get_host_port_from_stream_address(instance_stream_2)
        with source(host=pipeline_host, port=pipeline_port, mode=SUB) as stream:
            data = stream.receive()
            for key in ["image", "width", "height"]:
                self.assertIn(key, data.data.data.keys())
        server_info = self.pipeline_proxy_client.get_servers_info()
        status_info = self.pipeline_proxy_client.get_status_info()
        instance_info = self.pipeline_proxy_client.get_instances_info()
        print(server_info)
        print(instance_info)
        #Check if pipeline are equally distributed
        self.assertEqual(server_info[self.pipeline_server_address[0]]["load"], 1)
        self.assertEqual(server_info[self.pipeline_server_address[1]]["load"], 1)

        # Check if instance information is available  for each server instance
        for instance in server_info[self.pipeline_server_address[0]]["instances"]:
            self.assertIn(instance, instance_info)
        for instance in server_info[self.pipeline_server_address[1]]["instances"]:
            self.assertIn(instance, instance_info)

        #Test stopping instances
        self.pipeline_client.stop_instance(instance_id_1)
        self.pipeline_client.stop_instance(instance_id_2)
        server_info = self.pipeline_proxy_client.get_servers_info()
        self.assertEqual(server_info[self.pipeline_server_address[0]]["load"], 0)
        self.assertEqual(server_info[self.pipeline_server_address[1]]["load"], 0)

        #Server Config
        self.assertEqual(self.pipeline_proxy_client.get_config(), {'http://0.0.0.0:8890': {'expanding': True}, 'http://0.0.0.0:8891': {'expanding': True}})
        self.pipeline_proxy_client.set_config({'http://0.0.0.0:8890': {'expanding': True}, 'http://0.0.0.0:8891': {"instances":["DUMMY"], 'expanding': False}})

        instance_id_1, instance_stream_1 = self.pipeline_client.create_instance_from_config({"camera_name": "simulation"})
        instance_id_2, instance_stream_2 = self.pipeline_client.create_instance_from_config({"camera_name": "simulation2"})
        #Check if streams are alive
        pipeline_host, pipeline_port = get_host_port_from_stream_address(instance_stream_1)
        pipeline_host, pipeline_port = get_host_port_from_stream_address(instance_stream_2)
        server_info = self.pipeline_proxy_client.get_servers_info()
        self.assertEqual(server_info[self.pipeline_server_address[0]]["load"], 2)
        self.assertEqual(server_info[self.pipeline_server_address[1]]["load"], 0)

if __name__ == '__main__':
    unittest.main()
