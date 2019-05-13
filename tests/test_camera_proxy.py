import base64
import os
import signal
import unittest
from multiprocessing import Process
from time import sleep

import numpy
from bsread import source, SUB

from cam_server import config
from cam_server import CamClient, ProxyClient
from cam_server.camera.configuration import CameraConfig
from cam_server.camera.source.simulation import CameraSimulation
from cam_server.start_camera_worker import start_camera_worker
from cam_server.start_camera_manager import start_camera_manager
from cam_server.utils import get_host_port_from_stream_address

from tests import test_cleanup, is_port_available


class CameraClientProxyTest(unittest.TestCase):
    def setUp(self):
        test_base_dir = os.path.split(os.path.abspath(__file__))[0]
        self.config_folder = os.path.join(test_base_dir, "camera_config/")

        self.host = "0.0.0.0"
        self.cam_server_ports = [8880, 8881]
        self.cam_manager_port = 8888
        self.cam_server_address = []
        self.process_camserver = []
        port_range = config.CAMERA_STREAM_PORT_RANGE

        for port in self.cam_server_ports:
            print(port, is_port_available(port))

        for p in self.cam_server_ports:
            port_range = [port_range[0]+10000, port_range[1]+10000]
            self.cam_server_address.append("http://%s:%s" % (self.host, p))
            process = Process(target=start_camera_worker, args=(self.host, p, None, port_range))
            self.process_camserver.append(process)
            process.start()

        self.cam_proxy_host = "0.0.0.0"
        self.process_camproxy = Process(target=start_camera_manager,
                                        args=(self.host, self.cam_manager_port, ",".join(self.cam_server_address), self.config_folder))
        self.process_camproxy.start()
        sleep(1.0) # Give it some time to start.
        server_address = "http://%s:%s" % (self.host, self.cam_manager_port)
        self.client = CamClient(server_address)

        self.proxy_client = ProxyClient(server_address)

    def tearDown(self):
        test_cleanup([self.client], [self.process_camproxy, ] + self.process_camserver, [])

    def test_manager(self):
        stream_address_1 = self.client.get_instance_stream("simulation")
        stream_address_2 = self.client.get_instance_stream("simulation2")

        #Check if streams are alive
        camera_host, camera_port = get_host_port_from_stream_address(stream_address_1)
        with source(host=camera_host, port=camera_port, mode=SUB) as stream:
            data = stream.receive()
            for key in ["image", "width", "height"]:
                self.assertIn(key, data.data.data.keys())

        camera_host, camera_port = get_host_port_from_stream_address(stream_address_2)
        with source(host=camera_host, port=camera_port, mode=SUB) as stream:
            data = stream.receive()
            for key in ["image", "width", "height"]:
                self.assertIn(key, data.data.data.keys())

        server_info = self.proxy_client.get_servers_info()
        status_info = self.proxy_client.get_status_info()
        instance_info = self.proxy_client.get_instances_info()


        #Check if streams are equally distributed
        self.assertEqual(server_info[self.cam_server_address[0]]["load"], 1)
        self.assertEqual(server_info[self.cam_server_address[1]]["load"], 1)

        # Check if instance information is available  for each server instance
        for instance in server_info[self.cam_server_address[0]]["instances"]:
            self.assertIn(instance, instance_info)
        for instance in server_info[self.cam_server_address[1]]["instances"]:
            self.assertIn(instance, instance_info)
        self.client.stop_all_instances()

        #Server Config
        self.assertEqual(self.proxy_client.get_config(), {'http://0.0.0.0:8880': {'expanding': True}, 'http://0.0.0.0:8881': {'expanding': True}})
        self.proxy_client.set_config({'http://0.0.0.0:8880': {'expanding': True}, 'http://0.0.0.0:8881': {"instances":["DUMMY"], 'expanding': False}})

        stream_address_1 = self.client.get_instance_stream("simulation")
        stream_address_2 = self.client.get_instance_stream("simulation2")
        #Check if streams are alive
        camera_host, camera_port = get_host_port_from_stream_address(stream_address_1)
        camera_host, camera_port = get_host_port_from_stream_address(stream_address_2)
        server_info = self.proxy_client.get_servers_info()
        status_info = self.proxy_client.get_status_info()
        instance_info = self.proxy_client.get_instances_info()
        self.assertEqual(server_info[self.cam_server_address[0]]["load"], 2)
        self.assertEqual(server_info[self.cam_server_address[1]]["load"], 0)

    """
    def test_persisted_config(self):
        stream_address_1 = self.client.get_instance_stream("simulation")
        cfg = self.proxy_client.get_config()
        #Server Config
        self.assertEqual(self.proxy_client.get_config(), {
             "http://localhost:8888": {
              "instances": [
               "simulation"
              ],
              "expanding": True
             }
            })
        cfg["http://localhost:8888"]["expanding"] = False

        self.proxy_client.set_config(cfg)
        self.assertEqual(self.proxy_client.get_config(), {
             "http://localhost:8888": {
              "instances": [
               "simulation"
              ],
              "expanding": False
             }
            })
    """

if __name__ == '__main__':
    unittest.main()
