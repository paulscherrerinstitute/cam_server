import os
import signal
import unittest

from multiprocessing import Process
from time import sleep


from cam_server import CamClient, PipelineClient
from cam_server.start_camera_server import start_camera_server
from cam_server.start_pipeline_server import start_pipeline_server


class PipelineClientTest(unittest.TestCase):
    def setUp(self):
        self.host = "0.0.0.0"
        self.cam_port = 8888
        self.pipeline_port = 8889

        test_base_dir = os.path.split(os.path.abspath(__file__))[0]
        self.cam_config_folder = os.path.join(test_base_dir, "camera_config/")
        self.pipeline_config_folder = os.path.join(test_base_dir, "pipeline_config/")
        self.background_config_folder = os.path.join(test_base_dir, "background_config/")

        cam_server_address = "http://%s:%s" % (self.host, self.cam_port)
        pipeline_server_address = "http://%s:%s" % (self.host, self.pipeline_port)

        self.cam_process = Process(target=start_camera_server, args=(self.host, self.cam_port,
                                                                     self.cam_config_folder))
        self.cam_process.start()

        self.pipeline_process = Process(target=start_pipeline_server, args=(self.host, self.pipeline_port,
                                                                            self.pipeline_config_folder,
                                                                            self.background_config_folder,
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
            os.remove(os.path.join(self.pipeline_config_folder, "testing_config.json"))
        except:
            pass
        # Wait for the server to die.
        sleep(1)

    def test_get_simulated_camera(self):
        from cam_server import CamClient
        from cam_server.utils import get_host_port_from_stream_address
        from bsread import source, SUB

        # Change to match your camera server
        server_address = "http://0.0.0.0:8888"

        # Initialize the client.
        camera_client = CamClient(server_address)

        # Get stream address of simulation camera. Stream address in format tcp://hostname:port.
        camera_stream_address = camera_client.get_camera_stream("simulation")

        # Extract the stream hostname and port from the stream address.
        camera_host, camera_port = get_host_port_from_stream_address(camera_stream_address)

        # Subscribe to the stream.
        with source(host=camera_host, port=camera_port, mode=SUB) as stream:
            # Receive next message.
            data = stream.receive()

        image_width = data.data.data["width"].value
        image_height = data.data.data["height"].value
        image_bytes = data.data.data["image"].value

        print("Image size: %d x %d" % (image_width, image_height))
        print("Image data: %s" % image_bytes)

        self.assertIsNotNone(data)
        self.assertEqual(image_width, 1280)
        self.assertIsNotNone(image_height, 960)

    def test_get_basic_pipeline_with_simulated_camera(self):

        from cam_server import PipelineClient
        from cam_server.utils import get_host_port_from_stream_address
        from bsread import source, SUB

        # Change to match your pipeline server
        server_address = "http://0.0.0.0:8889"

        # Initialize the client.
        pipeline_client = PipelineClient(server_address)

        # Setup the pipeline config. Use the simulation camera as the pipeline source.
        pipeline_config = {"camera_name": "simulation"}

        # Create a new pipeline with the provided configuration. Stream address in format tcp://hostname:port.
        instance_id, pipeline_stream_address = pipeline_client.create_instance_from_config(pipeline_config)

        # Extract the stream hostname and port from the stream address.
        pipeline_host, pipeline_port = get_host_port_from_stream_address(pipeline_stream_address)

        # Subscribe to the stream.
        with source(host=pipeline_host, port=pipeline_port, mode=SUB) as stream:
            # Receive next message.
            data = stream.receive()

        image_width = data.data.data["width"].value
        image_height = data.data.data["height"].value
        image_bytes = data.data.data["image"].value

        print("Image size: %d x %d" % (image_width, image_height))
        print("Image data: %s" % image_bytes)

        self.assertIsNotNone(data)
        self.assertEqual(image_width, 1280)
        self.assertIsNotNone(image_height, 960)

    def test_create_pipeline_with_background(self):
        from cam_server import PipelineClient
        from cam_server.utils import get_host_port_from_stream_address
        from bsread import source, SUB

        # Change to match your pipeline server
        server_address = "http://0.0.0.0:8889"
        camera_name = "simulation"

        # Initialize the client.
        pipeline_client = PipelineClient(server_address)

        # Collect the background for the given camera.
        background_id = pipeline_client.collect_background(camera_name)

        # Setup the pipeline config. Use the simulation camera as the pipeline source, and the collected background.
        pipeline_config = {"camera_name": camera_name,
                           "background_id": background_id}

        # Create a new pipeline with the provided configuration. Stream address in format tcp://hostname:port.
        instance_id, pipeline_stream_address = pipeline_client.create_instance_from_config(pipeline_config)

        # Extract the stream hostname and port from the stream address.
        pipeline_host, pipeline_port = get_host_port_from_stream_address(pipeline_stream_address)

        # Subscribe to the stream.
        with source(host=pipeline_host, port=pipeline_port, mode=SUB) as stream:
            # Receive next message.
            data = stream.receive()

        image_width = data.data.data["width"].value
        image_height = data.data.data["height"].value
        image_bytes = data.data.data["image"].value

        print("Image size: %d x %d" % (image_width, image_height))
        print("Image data: %s" % image_bytes)

        self.assertIsNotNone(data)
        self.assertEqual(image_width, 1280)
        self.assertIsNotNone(image_height, 960)


if __name__ == '__main__':
    unittest.main()
