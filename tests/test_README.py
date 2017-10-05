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

        try:
            os.remove(os.path.join(self.pipeline_config_folder, "testing_config.json"))
        except:
            pass

        try:
            self.cam_client.delete_camera_config("test_camera")
        except:
            pass

        try:
            self.pipeline_client.delete_pipeline_config("test_pipeline")
        except:
            pass

        os.kill(self.cam_process.pid, signal.SIGINT)
        os.kill(self.pipeline_process.pid, signal.SIGINT)

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

    def test_modify_camera_config(self):
        self.cam_client.set_camera_config("test_camera", self.cam_client.get_camera_config("camera_example_2"))
        from cam_server import CamClient

        # Initialize the camera client.
        cam_client = CamClient()
        cam_client = self.cam_client

        # Print the list of available cameras.
        print(cam_client.get_cameras())

        # Put the name of the camera you want to modify.
        camera_to_modify = "test_camera"

        # Retrieve the camera config.
        camera_config = cam_client.get_camera_config(camera_to_modify)

        # Change the mirror_x setting.
        camera_config["mirror_x"] = False
        # Change the camera_calibration setting.
        camera_config["camera_calibration"] = {
            "reference_marker": [0, 0, 100, 100],
            "reference_marker_width": 100.0,
            "reference_marker_height": 100.0,
            "angle_horizontal": 0.0,
            "angle_vertical": 0.0
        }

        # Save the camera configuration.
        cam_client.set_camera_config(camera_to_modify, camera_config)

        # You can also save the same (or another) config under a different camera name.
        cam_client.set_camera_config("camera_to_delete", camera_config)

        # And also delete camera configs.
        cam_client.delete_camera_config("camera_to_delete")

    def test_modify_pipeline_config(self):
        self.pipeline_client.save_pipeline_config("test_pipeline",
                                                  self.pipeline_client.get_pipeline_config("pipeline_example_1"))

        from cam_server import PipelineClient

        # Initialize the pipeline client.
        pipeline_client = PipelineClient()
        pipeline_client = self.pipeline_client

        # Print the list of available pipelines.
        print(pipeline_client.get_pipelines())

        # Put the name of the pipeline you want to modify.
        pipeline_to_modify = "test_pipeline"

        # Retrieve the camera config.
        pipeline_config = pipeline_client.get_pipeline_config(pipeline_to_modify)

        # Change the image threshold.
        pipeline_config["image_threshold"] = 0.5
        # Change the image region of interest.
        pipeline_config["image_region_of_interest"] = [0, 100, 0, 100]

        # Save the camera configuration.
        pipeline_client.save_pipeline_config(pipeline_to_modify, pipeline_config)

        # You can also save the same (or another) config under a different camera name.
        pipeline_client.save_pipeline_config("pipeline_to_delete", pipeline_config)

        # And also delete camera configs.
        pipeline_client.delete_pipeline_config("pipeline_to_delete")

    def test_create_camera(self):

        # Specify the desired camera config.
        camera_config = {
            "name": "camera_example_3",
            "source": "EPICS:CAM1:EXAMPLE",
            "source_type": "epics",
            "mirror_x": False,
            "mirror_y": False,
            "rotate": 0,

            "camera_calibration": {
                "reference_marker": [0, 0, 100, 100],
                "reference_marker_width": 100.0,
                "reference_marker_height": 100.0,
                "angle_horizontal": 0.0,
                "angle_vertical": 0.0
            }
        }

        # Specify the new camera name.
        new_camera_name = "new_camera_name"

        # Save the camera configuration.
        self.cam_client.set_camera_config(new_camera_name, camera_config)

        self.assertTrue(new_camera_name in self.cam_client.get_cameras())

        # Delete the camera config you just added.
        self.cam_client.delete_camera_config(new_camera_name)

        self.assertTrue(new_camera_name not in self.cam_client.get_cameras())

    def get_single_message(self):
        self.pipeline_client.create_instance_from_name("pipeline_example_1", "simulation_sp1")

        # Name of the camera we want to get a message from.
        camera_name = "simulation"
        # Screen panel defines the instance name as: [CAMERA_NAME]_sp1
        instance_name = camera_name + "_sp1"

        # Get the data.
        data = self.pipeline_client.get_instance_message(instance_name)

        self.assertIsNotNone(data)
        self.assertTrue("image" in data.data.data)


if __name__ == '__main__':
    unittest.main()
