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

    def test_client(self):
        expected_pipelines = ["example_1", "example_2", "example_3", "example_4"]
        self.assertListEqual(self.pipeline_client.get_pipelines(), expected_pipelines,
                             "Test config pipelines have changed?")

        camera_config = self.pipeline_client.get_pipeline_config("example_4")
        self.pipeline_client.save_pipeline_config("testing_config", camera_config)

        with self.assertRaisesRegex(ValueError, "Config object cannot be empty"):
            self.pipeline_client.save_pipeline_config("testing_config", {})

        with self.assertRaisesRegex(ValueError, "The following mandatory attributes were not found"):
            self.pipeline_client.save_pipeline_config("testing_config", {"invalid": "config"})

        self.assertListEqual(self.pipeline_client.get_pipelines(), expected_pipelines + ["testing_config"],
                             "Testing config was not added.")

        stream_address_1 = self.pipeline_client.get_instance_stream("testing_config")
        stream_address_2 = self.pipeline_client.get_instance_stream("testing_config")
        self.assertEqual(len(self.pipeline_client.get_server_info()["active_instances"]), 1,
                         "Instance not started or too many instances started.")

        with self.assertRaisesRegex(ValueError, "Config object cannot be empty"):
            self.pipeline_client.set_instance_config("testing_config", {})

        with self.assertRaisesRegex(ValueError, "The following mandatory attributes were not found"):
            self.pipeline_client.set_instance_config("testing_config", {"invalid": "config"})

        with self.assertRaisesRegex(ValueError, "Cannot set config on a read only instance."):
            self.pipeline_client.set_instance_config("testing_config", {"camera_name": "simulation"})

        self.assertEqual(stream_address_1, stream_address_2, "Stream addresses should be equal.")

        self.pipeline_client.stop_instance("testing_config")

        self.assertEqual(len(self.pipeline_client.get_server_info()["active_instances"]), 0,
                         "The instance should be stopped.")

        instance_id_1, instance_stream_1 = self.pipeline_client.create_instance_from_name("testing_config")
        instance_id_2, instance_stream_2 = self.pipeline_client.create_instance_from_name("testing_config")

        # TODO: try to change config of created instance on the fly.
        # TODO: Try to change camera name on created instance.

        self.assertNotEqual(instance_id_1, instance_id_2, "Instances should be different.")
        self.assertNotEqual(instance_stream_1, instance_stream_2, "Stream addresses should be different.")

        self.assertEqual(len(self.pipeline_client.get_server_info()["active_instances"]), 2,
                         "Two instances should be running.")

        instance_stream_3 = self.pipeline_client.get_instance_stream(instance_id_2)
        self.assertEqual(instance_stream_2, instance_stream_3, "Stream addresses should be equal.")

        self.assertEqual(len(self.pipeline_client.get_server_info()["active_instances"]), 2,
                         "Two instances should be running, get does not increase the number of instances.")

        self.pipeline_client.stop_all_instances()

        self.assertEqual(len(self.pipeline_client.get_server_info()["active_instances"]), 0,
                         "All instances should be stopped now.")

        # TODO: Fix this, unknown reason why it does not work.

        # self.pipeline_client.create_instance_from_config({"camera_name": "simulation"})

        # TODO: Try to make an instance from an invalid config.
        #
        # self.assertEqual(len(self.pipeline_client.get_server_info()["active_instances"]), 3,
        #                  "A new instance from config was created.")
        #

        background_id = self.pipeline_client.collect_background("simulation")
        expected_background_file = os.path.join(self.background_config_folder, background_id + ".npy")

        self.assertTrue(os.path.exists(expected_background_file))
        # os.remove(expected_background_file)

        # self.pipeline_client.get_instance_config(instance_id)
        # self.pipeline_client.set_instance_config(instance_id, configuration)
        # self.pipeline_client.get_instance_info(instance_id):
        # self.pipeline_client.collect_background(instance_id):


if __name__ == '__main__':
    unittest.main()
