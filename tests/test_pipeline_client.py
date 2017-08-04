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

        with self.assertRaisesRegex(ValueError, "Config updates cannot be empty."):
            self.pipeline_client.set_instance_config("testing_config", {})

        with self.assertRaisesRegex(ValueError, "Cannot set config on a read only instance."):
            self.pipeline_client.set_instance_config("testing_config", {"camera_name": "simulation"})

        self.assertEqual(stream_address_1, stream_address_2, "Stream addresses should be equal.")

        self.pipeline_client.stop_instance("testing_config")

        self.assertEqual(len(self.pipeline_client.get_server_info()["active_instances"]), 0,
                         "The instance should be stopped.")

        instance_id_1, instance_stream_1 = self.pipeline_client.create_instance_from_name("testing_config")
        instance_id_2, instance_stream_2 = self.pipeline_client.create_instance_from_name("testing_config")

        # TODO: try to change config of created instance on the fly.

        with self.assertRaisesRegex(ValueError, "Cannot change the camera name on a running instance. "
                                                "Stop the instance first."):
            self.pipeline_client.set_instance_config(instance_id_1, {"camera_name": "different_camera"})

        self.assertNotEqual(instance_id_1, instance_id_2, "Instances should be different.")
        self.assertNotEqual(instance_stream_1, instance_stream_2, "Stream addresses should be different.")

        self.assertEqual(len(self.pipeline_client.get_server_info()["active_instances"]), 2,
                         "Two instances should be running.")

        instance_stream_3 = self.pipeline_client.get_instance_stream(instance_id_2)
        self.assertEqual(instance_stream_2, instance_stream_3, "Stream addresses should be equal.")

        self.assertEqual(len(self.pipeline_client.get_server_info()["active_instances"]), 2,
                         "Two instances should be running, get does not increase the number of instances.")

        self.pipeline_client.create_instance_from_config({"camera_name": "simulation"})
        self.pipeline_client.create_instance_from_config({"camera_name": "simulation"})

        self.assertEqual(len(self.pipeline_client.get_server_info()["active_instances"]), 4,
                         "Two new instances should be created.")

        with self.assertRaisesRegex(ValueError, "You must specify either the pipeline name or the "
                                                "configuration for the pipeline."):
            self.pipeline_client.create_instance_from_config({})

        with self.assertRaisesRegex(ValueError, "The following mandatory attributes were not"):
            self.pipeline_client.create_instance_from_config({"invalid": "config"})

        background_id = self.pipeline_client.collect_background("simulation")
        expected_background_file = os.path.join(self.background_config_folder, background_id + ".npy")

        pipeline_config = {"camera_name": "simulation",
                           "background_image": background_id}

        instance_id, stream_address = self.pipeline_client.create_instance_from_config(pipeline_config)

        instance_config = self.pipeline_client.get_instance_config(instance_id)
        self.assertDictEqual(pipeline_config, instance_config, "Set and retrieved instances are not equal.")

        instance_info = self.pipeline_client.get_instance_info(instance_id)
        self.assertEqual(instance_info["instance_id"], instance_id, "Requested and retireved instances are different.")
        self.assertEqual(instance_info["stream_address"], stream_address, "Different stream address.")
        self.assertTrue(instance_info["is_stream_active"], "Stream should be active.")
        self.assertFalse(instance_info["read_only"], "It should not be read only.")
        self.assertEqual(instance_info["camera_name"], "simulation", "Wrong camera name.")
        self.assertDictEqual(instance_info["config"], pipeline_config, "Config is not equal")

        self.pipeline_client.stop_instance(instance_id)

        stopped_instance_info = self.pipeline_client.get_instance_info(instance_id)
        self.assertFalse(stopped_instance_info["is_stream_active"], "Stream should not be active.")

        with self.assertRaisesRegex(ValueError, "Instance 'invalid instance' does not exist."):
            self.pipeline_client.get_instance_info("invalid instance")

        self.assertTrue(os.path.exists(expected_background_file))
        os.remove(expected_background_file)

        self.pipeline_client.stop_all_instances()

if __name__ == '__main__':
    unittest.main()
