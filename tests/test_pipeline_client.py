import os
import signal
import unittest

from multiprocessing import Process
from time import sleep

import numpy
from bsread import source, SUB, PULL

from cam_server import CamClient, PipelineClient
from cam_server.pipeline.configuration import PipelineConfig
from cam_server.start_camera_server import start_camera_server
from cam_server.start_pipeline_server import start_pipeline_server
from cam_server.utils import get_host_port_from_stream_address


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
        expected_pipelines = ["pipeline_example_1", "pipeline_example_2", "pipeline_example_3", "pipeline_example_4"]
        self.assertListEqual(self.pipeline_client.get_pipelines(), expected_pipelines,
                             "Test config pipelines have changed?")

        camera_config = self.pipeline_client.get_pipeline_config("pipeline_example_4")
        self.pipeline_client.save_pipeline_config("testing_config", camera_config)

        with self.assertRaisesRegex(ValueError, "Camera name not specified in configuration."):
            self.pipeline_client.save_pipeline_config("testing_config", {})

        with self.assertRaisesRegex(ValueError, "Camera name not specified in configuration."):
            self.pipeline_client.save_pipeline_config("testing_config", {"invalid": "config"})

        with self.assertRaisesRegex(ValueError, "pipeline_type 'invalid' not present in mapping. Available"):
            self.pipeline_client.save_pipeline_config("testing_config", {"camera_name": "simulation",
                                                                         "pipeline_type": "invalid"})

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

        instance_info = self.pipeline_client.get_instance_info(instance_id_1)
        self.assertTrue("last_start_time" in instance_info)

        with self.assertRaisesRegex(ValueError, "Cannot change the camera name on a running instance. "
                                                "Stop the instance first."):
            self.pipeline_client.set_instance_config(instance_id_1, {"camera_name": "different_camera"})

        self.pipeline_client.set_instance_config(instance_id_2, {"image_threshold": 99999,
                                                                 "image_region_of_interest": [0, 200, 0, 200]})

        # Wait for the config update.
        sleep(0.5)

        pipeline_host, pipeline_port = get_host_port_from_stream_address(instance_stream_2)

        with source(host=pipeline_host, port=pipeline_port, mode=SUB) as stream:
            data = stream.receive()
            self.assertIsNotNone(data, "This should really not happen anymore.")
            # shape 200, 200 -> Account for the region of interest change.
            self.assertTrue(numpy.array_equal(data.data.data["image"].value, numpy.zeros(shape=(200, 200))),
                            "Array should be all zeros, because of the threshold config.")

            # Adjust width and height with the region of interest.
            self.assertEqual(data.data.data["width"].value, 200, "Region of interest not takes into account.")
            self.assertEqual(data.data.data["height"].value, 200, "Region of interest not takes into account.")

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

        with self.assertRaisesRegex(ValueError, "Camera name not specified in configuration."):
            self.pipeline_client.create_instance_from_config({"invalid": "config"})

        background_id = self.pipeline_client.collect_background("simulation")
        expected_background_file = os.path.join(self.background_config_folder, background_id + ".npy")

        # Unfortunately there is not way of knowing (from client side) how many images were collected.
        background_id_2 = self.pipeline_client.collect_background("simulation", 5)

        self.assertNotEqual(background_id, background_id_2, "Background should be different.")

        with self.assertRaisesRegex(Exception, "n_images must be a number."):
            self.pipeline_client.collect_background("simulation", "invalid_number")

        pipeline_config = {"camera_name": "simulation",
                           "background_image": background_id}

        instance_id, stream_address = self.pipeline_client.create_instance_from_config(pipeline_config)

        instance_config = self.pipeline_client.get_instance_config(instance_id)

        # We need to account for the expended config.
        expected_config = {}
        expected_config.update(PipelineConfig.DEFAULT_CONFIGURATION)
        expected_config.update(pipeline_config)

        self.assertDictEqual(expected_config, instance_config, "Set and retrieved instances are not equal.")

        instance_info = self.pipeline_client.get_instance_info(instance_id)
        self.assertEqual(instance_info["instance_id"], instance_id, "Requested and retireved instances are different.")
        self.assertEqual(instance_info["stream_address"], stream_address, "Different stream address.")
        self.assertTrue(instance_info["is_stream_active"], "Stream should be active.")
        self.assertFalse(instance_info["read_only"], "It should not be read only.")
        self.assertEqual(instance_info["camera_name"], "simulation", "Wrong camera name.")
        self.assertDictEqual(instance_info["config"], expected_config, "Config is not equal")

        self.pipeline_client.stop_instance(instance_id)

        with self.assertRaisesRegex(ValueError, "Instance '%s' does not exist." % instance_id):
            self.pipeline_client.get_instance_info(instance_id)

        with self.assertRaisesRegex(ValueError, "Instance 'invalid instance' does not exist."):
            self.pipeline_client.get_instance_info("invalid instance")

        self.assertTrue(os.path.exists(expected_background_file))
        os.remove(expected_background_file)

        self.pipeline_client.stop_all_instances()

        self.assertTrue("testing_config" in self.pipeline_client.get_pipelines(),
                        "Pre requirement for next test.")

        self.pipeline_client.delete_pipeline_config("testing_config")
        self.assertFalse("testing_config" in self.pipeline_client.get_pipelines(),
                         "Pipeline should not exist anymore.")

        instance_id, stream_address = self.pipeline_client.create_instance_from_config(
            {"camera_name": "simulation",
             "image_threshold": 10}, "custom_instance"
        )

        self.assertEqual(instance_id, "custom_instance", "Custom instance not set properly.")

        self.assertTrue("custom_instance" in self.pipeline_client.get_server_info()["active_instances"],
                        "Instance with custom instance id not present.")

        self.assertEqual(self.pipeline_client.get_instance_config("custom_instance")["image_threshold"], 10,
                         "Config not set on custom instance id.")

        with self.assertRaisesRegex(ValueError, "Instance with id 'custom_instance' is already present and running. "
                                                "Use another instance_id or stop the current instance if you want "
                                                "to reuse the same instance_id."):
            self.pipeline_client.create_instance_from_config(
                {"camera_name": "simulation"}, "custom_instance"
            )

        self.pipeline_client.stop_instance("custom_instance")

        # The instance is now stopped, it should overwrite it.
        self.pipeline_client.create_instance_from_config(
            {"camera_name": "simulation",
             "image_threshold": 20}, "custom_instance"
        )

        self.assertEqual(self.pipeline_client.get_instance_config("custom_instance")["image_threshold"], 20,
                         "Instance with custom id not overwritten.")

        self.pipeline_client.save_pipeline_config("testing_config", {"camera_name": "simulation",
                                                                     "image_threshold": 30})

        with self.assertRaisesRegex(ValueError, "Instance with id 'custom_instance' is already present and running. "
                                                "Use another instance_id or stop the current instance if you want "
                                                "to reuse the same instance_id."):
            self.pipeline_client.create_instance_from_name("testing_config", "custom_instance")

        self.assertEqual(self.pipeline_client.get_instance_config("custom_instance")["image_threshold"], 20,
                         "Instance should not have changed.")

        data = self.pipeline_client.get_instance_message("custom_instance")
        self.assertIsNotNone(data)
        self.assertTrue("image" in data.data.data)

        self.pipeline_client.stop_instance("custom_instance")

        self.pipeline_client.create_instance_from_name("testing_config", "custom_instance")

        self.assertEqual(self.pipeline_client.get_instance_config("custom_instance")["image_threshold"], 30,
                         "Instance should have changed.")

        background_1 = self.pipeline_client.collect_background("simulation")
        background_2 = self.pipeline_client.collect_background("simulation")

        self.assertNotEqual(background_1, background_2)

        latest_background = self.pipeline_client.get_latest_background("simulation")

        self.assertEqual(latest_background, background_2, "Wrong background set as latest.")

        with self.assertRaisesRegex(ValueError, "No background matches for the specified prefix 'does not exist'."):
            self.pipeline_client.get_latest_background("does not exist")

        expected_cameras = ['camera_example_1', 'camera_example_3', 'camera_example_2', 'camera_example_4',
                            'simulation']

        self.assertEqual(set(self.pipeline_client.get_cameras()), set(expected_cameras),
                         "Expected cameras not present.")

        self.pipeline_client.stop_all_instances()

    def test_store_pipeline(self):
        instance_id, stream_address = self.pipeline_client.create_instance_from_config(
            {"camera_name": "simulation", "pipeline_type": "store"})

        with self.assertRaisesRegex(ValueError, "Cannot get message from 'store' pipeline type."):
            self.pipeline_client.get_instance_message(instance_id)

        host, port = get_host_port_from_stream_address(stream_address)

        with source(host=host, port=port, mode=PULL) as stream:
            data = stream.receive()

        self.assertIsNotNone(data)

        self.pipeline_client.stop_all_instances()


if __name__ == '__main__':
    unittest.main()
