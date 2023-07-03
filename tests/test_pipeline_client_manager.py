import base64
import os
import unittest
import time

from multiprocessing import Process
from time import sleep

import numpy

from bsread import source, SUB, PULL

from cam_server import CamClient, PipelineClient
from cam_server import config
from cam_server.pipeline.configuration import PipelineConfig
from cam_server.start_camera_worker import start_camera_worker
from cam_server.start_pipeline_worker import start_pipeline_worker
from cam_server.start_camera_manager import start_camera_manager
from cam_server.start_pipeline_manager import start_pipeline_manager
from cam_server.utils import get_host_port_from_stream_address
from tests import test_cleanup, is_port_available, require_folder, get_simulated_camera


class PipelineClientTest(unittest.TestCase):
    def setUp(self):
        self.host = "0.0.0.0"
        self.cam_port = 8880
        self.pipeline_port = 8881
        self.cam_manager_port = 8888
        self.pipeline_manager_port = 8889

        for port in self.cam_port, self.pipeline_port, self.cam_manager_port, self.pipeline_manager_port:
            print("Port ", port, "avilable: ", is_port_available(port))

        test_base_dir = os.path.split(os.path.abspath(__file__))[0]
        self.cam_config_folder = os.path.join(test_base_dir, "camera_config/")
        self.pipeline_config_folder = os.path.join(test_base_dir, "pipeline_config/")
        self.background_config_folder = os.path.join(test_base_dir, "background_config/")
        self.user_scripts_folder = os.path.join(test_base_dir, "user_scripts/")
        self.temp_folder = os.path.join(test_base_dir, "temp/")

        require_folder(self.background_config_folder)
        require_folder(self.user_scripts_folder)
        require_folder(self.temp_folder)

        cam_server_address = "http://%s:%s" % (self.host, self.cam_port)
        pipeline_server_address = "http://%s:%s" % (self.host, self.pipeline_port)
        cam_server_proxy_address = "http://%s:%s" % (self.host, self.cam_manager_port)
        pipeline_server_proxy_address = "http://%s:%s" % (self.host, self.pipeline_manager_port)

        self.cam_process = Process(target=start_camera_worker, args=(self.host, self.cam_port, self.user_scripts_folder))
        self.cam_process.start()

        self.pipeline_process = Process(target=start_pipeline_worker, args=(self.host, self.pipeline_port,
                                                                            self.temp_folder,
                                                                            self.temp_folder,
                                                                            cam_server_proxy_address))
        self.pipeline_process.start()

        self.cam_proxy_process =Process(target=start_camera_manager, args=(self.host, self.cam_manager_port,
                                                cam_server_address, self.cam_config_folder, self.user_scripts_folder))
        self.cam_proxy_process.start()

        self.pipeline_proxy_process = Process(target=start_pipeline_manager, args=(self.host, self.pipeline_manager_port,
                                                                    pipeline_server_address,
                                                                    self.pipeline_config_folder,
                                                                    self.background_config_folder,
                                                                    config.DEFAULT_BACKGROUND_FILES_DAYS_TO_LIVE,
                                                                    self.user_scripts_folder,
                                                                    cam_server_proxy_address))
        self.pipeline_proxy_process.start()
        self.cam_client = CamClient(cam_server_proxy_address)
        self.pipeline_client = PipelineClient(pipeline_server_proxy_address)
        self.pipeline_client.set_function_script(instance_id, filename)

        # Give it some time to start.
        sleep(1.0)  # Give it some time to start.

    def tearDown(self):
        test_cleanup(
            [self.pipeline_client, self.cam_client],
             [self.pipeline_proxy_process, self.cam_proxy_process, self.pipeline_process, self.cam_process],
             [
                 os.path.join(self.pipeline_config_folder, "testing_config.json"),
                 os.path.join(self.temp_folder, "Test.py"),
                 os.path.join(self.temp_folder, "Test2.py"),
                 os.path.join(self.user_scripts_folder, "Test.py"),
                 os.path.join(self.user_scripts_folder, "Test2.py"),
                 self.background_config_folder,
                 self.user_scripts_folder,
                 self.temp_folder,
             ])

    def test_client(self):
        expected_pipelines = ["pipeline_example_1", "pipeline_example_2", "pipeline_example_3",
                                  "pipeline_example_4"]
        for pipeline in  expected_pipelines:
            self.assertIn(pipeline, self.pipeline_client.get_pipelines(),
                            "Test config pipelines have changed?")
        expected_pipelines = set(self.pipeline_client.get_pipelines())

        camera_config = self.pipeline_client.get_pipeline_config("pipeline_example_4")
        self.pipeline_client.save_pipeline_config("testing_config", camera_config)

        with self.assertRaisesRegex(ValueError, "Camera name not specified in configuration."):
            self.pipeline_client.save_pipeline_config("testing_config", {})

        with self.assertRaisesRegex(ValueError, "Camera name not specified in configuration."):
            self.pipeline_client.save_pipeline_config("testing_config", {"invalid": "config"})

        with self.assertRaisesRegex(ValueError, "pipeline_type 'invalid' not present in mapping. Available"):
            self.pipeline_client.save_pipeline_config("testing_config", {"camera_name": "simulation",
                                                                         "pipeline_type": "invalid"})

        expected_pipelines.add("testing_config")

        self.assertSetEqual(set(self.pipeline_client.get_pipelines()), expected_pipelines,
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
        self.assertTrue("statistics" in instance_info)

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

        self.assertTrue(self.pipeline_client.is_instance_running("custom_instance"),
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
                            'simulation', 'simulation2']

        #self.assertEqual(set(self.pipeline_client.get_cameras()), set(expected_cameras),
        #                 "Expected cameras not present.")
        for camera in  expected_cameras:
            self.assertIn(camera, set(self.pipeline_client.get_cameras()), "Not getting expected camera: " + camera)

        configuration = {"camera_name": "simulation",
                         "threshold": 50}

        stream_address_1 = self.pipeline_client.get_instance_stream_from_config(configuration)
        stream_address_2 = self.pipeline_client.get_instance_stream_from_config(configuration)

        self.assertEqual(stream_address_1, stream_address_2,
                         "Requesting the same config should give you the same instance.")

        self.pipeline_client.stop_all_instances()
        self.cam_client.stop_all_instances()

        instance_id, stream_address = self.pipeline_client.create_instance_from_config(
            {"camera_name": "simulation", "pipeline_type": "store", 'stream_port':10000})

        with self.assertRaisesRegex(ValueError, "Cannot get message from 'store' pipeline type."):
            self.pipeline_client.get_instance_message(instance_id)

        host, port = get_host_port_from_stream_address(stream_address)

        with source(host=host, port=port, mode=PULL) as stream:
            data = stream.receive()

        self.assertIsNotNone(data)
        self.assertEqual(len(data.data.data), 1, "Only the image should be present in the received data.")
        self.assertTrue("simulation"+config.EPICS_PV_SUFFIX_IMAGE in data.data.data,
                        "Camera name should be used instead of 'image'.")

        self.pipeline_client.stop_all_instances()

        #Transparent pipeline
        instance_id, instance_stream = self.pipeline_client.create_instance_from_name("simulation")
        cfg = self.pipeline_client.get_instance_config(instance_id)
        cfg["function"] = "transparent"
        self.pipeline_client.set_instance_config(instance_id, cfg)
        time.sleep(.5)
        data = self.pipeline_client.get_instance_message(instance_id)
        self.assertIsNotNone(data)
        # Cam_server fields + processing_parameters
        required_fields = set(["image", "timestamp", "width", "height", "x_axis", "y_axis", "processing_parameters"])
        self.assertSetEqual(required_fields, set(data.data.data.keys()), "Bad transparent pipeline fields.")

        self.pipeline_client.stop_all_instances()
        time.sleep(1.0)


    def test_background_file(self):

        bg = self.pipeline_client.get_latest_background("simulation")
        image = self.pipeline_client.get_background_image(bg)
        self.assertGreater(len(image.content), 0)

        image = self.pipeline_client.get_background_image_bytes(bg)
        dtype = image["dtype"]
        shape = image["shape"]
        bytes = base64.b64decode(image["bytes"].encode())

        x_size, y_size = get_simulated_camera().get_geometry()
        self.assertEqual(shape, [y_size, x_size])

        image_array = numpy.frombuffer(bytes, dtype=dtype).reshape(shape)
        self.assertIsNotNone(image_array)

    def test_user_scripts(self):
        script_name = "Test.py"
        script_content = "print('Hello world')"
        self.pipeline_client.set_user_script(script_name, script_content)

        scripts = self.pipeline_client.get_user_scripts()
        self.assertIn(script_name, scripts)

        ret = self.pipeline_client.get_user_script(script_name)
        self.assertEqual(ret, script_content)
        filename="temp/Test2.py"
        with open(filename, "w") as data_file:
            data_file.write(script_content)

        self.pipeline_client.upload_user_script(filename)
        os.remove(filename)
        self.pipeline_client.download_user_script(filename)
        with open(filename, "r") as data_file:
            ret= data_file.read()
        self.assertEqual(ret, script_content)

        script_content = """
from cam_server.pipeline.data_processing import functions, processor
def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata):
    ret = processor.process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata)
    ret["average_value"] = float(ret["intensity"]) / len(ret["x_axis"]) / len(ret["y_axis"])
    return ret
        """
        filename="temp/Test.py"
        with open(filename, "w") as data_file:
            data_file.write(script_content)

        instance_id, stream_address = self.pipeline_client.create_instance_from_config({"camera_name": "simulation"})
        host, port = get_host_port_from_stream_address(stream_address)

        with source(host=host, port=port, mode=SUB) as stream:
            data = stream.receive()
            self.assertIsNotNone(data, "This should really not happen anymore.")
            self.assertIsNotNone(data.data.data["width"].value)
            self.assertIsNotNone(data.data.data["height"].value)

        self.pipeline_client.set_function_script(instance_id, filename)
        time.sleep(1.0)
        with source(host=host, port=port, mode=SUB) as stream:
            data = stream.receive()
            print (data.data.data.keys())
            self.assertIsNotNone(data.data.data["average_value"].value)

if __name__ == '__main__':
    unittest.main()
