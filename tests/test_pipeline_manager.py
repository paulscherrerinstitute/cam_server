import os
import signal
import unittest

from multiprocessing import Process
from time import sleep

import numpy
from bsread import source, SUB
from cam_server import CamClient
from cam_server.pipeline.configuration import PipelineConfig, PipelineConfigManager, BackgroundImageManager
from cam_server.pipeline.management import PipelineInstanceManager
from cam_server.start_camera_server import start_camera_server
from cam_server.utils import collect_background, get_host_port_from_stream_address
from tests.helpers.factory import get_test_pipeline_manager, get_test_pipeline_manager_with_real_cam, MockConfigStorage, \
    MockBackgroundManager


class PipelineManagerTest(unittest.TestCase):
    def setUp(self):
        self.host = "0.0.0.0"
        self.port = 8888

        test_base_dir = os.path.split(os.path.abspath(__file__))[0]
        self.config_folder = os.path.join(test_base_dir, "camera_config/")
        self.background_folder = os.path.join(test_base_dir, "background_config/")

        self.process = Process(target=start_camera_server, args=(self.host, self.port, self.config_folder))
        self.process.start()

        # Give it some time to start.
        sleep(0.5)

        server_address = "http://%s:%s" % (self.host, self.port)
        self.client = CamClient(server_address)

    def tearDown(self):
        self.client.stop_all_cameras()
        try:
            os.kill(self.process.pid, signal.SIGINT)
        except:
            pass
        try:
            os.remove(os.path.join(self.config_folder, "testing_camera.json"))
        except:
            pass

        try:
            os.remove(os.path.join(self.background_folder, "white_background.npy"))
        except:
            pass
        # Wait for the server to die.
        sleep(1)

    def test_get_pipeline_list(self):
        pipeline_manager = get_test_pipeline_manager()
        self.assertEqual(len(pipeline_manager.get_pipeline_list()), 0, "Pipeline manager should be empty by default.")

        initial_config = {"test_pipeline1": {},
                          "test_pipeline2": {}}

        pipeline_manager.config_manager.config_provider.configs = initial_config

        self.assertListEqual(sorted(list(initial_config.keys())), sorted(pipeline_manager.get_pipeline_list()),
                             "Set and received lists are not the same.")

    def test_create_pipeline_instance(self):
        instance_manager = get_test_pipeline_manager_with_real_cam()

        pipeline_config = {
            "camera_name": "simulation"
        }

        instance_manager.config_manager.save_pipeline_config("test_pipeline", pipeline_config)

        pipeline_id_1, stream_address_1 = instance_manager.create_pipeline("test_pipeline")
        pipeline_id_2, stream_address_2 = instance_manager.create_pipeline("test_pipeline")

        self.assertNotEqual(pipeline_id_1, pipeline_id_2, "The pipeline ids should be different.")
        self.assertNotEqual(stream_address_1, stream_address_2, "The pipeline streams should be different.")

        self.assertFalse(instance_manager.get_info()["active_instances"][pipeline_id_1]["read_only"],
                         "Instance should not be read only.")
        self.assertFalse(instance_manager.get_info()["active_instances"][pipeline_id_2]["read_only"],
                         "Instance should not be read only.")

        pipeline_id_3, stream_address_3 = instance_manager.create_pipeline(configuration=pipeline_config)

        self.assertNotEqual(pipeline_id_2, pipeline_id_3, "The pipeline ids should be different.")
        self.assertNotEqual(stream_address_2, stream_address_3, "The pipeline streams should be different.")

        self.assertFalse(instance_manager.get_info()["active_instances"][pipeline_id_3]["read_only"],
                         "Instance should not be read only.")

        with self.assertRaisesRegex(ValueError, "You must specify either the pipeline name or the "
                                                "configuration for the pipeline."):
            instance_manager.create_pipeline(pipeline_name="test_pipeline", configuration=pipeline_config)

        instance_manager.stop_all_instances()

    def test_get_instance_stream(self):
        instance_manager = get_test_pipeline_manager_with_real_cam()
        instance_manager.stop_all_instances()

        self.assertTrue(len(instance_manager.get_info()["active_instances"]) == 0,
                        "There should be no running instances.")

        pipeline_id = "test_pipeline"
        pipeline_config = PipelineConfig(pipeline_id, parameters={
            "camera_name": "simulation"
        })

        instance_manager.config_manager.save_pipeline_config(pipeline_id, pipeline_config.get_configuration())
        instance_stream_1 = instance_manager.get_instance_stream(pipeline_id)

        self.assertTrue(instance_manager.get_info()["active_instances"][pipeline_id]["read_only"],
                        "Instance should be read only.")

        with self.assertRaisesRegex(ValueError, "Cannot set config on a read only instance."):
            instance_manager.get_instance(pipeline_id).set_parameter({})

        instance_stream_2 = instance_manager.get_instance_stream(pipeline_id)

        self.assertEqual(instance_stream_1, instance_stream_2, "Only one instance should be present.")

        instance_manager.stop_all_instances()

    def test_collect_background(self):
        instance_manager = get_test_pipeline_manager_with_real_cam()

        pipeline_id = "test_pipeline"
        number_of_images = 10

        pipeline_config = PipelineConfig(pipeline_id, parameters={
            "camera_name": "simulation"
        })

        instance_manager.config_manager.save_pipeline_config(pipeline_id, pipeline_config.get_configuration())

        pipeline_stream_address = instance_manager.get_instance_stream(pipeline_id)
        pipeline_host, pipeline_port = get_host_port_from_stream_address(pipeline_stream_address)

        # Collect from the pipeline.
        with source(host=pipeline_host, port=pipeline_port, mode=SUB) as stream:
            data = stream.receive()
            self.assertIsNotNone(data, "This should really not happen anymore.")

        camera_name = instance_manager.get_instance(pipeline_id).get_info()["camera_name"]
        background_id = collect_background(camera_name, pipeline_stream_address, number_of_images,
                                           instance_manager.background_manager)

        self.assertTrue(background_id.startswith("simulation"), "Background id not as expected.")

        host, port = get_host_port_from_stream_address(instance_manager.
                                                       cam_server_client.get_camera_stream("simulation"))

        # Collect from the camera.
        with source(host=host, port=port, mode=SUB) as stream:
            data = stream.receive()
            self.assertIsNotNone(data, "This should really not happen anymore.")

        self.assertEqual(instance_manager.background_manager.get_background(background_id).shape,
                         data.data.data["image"].value.shape,
                         "Background and image have to be of the same shape.")

        instance_manager.stop_all_instances()

    def test_custom_hostname(self):
        config_manager = PipelineConfigManager(config_provider=MockConfigStorage())
        pipeline_instance_manager = PipelineInstanceManager(config_manager, MockBackgroundManager(),
                                                            CamClient("http://0.0.0.0:8888"),
                                                            hostname="custom_cam_hostname")

        _, stream_address = pipeline_instance_manager.create_pipeline(configuration={"camera_name": "simulation"})

        self.assertTrue(stream_address.startswith("tcp://custom_cam_hostname"))

        pipeline_instance_manager.stop_all_instances()

    def test_update_instance_config_without_running(self):
        pipeline_manager = get_test_pipeline_manager()

        instance_id, _ = pipeline_manager.create_pipeline(configuration={"camera_name": "simulation"})

        config_updates = {
            "camera_name": "different_name",
        }

        with self.assertRaisesRegex(ValueError, "Cannot change the camera name on a running instance. "
                                                "Stop the instance first."):
            pipeline_manager.update_instance_config(instance_id, config_updates)

        config_updates = {
            "camera_name": "simulation",
            "camera_calibration": {
                "reference_marker": [1, 2, 3, 4],
                "reference_marker_width": 5.0,
                "reference_marker_height": 6.0,
                "angle_horizontal": 7.0,
                "angle_vertical": 8.0
            },
            "image_background": None,
            "image_background_enable": False,
            "image_threshold": 2,
            "image_region_of_interest": [3, 4, 5, 6],
            "image_good_region": {
                "threshold": 0.9,
                "gfscale": 3.6
            },
            "image_slices": {
                "number_of_slices": 6,
                "scale": 7
            }
        }

        pipeline_manager.update_instance_config(instance_id, config_updates)

        self.assertDictEqual(pipeline_manager.get_instance(instance_id).get_configuration(),
                             config_updates, "Update was not successful.")

        self.assertDictEqual(pipeline_manager.get_instance(instance_id).get_configuration()["camera_calibration"],
                             config_updates["camera_calibration"], "Update was not successful.")

        self.assertDictEqual(pipeline_manager.get_instance(instance_id).get_configuration()["image_good_region"],
                             config_updates["image_good_region"], "Update was not successful.")

        self.assertDictEqual(pipeline_manager.get_instance(instance_id).get_configuration()["image_slices"],
                             config_updates["image_slices"], "Update was not successful.")

        with self.assertRaisesRegex(ValueError, "Requested background 'non_existing' does not exist."):
            pipeline_manager.update_instance_config(instance_id, {"image_background": "non_existing"})

        pipeline_manager.background_manager.save_background("non_existing", None, append_timestamp=False)
        pipeline_manager.update_instance_config(instance_id, {"image_background": "non_existing"})

        self.assertEqual(pipeline_manager.get_instance(instance_id).get_configuration()["image_background"],
                         "non_existing", "Background not updated.")

        pipeline_manager.update_instance_config(instance_id, {"image_background": None})
        self.assertIsNone(pipeline_manager.get_instance(instance_id).get_configuration()["image_background"],
                          "Background should be None.")

        pipeline_manager.update_instance_config(instance_id, {"image_slices": None})
        self.assertIsNone(pipeline_manager.get_instance(instance_id).get_configuration()["image_slices"],
                          "Sub dictionary not set to None.")

        pipeline_manager.update_instance_config(instance_id, {"image_good_region": {"gfscale": None}})
        self.assertEqual(pipeline_manager.get_instance(instance_id).get_configuration()["image_good_region"]["gfscale"],
                         PipelineConfig.DEFAULT_IMAGE_GOOD_REGION["gfscale"],
                         "Default value not set correctly.")

        pipeline_manager.stop_all_instances()

    def test_update_instance_config_with_running(self):
        instance_manager = get_test_pipeline_manager_with_real_cam()
        instance_manager.background_manager = BackgroundImageManager(self.background_folder)

        pipeline_config = {
            "camera_name": "simulation"
        }

        black_image = numpy.zeros(shape=(960, 1280))

        pipeline_id, pipeline_stream_address = instance_manager.create_pipeline(configuration=pipeline_config)
        pipeline_host, pipeline_port = get_host_port_from_stream_address(pipeline_stream_address)

        # Collect from the pipeline.
        with source(host=pipeline_host, port=pipeline_port, mode=SUB) as stream:
            normal_image_data = stream.receive()
            self.assertIsNotNone(normal_image_data, "This should really not happen anymore.")
            self.assertFalse(numpy.array_equal(normal_image_data.data.data['image'].value, black_image),
                             "There should be a non black image.")

            self.assertEqual(normal_image_data.data.data["width"].value, 1280, "Incorrect width.")
            self.assertEqual(normal_image_data.data.data["height"].value, 960, "Incorrect height.")

        white_image = numpy.zeros(shape=(960, 1280))
        white_image.fill(99999)

        instance_manager.background_manager.save_background("white_background", white_image, append_timestamp=False)

        instance_manager.update_instance_config(pipeline_id, {"image_background": "white_background"})

        # Give it some time to load the background.
        sleep(0.5)

        # Collect from the camera.
        with source(host=pipeline_host, port=pipeline_port, mode=SUB) as stream:
            image_data = stream.receive()
            self.assertIsNotNone(image_data, "This should really not happen anymore.")

        # Because we didn't set the "image_background_enable" yet.
        self.assertFalse(numpy.array_equal(image_data.data.data["image"].value, black_image),
                         "Background should not be yet applied.")

        instance_manager.update_instance_config(pipeline_id, {"image_background_enable": True})

        # Give it some time to load the background.
        sleep(0.5)

        # Collect from the camera.
        with source(host=pipeline_host, port=pipeline_port, mode=SUB) as stream:
            black_image_data = stream.receive()
            self.assertIsNotNone(black_image_data, "This should really not happen anymore.")

        self.assertTrue(numpy.array_equal(black_image_data.data.data["image"].value, black_image),
                        "Now background should work.")

        instance_manager.stop_all_instances()

    def test_delete_pipeline(self):
        instance_manager = get_test_pipeline_manager()
        self.assertEqual(len(instance_manager.get_pipeline_list()), 0, "Pipeline should be empty")

        instance_manager.config_manager.save_pipeline_config("test", {"camera_name": "simulation"})
        self.assertEqual(len(instance_manager.get_pipeline_list()), 1, "Pipeline should not be empty")

        instance_manager.config_manager.delete_pipeline_config("test")
        self.assertEqual(len(instance_manager.get_pipeline_list()), 0, "Pipeline should be empty")

    def test_custom_instance_id(self):
        instance_manager = get_test_pipeline_manager()
        instance_manager.config_manager.save_pipeline_config("test_pipeline", {"camera_name": "simulation"})

        instance_id, stream_address = instance_manager.create_pipeline("test_pipeline", instance_id="custom_instance")

        self.assertEqual(instance_id, "custom_instance", "Custom instance name was not set.")

        with self.assertRaisesRegex(ValueError, "Instance with id 'custom_instance' is already present and running. "
                                                "Use another instance_id or stop the current instance "
                                                "if you want to reuse the same instance_id."):
            instance_manager.create_pipeline("test_pipeline", instance_id="custom_instance")

        instance_manager.stop_instance("custom_instance")

        instance_id, stream_address = instance_manager.create_pipeline(configuration={"camera_name": "simulation"},
                                                                       instance_id="custom_instance")

        self.assertEqual(instance_id, "custom_instance", "Custom instance name was not set.")

    def test_reload_after_stop(self):
        instance_manager = get_test_pipeline_manager()

        instance_id = "trying_to_test"

        initial_config = {"camera_name": "simulation"}

        instance_manager.config_manager.save_pipeline_config(instance_id, initial_config)
        instance_manager.get_instance_stream(instance_id)

        self.assertEqual(instance_manager.get_instance(instance_id).get_configuration()["image_threshold"],
                         None)

        updated_config = {"camera_name": "simulation",
                          "threshold": 9999}

        instance_manager.config_manager.save_pipeline_config("test_pipeline", updated_config)

        self.assertEqual(instance_manager.get_instance(instance_id).get_configuration()["image_threshold"],
                         None, "It should not change at this point.")

        instance_manager.stop_instance(instance_id)
        instance_manager.get_instance_stream(instance_id)

        # self.assertEqual(instance_manager.get_instance(instance_id).get_configuration()["image_threshold"],
        #                  9999, "It should have changed now - reload should happen.")

        instance_manager.stop_all_instances()


if __name__ == '__main__':
    unittest.main()
