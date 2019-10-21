import os
import signal
import unittest

from multiprocessing import Process
from time import sleep

import numpy
from bsread import source, SUB
from cam_server import CamClient, config
from cam_server.pipeline.configuration import PipelineConfig, PipelineConfigManager, BackgroundImageManager
from cam_server.pipeline.management import PipelineInstanceManager
from cam_server.start_camera_server import start_camera_server
from cam_server.utils import get_host_port_from_stream_address
from tests.helpers.factory import get_test_pipeline_manager, get_test_pipeline_manager_with_real_cam, \
    MockConfigStorage, MockBackgroundManager, MockCamServerClient
from tests import test_cleanup, require_folder


class PipelineManagerTest(unittest.TestCase):
    def setUp(self):
        self.host = "0.0.0.0"
        self.port = 8888

        test_base_dir = os.path.split(os.path.abspath(__file__))[0]
        self.config_folder = os.path.join(test_base_dir, "camera_config/")
        self.background_folder = os.path.join(test_base_dir, "background_config/")

        require_folder(self.background_folder)

        self.process = Process(target=start_camera_server, args=(self.host, self.port, self.config_folder))
        self.process.start()

        # Give it some time to start.
        sleep(0.5)

        server_address = "http://%s:%s" % (self.host, self.port)
        self.client = CamClient(server_address)

    def tearDown(self):
        test_cleanup([self.client], [self.process],
                     [
                         os.path.join(self.config_folder, "testing_camera.json"),
                         os.path.join(self.background_folder, "white_background.npy"),
                         self.background_folder
                     ])

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

        #This is not valid test anymore: if both are set, the configuration fields are added to the the named pipeline
        #with self.assertRaisesRegex(ValueError, "You must specify either the pipeline name or the "
        #                                        "configuration for the pipeline."):
        #    instance_manager.create_pipeline(pipeline_name="test_pipeline", configuration=pipeline_config)

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
        instance_manager.background_manager = BackgroundImageManager(self.background_folder)

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
        background_id = instance_manager.collect_background(camera_name, number_of_images)

        self.assertTrue(background_id.startswith("simulation"), "Background id not as expected.")

        host, port = get_host_port_from_stream_address(instance_manager.
                                                       cam_server_client.get_instance_stream("simulation"))

        # Collect from the camera.
        with source(host=host, port=port, mode=SUB) as stream:
            data = stream.receive()
            self.assertIsNotNone(data, "This should really not happen anymore.")

        self.assertEqual(instance_manager.background_manager.get_background(background_id).shape,
                         data.data.data["image"].value.shape,
                         "Background and image have to be of the same shape.")

        self.assertEqual(instance_manager.background_manager.get_background(background_id).dtype,
                         data.data.data["image"].value.dtype,
                         "Background and image have to be of the same dtype.")

        instance_manager.stop_all_instances()

    def test_custom_hostname(self):
        config_manager = PipelineConfigManager(config_provider=MockConfigStorage())
        pipeline_instance_manager = PipelineInstanceManager(config_manager, MockBackgroundManager(), None,
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
                "scale": 7,
                "orientation": "vertical"
            },
            "pipeline_type": "processing"
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

        pipeline_manager.update_instance_config(instance_id, {"image_background_enable": True})
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

        pipeline_manager.update_instance_config(instance_id, {"pipeline_type": None})
        self.assertEqual(pipeline_manager.get_instance(instance_id).get_configuration()["pipeline_type"],
                         PipelineConfig.DEFAULT_CONFIGURATION["pipeline_type"],
                         "Default value not set correctly.")

        pipeline_manager.stop_all_instances()

    def test_update_instance_config_with_running(self):
        instance_manager = get_test_pipeline_manager_with_real_cam()
        instance_manager.background_manager = BackgroundImageManager(self.background_folder)

        pipeline_config = {
            "camera_name": "simulation"
        }

        black_image = numpy.zeros(shape=(960, 1280)).astype(dtype="uint16")

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
        white_image.fill(65535)

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

        instance_manager.update_instance_config(pipeline_id, {"image_background_enable": True,
                                                              "image_threshold": 0})

        # Give it some time to load the background.
        sleep(0.5)

        # Collect from the camera.
        with source(host=pipeline_host, port=pipeline_port, mode=SUB) as stream:
            black_image_data = stream.receive()
            print (pipeline_host,pipeline_port)
            self.assertIsNotNone(black_image_data, "This should really not happen anymore.")

        print ("---")
        print (black_image_data.data.data["image"].value.shape, black_image_data.data.data["image"].value.dtype)
        print(black_image.shape, black_image.dtype)
        black_image.astype(dtype="uint16")
        for i in range(len(black_image)):
            for j in range(len(black_image[0])):
                if  black_image_data.data.data["image"].value[i][j] != black_image[i][j]:
                    print (i, j, black_image_data.data.data["image"].value[i][j], black_image[i][j])
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

        instance_manager.stop_all_instances()

    def test_reload_after_stop(self):
        instance_manager = get_test_pipeline_manager()

        instance_id = "trying_to_test"

        initial_config = {"camera_name": "simulation"}

        instance_manager.config_manager.save_pipeline_config(instance_id, initial_config)
        instance_manager.get_instance_stream(instance_id)

        self.assertEqual(instance_manager.get_instance(instance_id).get_configuration()["image_threshold"],
                         None)

        updated_config = {"camera_name": "simulation",
                          "image_threshold": 9999}

        instance_manager.config_manager.save_pipeline_config(instance_id, updated_config)

        self.assertEqual(instance_manager.get_instance(instance_id).get_configuration()["image_threshold"],
                         None, "It should not change at this point.")

        instance_manager.stop_instance(instance_id)
        instance_manager.get_instance_stream(instance_id)

        self.assertEqual(instance_manager.get_instance(instance_id).get_configuration()["image_threshold"],
                         9999, "It should have changed now - reload should happen.")

        instance_manager.stop_all_instances()

    def test_update_stopped_instance(self):
        instance_manager = get_test_pipeline_manager()
        instance_id, _ = instance_manager.create_pipeline(configuration={"camera_name": "simulation"})
        instance_manager.update_instance_config(instance_id, {"camera_name": "simulation",
                                                              "image_threshold": 9999})

        instance_manager.stop_instance(instance_id)

        with self.assertRaisesRegex(ValueError, "Instance '%s' does not exist." % instance_id):
            instance_manager.update_instance_config(instance_id, {"camera_name": "simulation",
                                                                  "image_threshold": 9999})

        instance_manager.stop_all_instances()

    def test_multiple_get_stream(self):
        instance_manager = get_test_pipeline_manager()
        instance_manager.config_manager.save_pipeline_config("simulation", {"camera_name": "simulation"})

        stream_address_1 = instance_manager.get_instance_stream("simulation")
        stream_address_2 = instance_manager.get_instance_stream("simulation")

        self.assertEqual(stream_address_1, stream_address_2)
        self.assertTrue(instance_manager.is_instance_present("simulation"))

        instance_port = instance_manager.get_instance("simulation").get_stream_port()

        self.assertEqual(instance_manager._used_ports[instance_port], "simulation")

        instance_manager.stop_instance("simulation")

        self.assertEqual(len(instance_manager._used_ports), 0)

        self.assertFalse(instance_manager.is_instance_present("simulation"), "Instance should have been deleted.")

        stream_address_3 = instance_manager.get_instance_stream("simulation")

        self.assertNotEqual(stream_address_1, stream_address_3,
                            "The instance was stopped, the stream should have changed.")

        instance_manager.create_pipeline(configuration={"camera_name": "simulation"})
        instance_manager.create_pipeline(configuration={"camera_name": "simulation"})
        instance_manager.create_pipeline(configuration={"camera_name": "simulation"})

        self.assertEqual(len(instance_manager._used_ports), 4,
                         "There should be 4 used ports.")

        instance_manager.stop_all_instances()

        self.assertEqual(len(instance_manager._used_ports), 0,
                         "All ports should be free now.")

    def test_out_of_ports(self):
        config_manager = PipelineConfigManager(config_provider=MockConfigStorage())
        instance_manager = PipelineInstanceManager(config_manager, MockBackgroundManager(), None,
                                                   MockCamServerClient(), port_range=(12000, 12003))

        instance_id_0, _ = instance_manager.create_pipeline(configuration={"camera_name": "simulation"})
        instance_id_1, _ = instance_manager.create_pipeline(configuration={"camera_name": "simulation"})
        instance_id_2, _ = instance_manager.create_pipeline(configuration={"camera_name": "simulation"})

        self.assertEqual(instance_manager.get_instance(instance_id_0).get_stream_port(), 12000)
        self.assertEqual(instance_manager.get_instance(instance_id_1).get_stream_port(), 12001)
        self.assertEqual(instance_manager.get_instance(instance_id_2).get_stream_port(), 12002)

        with self.assertRaisesRegex(Exception, "All ports are used. Stop some instances before opening a new stream."):
            instance_manager.create_pipeline(configuration={"camera_name": "simulation"})

        instance_manager.stop_instance(instance_id_1)

        instance_id_1, _ = instance_manager.create_pipeline(configuration={"camera_name": "simulation"})

        self.assertEqual(instance_manager.get_instance(instance_id_1).get_stream_port(), 12001,
                         "Instance_id_1 should have freeded the port 10001, but some other port was assigned.")

        instance_manager.stop_all_instances()

        # Check if the port rotation works as expected.
        instance_id_2, _ = instance_manager.create_pipeline(configuration={"camera_name": "simulation"})
        instance_id_0, _ = instance_manager.create_pipeline(configuration={"camera_name": "simulation"})
        instance_id_1, _ = instance_manager.create_pipeline(configuration={"camera_name": "simulation"})

        self.assertEqual(instance_manager.get_instance(instance_id_2).get_stream_port(), 12002)
        self.assertEqual(instance_manager.get_instance(instance_id_0).get_stream_port(), 12000)
        self.assertEqual(instance_manager.get_instance(instance_id_1).get_stream_port(), 12001)

        instance_manager.stop_all_instances()

        old_timeout = config.MFLOW_NO_CLIENTS_TIMEOUT
        config.MFLOW_NO_CLIENTS_TIMEOUT = 1

        # Test the cleanup procedure.
        instance_id_2, _ = instance_manager.create_pipeline(configuration={"camera_name": "simulation"})
        instance_id_0, _ = instance_manager.create_pipeline(configuration={"camera_name": "simulation"})
        instance_id_1, _ = instance_manager.create_pipeline(configuration={"camera_name": "simulation"})

        self.assertEqual(len(instance_manager.get_info()["active_instances"]), 3,
                         "All 3 instances should be running")

        with self.assertRaisesRegex(Exception, "All ports are used. Stop some instances before opening a new stream."):
            instance_manager.create_pipeline(configuration={"camera_name": "simulation"})

        # Wait for the instances to die.
        sleep(5)

        self.assertEqual(len(instance_manager.get_info()["active_instances"]), 0,
                         "All instances should be dead by now.")

        instance_id_2, _ = instance_manager.create_pipeline(configuration={"camera_name": "simulation"})
        instance_id_0, _ = instance_manager.create_pipeline(configuration={"camera_name": "simulation"})
        instance_id_1, _ = instance_manager.create_pipeline(configuration={"camera_name": "simulation"})

        # Restore previous state.
        config.MFLOW_NO_CLIENTS_TIMEOUT = old_timeout
        instance_manager.stop_all_instances()

    def test_camera_offline(self):
        instance_manager = get_test_pipeline_manager_with_real_cam()
        with self.assertRaisesRegex(ValueError, "Camera camera_example_1 is not online. Cannot start pipeline."):
            instance_manager.create_pipeline(configuration={"camera_name": "camera_example_1"})

    def test_last_start_time(self):
        instance_manager = get_test_pipeline_manager_with_real_cam()
        instance_id_0, _ = instance_manager.create_pipeline(configuration={"camera_name": "simulation"})

        pipeline_info = instance_manager.get_instance(instance_id_0).get_info()
        self.assertTrue("last_start_time" in pipeline_info)

        last_time = pipeline_info["last_start_time"]
        new_last_time = instance_manager.get_instance(instance_id_0).get_info()["last_start_time"]
        self.assertEqual(last_time, new_last_time, "The instance was still running, the times should be the same.")

        instance_manager.stop_all_instances()

    def test_statistics(self):
        instance_manager = get_test_pipeline_manager_with_real_cam()
        instance_id_0, _ = instance_manager.create_pipeline(configuration={"camera_name": "simulation"})
        latest_statistics = instance_manager.get_instance(instance_id_0).get_statistics()

        for stat in "total_bytes", "clients", "throughput", "frame_rate", "pid", "cpu", "memory":
            self.assertTrue(stat in latest_statistics)

        instance_manager.stop_all_instances()

    def test_get_instance_from_config(self):
        instance_manager = get_test_pipeline_manager_with_real_cam()
        configuration_1 = {"camera_name": "simulation",
                           "pipeline_type": "store",
                           "stream_port": 10000,
                           }

        instance_id_1, instance_stream_1 = instance_manager.get_instance_stream_from_config(configuration_1)
        instance_id_2, instance_stream_2 = instance_manager.get_instance_stream_from_config(configuration_1)

        self.assertEqual(instance_id_1, instance_id_2)
        self.assertEqual(instance_stream_1, instance_stream_2)
        self.assertTrue(instance_manager.get_instance(instance_id_1).is_read_only_config())

        configuration_2 = {"camera_name": "simulation",
                           "pipeline_type": "processing"}

        instance_id_3, instance_stream_3 = instance_manager.get_instance_stream_from_config(configuration_2)

        self.assertTrue(instance_manager.get_instance(instance_id_3).is_read_only_config())

        self.assertNotEqual(instance_id_1, instance_id_3)
        self.assertNotEqual(instance_stream_1, instance_stream_3)

        instance_id_4, instance_stream_4 = instance_manager.create_pipeline(configuration=configuration_1)
        instance_id_5, instance_stream_5 = instance_manager.get_instance_stream_from_config(configuration_1)

        self.assertFalse(instance_manager.get_instance(instance_id_4).is_read_only_config())
        self.assertTrue(instance_manager.get_instance(instance_id_5).is_read_only_config())

        self.assertNotEqual(instance_id_4, instance_id_5,
                            "Only read only instances can be returned by get_instance_stream_from_config.")
        self.assertNotEqual(instance_stream_4, instance_stream_5,
                            "Only read only instances can be returned by get_instance_stream_from_config.")

        instance_manager.stop_all_instances()

        old_timeout = config.MFLOW_NO_CLIENTS_TIMEOUT
        config.MFLOW_NO_CLIENTS_TIMEOUT = 1

        instance_id_6, instance_stream_6 = instance_manager.get_instance_stream_from_config(configuration_1)
        # Lets be sure that the instance goes down.
        sleep(2)
        self.assertFalse(instance_id_6 in instance_manager.get_info()["active_instances"],
                         "Instance not stopped yet.")

        instance_id_7, instance_stream_7 = instance_manager.get_instance_stream_from_config(configuration_1)

        self.assertNotEqual(instance_id_6, instance_id_7, "A new instance id should be assigned here.")

        config.MFLOW_NO_CLIENTS_TIMEOUT = old_timeout
        instance_manager.stop_all_instances()

    def test_get_invalid_instance_stream(self):
        instance_manager = get_test_pipeline_manager_with_real_cam()

        with self.assertRaisesRegex(ValueError, "is not present on server and it is not a saved"):
            instance_manager.get_instance_stream("simulation_sp1")

        instance_manager.stop_all_instances()


if __name__ == '__main__':
    unittest.main()
