import os
import signal
import unittest

from multiprocessing import Process
from time import sleep

from bsread import source, SUB
from cam_server import CamClient
from cam_server.pipeline.configuration import PipelineConfig
from cam_server.start_camera_server import start_camera_server
from cam_server.utils import collect_background, get_host_port_from_stream_address
from tests.helpers.factory import get_test_pipeline_manager, get_test_pipeline_manager_with_real_cam


class PipelineManagerTest(unittest.TestCase):
    def setUp(self):
        self.host = "0.0.0.0"
        self.port = 8888

        test_base_dir = os.path.split(os.path.abspath(__file__))[0]
        self.config_folder = os.path.join(test_base_dir, "camera_config/")

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

        instance_manager.config_manager.save_pipeline_config(pipeline_id, pipeline_config.get_parameters())
        instance_stream_1 = instance_manager.get_instance_stream(pipeline_id)

        self.assertTrue(instance_manager.get_info()["active_instances"][pipeline_id]["read_only"],
                        "Instance should be read only.")

        with self.assertRaisesRegex(ValueError, "Cannot set config on a read only instance."):
            instance_manager.get_instance(pipeline_id).set_parameter({})

        instance_stream_2 = instance_manager.get_instance_stream(pipeline_id)

        self.assertEqual(instance_stream_1, instance_stream_2, "Only one instance should be present.")


    def test_pipeline_image(self):
        # TODO: Write tests.
        pass

    def test_collect_background(self):
        instance_manager = get_test_pipeline_manager_with_real_cam()

        pipeline_id = "test_pipeline"
        number_of_images = 10

        pipeline_config = PipelineConfig(pipeline_id, parameters={
            "camera_name": "simulation"
        })

        instance_manager.config_manager.save_pipeline_config(pipeline_id, pipeline_config.get_parameters())

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


if __name__ == '__main__':
    unittest.main()
