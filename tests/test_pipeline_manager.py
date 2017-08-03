# import os
# import signal
# import unittest
# 
# from multiprocessing import Process
# from time import sleep
# 
# from cam_server import CamClient
# from cam_server.start_cam_server import start_camera_server
# from tests.helpers.factory import get_test_pipeline_manager, get_test_pipeline_manager_with_real_cam
# 
# 
# class PipelineManagerTest(unittest.TestCase):
# 
#     def setUp(self):
#         self.host = "0.0.0.0"
#         self.port = 8888
# 
#         test_base_dir = os.path.split(os.path.abspath(__file__))[0]
#         self.config_folder = os.path.join(test_base_dir, "camera_config/")
# 
#         self.process = Process(target=start_camera_server, args=(self.host, self.port, self.config_folder))
#         self.process.start()
# 
#         # Give it some time to start.
#         sleep(0.5)
# 
#         server_address = "http://%s:%s" % (self.host, self.port)
#         self.client = CamClient(server_address)
# 
#     def tearDown(self):
#         self.client.stop_all_cameras()
#         os.kill(self.process.pid, signal.SIGINT)
#         try:
#             os.remove(os.path.join(self.config_folder, "testing_camera.json"))
#         except:
#             pass
#         # Wait for the server to die.
#         sleep(1)
# 
#     def test_get_pipeline_list(self):
#         pipeline_manager = get_test_pipeline_manager()
#         self.assertEqual(len(pipeline_manager.get_pipeline_list()), 0, "Pipeline manager should be empty by default.")
# 
#         initial_config = {"test_pipeline1": {},
#                           "test_pipeline2": {}}
# 
#         pipeline_manager.config_manager.config_provider.configs = initial_config
# 
#         self.assertListEqual(sorted(list(initial_config.keys())), sorted(pipeline_manager.get_pipeline_list()),
#                              "Set and received lists are not the same.")
# 
#     def test_create_pipeline_instance(self):
#         pass
#         # pipeline_manager = get_test_pipeline_manager_with_real_cam()
#         #
#         # pipeline_config = {
#         #     "camera_name": "simulation"
#         # }
#         #
#         # pipeline_manager.config_manager.save_pipeline_config("test_pipeline", pipeline_config)
#         #
#         # pipeline_id, stream_address = pipeline_manager.create_pipeline("test_pipeline")
# 
# 
#     def test_multiple_create_requests(self):
#         # TODO: Write tests.
#         pass
# 
#     def test_multiple_get_requests(self):
#         # TODO: Write tests.
#         pass
# 
#     def test_get_instance_stream(self):
#         # TODO: Write tests.
#         pipeline_manager = get_test_pipeline_manager()
# 
#     def test_pipeline_image(self):
#         # TODO: Write tests.
#         pass
# 
# 
# if __name__ == '__main__':
#     unittest.main()
