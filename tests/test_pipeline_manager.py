import unittest

from tests.helpers.factory import get_test_pipeline_manager


class PipelineManager(unittest.TestCase):

    def test_get_pipeline_list(self):
        pipeline_manager = get_test_pipeline_manager()
        self.assertEqual(len(pipeline_manager.get_pipeline_list()), 0, "Pipeline manager should be empty by default.")

        initial_config = {"test_pipeline1": {},
                          "test_pipeline2": {}}

        pipeline_manager.config_manager.config_provider.configs = initial_config

        self.assertListEqual(sorted(list(initial_config.keys())), sorted(pipeline_manager.get_pipeline_list()),
                             "Set and received lists are not the same.")

    def test_create_pipeline_instance(self):
        pipeline_manager = get_test_pipeline_manager()

    def test_multiple_create_requests(self):
        pass

    def test_multiple_get_requests(self):
        pass

    def test_get_instance_stream(self):
        pipeline_manager = get_test_pipeline_manager()

    def test_pipeline_image(self):
        pass


if __name__ == '__main__':
    unittest.main()
