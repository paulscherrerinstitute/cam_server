import os
import unittest

from cam_server.instance_management.configuration import ConfigFileStorage


class ConfigFileStorageTest(unittest.TestCase):
    def setUp(self):
        test_base_dir = os.path.split(os.path.abspath(__file__))[0]
        self.config_folder = os.path.join(test_base_dir, "camera_config/")
        self.file_storage = ConfigFileStorage(config_folder=self.config_folder)

        self.config_to_save_name = "test_config"

    def tearDown(self):
        try:
            self.file_storage.delete_config(self.config_to_save_name)
        except:
            pass

    def test_list_configs(self):
        expected_config_files = ["camera_example_1", "camera_example_2", "camera_example_3", "camera_example_4"]
        available_configs_1 = self.file_storage.get_available_configs()
        self.assertListEqual(available_configs_1, expected_config_files,
                             "Available configs not as expected.")

    def test_get_config(self):
        config_example_1 = self.file_storage.get_config("camera_example_1")

        expected_config_example_1 = {
            "name": "camera_example_1",
            "source": "EPICS_example_1",
            "source_type": "epics",
            "mirror_x": True,
            "mirror_y": True,
            "rotate": 1,
            "camera_calibration": None

        }

        self.assertDictEqual(config_example_1, expected_config_example_1,
                             "Config not as it should be.")

    def test_set_config(self):
        config_to_save = {
            "name": "test_config",
            "source": "test",
            "source_type": "epics",
            "mirror_x": True,
            "mirror_y": False,
            "rotate": 4
        }

        self.file_storage.save_config(self.config_to_save_name, config_to_save)
        stored_config = self.file_storage.get_config(self.config_to_save_name)

        self.assertDictEqual(stored_config, config_to_save,
                             "Saved and stored configs do not match.")

        config_to_modify_and_save = {
            "name": "invalid_name",
            "source": "test",
            "source_type": "epics",
            "mirror_x": True,
            "mirror_y": False,
            "rotate": 4
        }

        self.file_storage.save_config(self.config_to_save_name, config_to_modify_and_save)
        stored_config = self.file_storage.get_config(self.config_to_save_name)

        self.assertEqual(stored_config["name"], self.config_to_save_name,
                         "Config name not modified as expected.")

    def test_delete_config(self):
        config_to_save = {
            "name": "test_config",
            "source": "test",
            "source_type": "epics",
            "mirror_x": True,
            "mirror_y": False,
            "rotate": 4
        }

        self.file_storage.save_config(self.config_to_save_name, config_to_save)

        self.assertTrue(self.config_to_save_name in self.file_storage.get_available_configs(),
                        "New config not present in available configs.")

        self.file_storage.delete_config(self.config_to_save_name)

        self.assertFalse(self.config_to_save_name in self.file_storage.get_available_configs(),
                         "New config was not deleted.")


if __name__ == '__main__':
    unittest.main()
