from cam_server.camera.configuration import CameraConfigManager
from cam_server.camera.management import CameraInstanceManager


def get_test_instance_manager():
    config_manager = CameraConfigManager(config_provider=MockConfigStorage())
    camera_instance_manager = CameraInstanceManager(config_manager)

    return camera_instance_manager


class MockConfigStorage:
    def __init__(self):
        self.configs = {}

    def get_available_configs(self):
        return self.configs.keys()

    def get_config(self, config_name):
        if config_name not in self.configs:
            # Replicate the error in the real config provider.
            raise ValueError("Unable to load config '%s'" % config_name)

        return self.configs[config_name]

    def save_config(self, config_name, configuration):
        self.configs[config_name] = configuration
