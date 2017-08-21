from datetime import datetime
from cam_server import CamClient
from cam_server.camera.configuration import CameraConfigManager
from cam_server.camera.management import CameraInstanceManager
from cam_server.pipeline.configuration import PipelineConfigManager
from cam_server.pipeline.management import PipelineInstanceManager


def get_test_instance_manager():
        config_manager = CameraConfigManager(config_provider=MockConfigStorage())
        camera_instance_manager = CameraInstanceManager(config_manager)

        return camera_instance_manager


def get_test_pipeline_manager():
    config_manager = PipelineConfigManager(config_provider=MockConfigStorage())
    pipeline_instance_manager = PipelineInstanceManager(config_manager, MockBackgroundManager(), MockCamServerClient())

    return pipeline_instance_manager


def get_test_pipeline_manager_with_real_cam():
    config_manager = PipelineConfigManager(config_provider=MockConfigStorage())
    pipeline_instance_manager = PipelineInstanceManager(config_manager, MockBackgroundManager(),
                                                        CamClient("http://0.0.0.0:8888"))

    return pipeline_instance_manager


class MockBackgroundManager:
    def __init__(self):
        self.backgrounds = {}

    def get_background(self, background_name):
        if not background_name:
            return None

        if background_name not in self.backgrounds:
            raise ValueError("Requested background '%s' does not exist." % background_name)

        return self.backgrounds[background_name]

    def save_background(self, background_name, image, append_timestamp=True):
        if append_timestamp:
            background_name += datetime.now().strftime("_%Y%m%d_%H%M%S_%f")

        self.backgrounds[background_name] = image

        return background_name

    def get_latest_background_id(self, background_prefix):
        raise NotImplementedError("This cannot work in the mock.")


class MockConfigStorage:
    def __init__(self):
        self.configs = {}

    def get_available_configs(self):
        return list(self.configs.keys())

    def get_config(self, config_name):
        if config_name not in self.configs:
            # Replicate the error in the real config provider.
            raise ValueError("Config '%s' does not exist." % config_name)

        return self.configs[config_name]

    def save_config(self, config_name, configuration):
        self.configs[config_name] = configuration

    def delete_config(self, config_name):
        del self.configs[config_name]


class MockCamServerClient:

    def get_camera_geometry(self, camera_name):
        return 100, 101

    def get_camera_stream(self, camera_name):
        return "tcp://127.0.0.1:10000"

    def is_camera_online(self, camera_name):
        return True
