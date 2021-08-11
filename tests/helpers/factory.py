from datetime import datetime
from os import listdir

from mflow import mflow, PUSH, sleep
from os.path import join, isfile

from cam_server import CamClient
from cam_server.camera.configuration import CameraConfigManager
from cam_server.camera.management import CameraInstanceManager
from cam_server.camera.source.bsread import CameraBsread
from cam_server.pipeline.configuration import PipelineConfigManager
from cam_server.pipeline.management import PipelineInstanceManager


def get_test_instance_manager():
    config_manager = CameraConfigManager(config_provider=MockConfigStorage())
    camera_instance_manager = CameraInstanceManager(config_manager, None)

    return camera_instance_manager


def get_test_pipeline_manager():
    config_manager = PipelineConfigManager(config_provider=MockConfigStorage())
    pipeline_instance_manager = PipelineInstanceManager(config_manager, MockBackgroundManager(), None, MockCamServerClient())

    return pipeline_instance_manager


def get_test_pipeline_manager_with_real_cam():
    config_manager = PipelineConfigManager(config_provider=MockConfigStorage())
    pipeline_instance_manager = PipelineInstanceManager(config_manager, MockBackgroundManager(), None,
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

    def get_background_ids(self, background_prefix):
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

    def get_instance_stream(self, camera_name):
        return "tcp://127.0.0.1:10000"

    def is_camera_online(self, camera_name):
        return True


class MockCameraBsread(CameraBsread):
    def __init__(self, camera_config, width, height, stream_address):
        super(MockCameraBsread, self).__init__(camera_config)
        self.width_raw = width
        self.height_raw = height
        self.bsread_stream_address = stream_address

    def verify_camera_online(self):
        pass

    def _collect_camera_settings(self):
        pass


def replay_dump(bind_address, input_folder, wait=0.5):
    sleep(wait)
    stream = mflow.connect(bind_address, conn_type="bind", mode=PUSH)

    files = sorted(listdir(input_folder))

    for index, raw_file in enumerate(files):
        filename = join(input_folder, raw_file)
        if not (raw_file.endswith('.raw') and isfile(filename)):
            continue

        with open(filename, mode='rb') as file_handle:
            send_more = False
            if index + 1 < len(files):  # Ensure that we don't run out of bounds
                send_more = raw_file.split('_')[0] == files[index + 1].split('_')[0]

            print('Sending %s [%s]' % (raw_file, send_more))
            stream.send(file_handle.read(), send_more=send_more)