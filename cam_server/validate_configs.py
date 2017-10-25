import argparse
import logging
import os

from cam_server import config
from cam_server.camera.configuration import CameraConfigManager
from cam_server.instance_management.configuration import ConfigFileStorage
from cam_server.pipeline.configuration import PipelineConfigManager, BackgroundImageManager

_logger = logging.getLogger(__name__)


def verify_camera_configs(folder):
    _logger.info("Verifying camera configs in folder '%s'.", folder)

    config_manager = CameraConfigManager(config_provider=ConfigFileStorage(folder))

    for camera_name in config_manager.get_camera_list():
        try:
            config_manager.load_camera(camera_name)
        except:
            _logger.exception("Error while trying to load camera '%s'", camera_name)

    _logger.info("Camera configs verification completed.")


def verify_pipeline_configs(folder, background_folder):
    if not os.path.isdir(background_folder):
        _logger.error("Specified background folder '%s' does not exist.", background_folder)
        background_manager = None
    else:
        background_manager = BackgroundImageManager(background_folder)

    _logger.info("Verifying pipeline configs in folder '%s'.", folder)

    config_manager = PipelineConfigManager(config_provider=ConfigFileStorage(folder))

    for pipeline_name in config_manager.get_pipeline_list():
        try:
            pipeline = config_manager.load_pipeline(pipeline_name)

            if background_manager:
                background_manager.get_background(pipeline.get_background_id())

        except:
            _logger.exception("Error while trying to load pipeline '%s'", pipeline_name)

    _logger.info("Pipeline configs verification completed.")


def validate_configs(camera_folder, pipeline_folder, background_folder):

    if not os.path.isdir(camera_folder):
        _logger.error("Specified camera folder '%s' does not exist.", camera_folder)
    else:
        verify_camera_configs(camera_folder)

    if not os.path.isdir(pipeline_folder):
        _logger.error("Specified pipeline folder '%s' does not exist.", camera_folder)
    else:
        verify_pipeline_configs(pipeline_folder, background_folder)


def main():
    parser = argparse.ArgumentParser(description='Validate the camera and pipeline configs.')
    parser.add_argument('-c', '--camera', default=config.DEFAULT_CAMERA_CONFIG_FOLDER,
                        help="(Camera) Configuration base directory")
    parser.add_argument('-p', '--pipeline', default=config.DEFAULT_PIPELINE_CONFIG_FOLDER,
                        help="(Pipeline) Configuration base directory")
    parser.add_argument('-b', '--background', default=config.DEFAULT_BACKGROUND_CONFIG_FOLDER)
    arguments = parser.parse_args()

    logging.basicConfig(level="DEBUG")

    validate_configs(arguments.camera, arguments.pipeline, arguments.background)

if __name__ == "__main__":
    main()
