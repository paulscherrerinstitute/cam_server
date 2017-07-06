import argparse
import logging
import os

from cam.camera_server.camera.manager import CameraConfigManager, CameraConfigFileStorage, CameraInstanceManager
from cam.camera_server.rest_server import start_rest_interface

from camera_server import config


def start_camera_server(host, port, config_base):

    # Check if config directory exists
    if not os.path.isdir(config_base):
        print('Configuration directory %s does not exist.' % config_base)
        exit(-1)

    config_manager = CameraConfigManager(config_provider=CameraConfigFileStorage(config_base))
    instance_manager = CameraInstanceManager()

    start_rest_interface(host=host, port=port, instance_manager=instance_manager, config_manager=config_manager)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camera processing server')
    parser.add_argument('-p', '--port', default=8888, help="Server port")
    parser.add_argument('-i', '--interface', default='0.0.0.0', help="Hostname interface to bind to")
    parser.add_argument('-b', '--base', default=config.DEFAULT_CAMERA_CONFIG_FOLDER,
                        help="(Camera) Configuration base directory")
    parser.add_argument("--log_level", default=config.DEFAULT_LOGGING_LEVEL,
                        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                        help="Log level to use.")
    arguments = parser.parse_args()

    # Setup the logging level.
    logging.basicConfig(level=arguments.log_level)

    start_camera_server(arguments.interface, arguments.port, arguments.base)
