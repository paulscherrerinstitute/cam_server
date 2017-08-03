import argparse
import logging
import os

import bottle

from cam_server import config
from cam_server.camera.management import CameraInstanceManager
from cam_server.camera.configuration import CameraConfigManager
from cam_server.instance_management.configuration import ConfigFileStorage
from cam_server.camera.rest_api.rest_server import register_rest_interface as register_camera_rest_interface

_logger = logging.getLogger(__name__)


def start_camera_server(host, port, config_base):

    # Check if config directory exists
    if not os.path.isdir(config_base):
        _logger.error("Configuration directory '%s' does not exist." % config_base)
        exit(-1)

    config_manager = CameraConfigManager(config_provider=ConfigFileStorage(config_base))
    camera_instance_manager = CameraInstanceManager(config_manager)

    app = bottle.Bottle()

    register_camera_rest_interface(app=app, instance_manager=camera_instance_manager)

    try:
        bottle.run(app=app, host=host, port=port)
    finally:
        # Close the external processor when terminating the web server.
        camera_instance_manager.stop_all_instances()


def main():
    parser = argparse.ArgumentParser(description='Camera acquisition server')
    parser.add_argument('-p', '--port', default=8888, help="Server cam_port")
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

if __name__ == "__main__":
    main()
