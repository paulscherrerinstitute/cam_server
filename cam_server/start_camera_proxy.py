import argparse
import logging
import os
import bottle

from cam_server import config
from cam_server.camera.configuration import CameraConfigManager
from cam_server.instance_management.configuration import ConfigFileStorage
from cam_server.camera.rest_api.rest_server import register_rest_interface as register_camera_rest_interface
from cam_server.camera.proxy import Proxy as CameraProxy
from cam_server.utils import initialize_api_logger, string_to_dict


_logger = logging.getLogger(__name__)



def start_camera_proxy(host, port, server_config, config_base,
                           web_server=config.DEFAULT_WEB_SERVER, web_server_args={}):
    if not os.path.isdir(config_base):
        _logger.error("Configuration directory '%s' does not exist." % config_base)
        exit(-1)

    config_manager = CameraConfigManager(config_provider=ConfigFileStorage(config_base))

    app = bottle.Bottle()

    proxy = CameraProxy(config_manager, server_config)
    register_camera_rest_interface(app=app, instance_manager=proxy)
    proxy.register_rest_interface(app)
    proxy.register_management_page(app)

    try:
        bottle.run(app=app, server=web_server, host=host, port=port, **web_server_args)
    finally:
        #clenup
        pass


def main():
    parser = argparse.ArgumentParser(description='Camera proxy server')
    parser.add_argument('-p', '--port', default=8898, help="Camera proxy server port")
    parser.add_argument('-i', '--interface', default='0.0.0.0', help="Hostname interface to bind to")
    parser.add_argument('-s', '--servers', default="",
                        help="Comma-separated list of servers (if not provided, configuration read from servers.json)")
    parser.add_argument('-b', '--base', default=config.DEFAULT_CAMERA_CONFIG_FOLDER,
                        help="(Camera) Configuration base directory")
    parser.add_argument('-w', '--web_server', default=config.DEFAULT_WEB_SERVER)
    parser.add_argument('-a', '--web_server_args', default="")
    parser.add_argument("--log_level", default=config.DEFAULT_LOGGING_LEVEL,
                        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                        help="Log level to use.")
    arguments = parser.parse_args()
    # Setup the logging level.
    logging.basicConfig(level=arguments.log_level)
    initialize_api_logger(arguments.log_level)
    start_camera_proxy(arguments.interface, arguments.port, arguments.servers, arguments.base,
                          arguments.web_server , string_to_dict(arguments.web_server_args))


if __name__ == "__main__":
    main()
