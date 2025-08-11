import argparse
import logging
import os
import bottle

from cam_server import config
from cam_server.camera.configuration import CameraConfigManager
from cam_server.instance_management.configuration import ConfigFileStorage, UserScriptsManager, BackgroundImageManager
from cam_server.camera.rest_api.rest_server import register_rest_interface as register_camera_rest_interface
from cam_server.camera.manager import Manager as CameraManager
from cam_server.utils import initialize_api_logger, string_to_dict, validate_web_server

_logger = logging.getLogger(__name__)


def start_camera_manager(host, port, server_config, config_base, background_base, scripts_base, client_timeout=None, info_update_timeout=None,
                           web_server=config.DEFAULT_WEB_SERVER, web_server_args={}):
    if not os.path.isdir(config_base):
        _logger.error("Configuration directory '%s' does not exist." % config_base)
        exit(-1)

    config_manager = CameraConfigManager(config_provider=ConfigFileStorage(config_base))
    background_manager = BackgroundImageManager(background_base)
    user_scripts_manager = UserScriptsManager(scripts_base)

    app = bottle.Bottle()

    proxy = CameraManager(config_manager, background_manager, user_scripts_manager, server_config,
                          float(client_timeout) if client_timeout else None,
                          float(info_update_timeout) if info_update_timeout else None)
    register_camera_rest_interface(app=app, instance_manager=proxy)
    proxy.register_rest_interface(app)
    proxy.register_management_page(app)

    try:
        bottle.run(app=app, server=validate_web_server(web_server), host=host, port=port, **web_server_args)
    finally:
        #clenup
        pass


def main():
    parser = argparse.ArgumentParser(description='Camera proxy server')
    parser.add_argument('-p', '--port', default=8888, help="Camera proxy server port")
    parser.add_argument('-i', '--interface', default='0.0.0.0', help="Hostname interface to bind to")
    parser.add_argument('-s', '--servers', default="",
                        help="Comma-separated list of servers (if not provided, configuration read from servers.json)")
    parser.add_argument('-t', '--client_timeout', default=config.DEFAULT_SERVER_CLIENT_TIMEOUT, help="Server client timeout in seconds")
    parser.add_argument('-m', '--update_timeout', default=config.DEFAULT_SERVER_INFO_TIMEOUT, help="Timeout for server info updates in seconds")
    parser.add_argument('-b', '--base', default=config.DEFAULT_CAMERA_CONFIG_FOLDER,
                        help="(Camera) Configuration base directory")
    parser.add_argument('-g', '--background_base', default=config.DEFAULT_BACKGROUND_CONFIG_FOLDER)
    parser.add_argument('-u', '--scripts_base', default=config.DEFAULT_USER_SCRIPT_FOLDER)
    parser.add_argument('-w', '--web_server', default=config.DEFAULT_WEB_SERVER)
    parser.add_argument('-a', '--web_server_args', default="")
    parser.add_argument('-e', '--epics_timeout', default=None)
    parser.add_argument("--log_level", default=config.DEFAULT_LOGGING_LEVEL,
                        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                        help="Log level to use.")
    arguments = parser.parse_args()
    # Setup the logging level.
    logging.basicConfig(level=arguments.log_level)
    initialize_api_logger(arguments.log_level)
    if arguments.epics_timeout is not None:
        config.EPICS_TIMEOUT = float(arguments.epics_timeout)
    start_camera_manager(arguments.interface, arguments.port, arguments.servers, arguments.base,
                         arguments.background_base, arguments.scripts_base,
                         arguments.client_timeout, arguments.update_timeout,
                         arguments.web_server , string_to_dict(arguments.web_server_args))


if __name__ == "__main__":
    main()
