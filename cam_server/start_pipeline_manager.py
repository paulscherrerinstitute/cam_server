import argparse
import logging
import os
import bottle

from cam_server.pipeline.configuration import PipelineConfigManager, BackgroundImageManager, UserScriptsManager
from cam_server.pipeline.rest_api.rest_server import register_rest_interface as register_pipeline_rest_interface
from cam_server import config, CamClient
from cam_server.instance_management.configuration import ConfigFileStorage
from cam_server.pipeline.manager import Manager as PipelineManager
from cam_server.utils import initialize_api_logger

_logger = logging.getLogger(__name__)


def start_pipeline_manager(host, port, server_config, config_base, background_base, background_files_days_to_live,
                           scripts_base, cam_server_api_address, client_timeout=None, info_update_timeout=None):


    # Check if config directory exists
    if not os.path.isdir(config_base):
        _logger.error("Configuration directory '%s' does not exist." % config_base)
        exit(-1)

    if not os.path.isdir(background_base):
        _logger.error("Background image directory '%s' does not exist." % background_base)
        exit(-1)

    if not os.path.isdir(scripts_base):
        _logger.error("Scripts directory '%s' does not exist." % scripts_base)
        exit(-1)

    cam_server_client = CamClient(cam_server_api_address)
    config_manager = PipelineConfigManager(config_provider=ConfigFileStorage(config_base))
    background_manager = BackgroundImageManager(background_base)
    user_scripts_manager = UserScriptsManager(scripts_base)

    app = bottle.Bottle()

    proxy = PipelineManager(config_manager, background_manager,user_scripts_manager,
                            cam_server_client, server_config, int(background_files_days_to_live),
                            float(client_timeout) if client_timeout else None,
                            float(info_update_timeout) if info_update_timeout else None)
    register_pipeline_rest_interface(app=app, instance_manager=proxy)
    proxy.register_rest_interface(app)
    proxy.register_management_page(app)
    try:
        bottle.run(app=app, host=host, port=port)
    finally:
        #cleanup
        pass


def main():
    parser = argparse.ArgumentParser(description='Pipeline processing server')
    parser.add_argument("-c", '--cam_server', default="http://0.0.0.0:8888", help="Cam server rest api address.")
    parser.add_argument('-p', '--port', default=8889, help="Server port")
    parser.add_argument('-i', '--interface', default='0.0.0.0', help="Hostname interface to bind to")
    parser.add_argument('-s', '--servers', default="",
                        help="Comma-separated list of servers (if not provided, configuration read from servers.json)")
    parser.add_argument('-t', '--client_timeout', default=config.DEFAULT_SERVER_CLIENT_TIMEOUT, help="Server client timeout in seconds")
    parser.add_argument('-m', '--update_timeout', default=config.DEFAULT_SERVER_INFO_TIMEOUT, help="Timeout for server info updates in seconds")
    parser.add_argument('-b', '--base', default=config.DEFAULT_PIPELINE_CONFIG_FOLDER,
                        help="(Pipeline) Configuration base directory")
    parser.add_argument('-g', '--background_base', default=config.DEFAULT_BACKGROUND_CONFIG_FOLDER)
    parser.add_argument('-l', '--background_files_days_to_live', default=config.DEFAULT_BACKGROUND_FILES_DAYS_TO_LIVE)
    parser.add_argument('-u', '--scripts_base', default=config.DEFAULT_USER_SCRIPT_FOLDER)

    parser.add_argument("--log_level", default=config.DEFAULT_LOGGING_LEVEL,
                        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                        help="Log level to use.")
    arguments = parser.parse_args()

    # Setup the logging level.
    logging.basicConfig(level=arguments.log_level)
    initialize_api_logger(arguments.log_level)
    start_pipeline_manager(arguments.interface, arguments.port, arguments.servers, arguments.base,
                          arguments.background_base, arguments.background_files_days_to_live, arguments.scripts_base,
                          arguments.cam_server, arguments.client_timeout, arguments.update_timeout)


if __name__ == "__main__":
    main()
