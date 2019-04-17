import argparse
import logging
import os

import bottle
from cam_server.pipeline.configuration import PipelineConfigManager, BackgroundImageManager

from cam_server.pipeline.management import PipelineInstanceManager
from cam_server.pipeline.rest_api.rest_server import register_rest_interface as register_pipeline_rest_interface

from cam_server import config, CamClient
from cam_server.instance_management.configuration import TransientConfig

_logger = logging.getLogger(__name__)


def start_pipeline_worker(host, port, background_base, cam_server_api_address, hostname=None):

    if not os.path.isdir(background_base):
        _logger.error("Background image directory '%s' does not exist." % background_base)
        exit(-1)

    if hostname:
        _logger.warning("Using custom hostname '%s'." % hostname)

    cam_server_client = CamClient(cam_server_api_address)
    config_manager = PipelineConfigManager(config_provider=TransientConfig())
    background_manager = BackgroundImageManager(background_base)
    pipeline_instance_manager = PipelineInstanceManager(config_manager, background_manager, cam_server_client,
                                                        hostname=hostname)

    app = bottle.Bottle()

    register_pipeline_rest_interface(app=app, instance_manager=pipeline_instance_manager)

    try:
        bottle.run(app=app, host=host, port=port)
    finally:
        # Close the external processor when terminating the web server.
        pipeline_instance_manager.stop_all_instances()


def main():
    parser = argparse.ArgumentParser(description='Pipeline processing server')
    parser.add_argument("-c", '--cam_server', default="http://0.0.0.0:8898", help="Cam server rest api address.")
    parser.add_argument('-p', '--port', default=8889, help="Server port")
    parser.add_argument('-i', '--interface', default='0.0.0.0', help="Hostname interface to bind to")
    parser.add_argument('-g', '--background_base', default=config.DEFAULT_BACKGROUND_CONFIG_FOLDER)
    parser.add_argument('-n', '--hostname', default=None, help="Hostname to use when returning the stream address.")
    parser.add_argument("--log_level", default=config.DEFAULT_LOGGING_LEVEL,
                        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                        help="Log level to use.")
    arguments = parser.parse_args()

    # Setup the logging level.
    logging.basicConfig(level=arguments.log_level)

    start_pipeline_worker(arguments.interface, arguments.port,
                          arguments.background_base, arguments.cam_server,
                          arguments.hostname)


if __name__ == "__main__":
    main()
