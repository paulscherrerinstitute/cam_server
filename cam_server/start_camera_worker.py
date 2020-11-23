import argparse
import logging
import bottle

from cam_server import config
from cam_server.camera.management import CameraInstanceManager
from cam_server.camera.configuration import CameraConfigManager
from cam_server.instance_management.configuration import TransientConfig
from cam_server.camera.rest_api.rest_server import register_rest_interface as register_camera_rest_interface
from cam_server.utils import initialize_api_logger, string_to_dict, validate_web_server

_logger = logging.getLogger(__name__)


def start_camera_worker(host, port, hostname=None, port_range=None, mode=0,
                        web_server=config.DEFAULT_WEB_SERVER, web_server_args={}):

    if hostname:
        _logger.warning("Using custom hostname '%s'." % hostname)

    config_manager = CameraConfigManager(config_provider=TransientConfig())
    camera_instance_manager = CameraInstanceManager(config_manager, hostname=hostname, port_range=port_range, mode=mode)

    app = bottle.Bottle()

    register_camera_rest_interface(app=app, instance_manager=camera_instance_manager)

    try:
        bottle.run(app=app, server=validate_web_server(web_server), host=host, port=port, **web_server_args)
    finally:
        # Close the external processor when terminating the web server.
        camera_instance_manager.stop_all_instances()

def main():
    parser = argparse.ArgumentParser(description='Camera acquisition server')
    parser.add_argument('-p', '--port', default=8880, help="Server cam_port")
    parser.add_argument('-i', '--interface', default='0.0.0.0', help="Hostname interface to bind to")
    parser.add_argument('-n', '--hostname', default=None, help="Hostname to use when returning the stream address.")
    parser.add_argument('-m', '--mode', default=0)
    parser.add_argument('-w', '--web_server', default=config.DEFAULT_WEB_SERVER)
    parser.add_argument('-a', '--web_server_args', default="")
    parser.add_argument('-e', '--epics_timeout', default=None)
    parser.add_argument('-f', '--ipc_feed_folder', default="/tmp")
    parser.add_argument("--log_level", default=config.DEFAULT_LOGGING_LEVEL,
                        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                        help="Log level to use.")
    arguments = parser.parse_args()

    # Setup the logging level.
    logging.basicConfig(level=arguments.log_level)
    initialize_api_logger(arguments.log_level)
    if arguments.epics_timeout is not None:
        config.EPICS_TIMEOUT = float(arguments.epics_timeout)
    if arguments.ipc_feed_folder is not None:
        config.IPC_FEEDS_FOLDER = str(arguments.ipc_feed_folder)
    start_camera_worker(arguments.interface, arguments.port, arguments.hostname, None,
                          int(arguments.mode), arguments.web_server, string_to_dict(arguments.web_server_args))


if __name__ == "__main__":
    main()
