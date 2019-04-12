import argparse
import logging

import bottle
from cam_server import config
from cam_server.camera.configuration import CameraConfigManager
from cam_server.instance_management.configuration import ConfigFileStorage
from cam_server.camera.rest_api.rest_server import register_rest_interface as register_camera_rest_interface
from cam_server.camera.proxy import Proxy as CameraProxy

from cam_server import CamClient

_logger = logging.getLogger(__name__)



def start_camera_proxy_server(host, port, servers, config_base, hostname=None):
    if not os.path.isdir(config_base):
        _logger.error("Configuration directory '%s' does not exist." % config_base)
        exit(-1)


    sever_pool = []
    try:
        servers = [s.strip() for s in servers.split(",")]
    except:
        servers = ["http://localhost:8888"]


    for server in servers:
        sever_pool.append(CamClient(server))

    if hostname:
        _logger.warning("Using custom hostname '%s'." % hostname)

    config_manager = CameraConfigManager(config_provider=ConfigFileStorage(config_base))

    app = bottle.Bottle()

    register_camera_rest_interface(app=app, instance_manager=CameraProxy(config_manager, sever_pool))
    try:
        bottle.run(app=app, host=host, port=port)
    finally:
        #clenup
        pass


def main():
    parser = argparse.ArgumentParser(description='Camera proxy server')
    parser.add_argument('-p', '--port', default=8898, help="Camera proxy server port")
    parser.add_argument('-i', '--interface', default='0.0.0.0', help="Hostname interface to bind to")
    parser.add_argument('-s', '--servers', default="http://localhost:8888",
                        help="Comma-separated list of servers")
    parser.add_argument('-b', '--base', default=config.DEFAULT_CAMERA_CONFIG_FOLDER,
                        help="(Camera) Configuration base directory")
    parser.add_argument('-n', '--hostname', default=None, help="Hostname to use when returning the stream address.")
    parser.add_argument("--log_level", default=config.DEFAULT_LOGGING_LEVEL,
                        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                        help="Log level to use.")
    arguments = parser.parse_args()
    print(arguments.base)
    # Setup the logging level.
    logging.basicConfig(level=arguments.log_level)

    start_camera_proxy_server(arguments.interface, arguments.port, arguments.servers, arguments.base, arguments.hostname)


if __name__ == "__main__":
    main()
