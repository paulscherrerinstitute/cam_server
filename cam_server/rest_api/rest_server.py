import json
import logging

import bottle
from bottle import get
from bottle import request, response

from cam_server import config
from cam_server.camera.utils import get_image_from_camera

_logger = logging.getLogger(__name__)


def start_rest_interface(host, port, instance_manager, config_manager):
    """
    Start the rest api server.
    :param host: Interface to start the rest server on.
    :param port: Port to start the rest server on.
    :param instance_manager: Manager for camera instances.
    :param config_manager: Provider of cameras.
    """

    api_prefix = config.API_PREFIX
    app = bottle.Bottle()

    @app.get(api_prefix + '/cam_server')
    def list_cameras():
        """List existing cameras"""
        return {"state": "ok",
                "status": "List of cameras retrieved.",
                'cameras': config_manager.get_camera_list()}

    @app.get(api_prefix + "/cam_server/info")
    def get_server_info():
        return {"state": "ok",
                "status": "Server info retrieved.",
                "info": instance_manager.get_info()}

    @app.get(api_prefix + "/cam_server/<camera_name>")
    def get_camera_stream(camera_name):
        return {"state": "ok",
                "status": "Stream address for camera %s." % camera_name,
                "stream": instance_manager.get_camera_stream(config_manager.get_camera_config(camera_name))}

    @app.get(api_prefix + '/cam_server/<camera_name>/config')
    def get_camera_config(camera_name):
        """
        Get cam_server config.
        :param camera_name: Name of the cam_server to retrieve the config for.
        :return: Camera config.
        """
        return {"state": "ok",
                "status": "Camera %s configuration retrieved." % camera_name,
                "config": config_manager.get_camera_config(camera_name)}

    @app.post(api_prefix + '/cam_server/<camera_name>/config')
    def set_camera_config(camera_name):
        """
        Set the camera settings.
        :param camera_name: Name of the camera to change the config for.
        :return: New config.
        """

        config_manager.save_camera_config(camera_name, json.request)

        return {"state": "ok",
                "status": "Camera %s configuration saved." % camera_name,
                "config": config_manager.get_camera_config(camera_name)}

    @app.get(api_prefix + '/cam_server/<camera_name>/geometry')
    def get_camera_geometry(camera_name):
        """
        Return cam_server geometry. This geometry can change when the cam_server is rebooted
        therefor this is a special call.
        """
        width, height = config_manager.get_camera_geometry(camera_name)

        return {"state": "ok",
                "status": "Geometry of camera %s retrieved." % camera_name,
                "width": width,
                "height": height}

    @app.get(api_prefix + '/cam_server/<camera_name>/image')
    def get_camera_image(camera_name):

        camera = config_manager.load_camera(camera_name)
        raw = 'raw' in request.params
        scale = float(request.params["scale"]) if "scale" in request.params else None
        min_value = float(request.params["min_value"]) if "min_value" in request.params else None
        max_value = float(request.params["max_value"]) if "max_value" in request.params else None
        colormap_name = request.params.get("colormap")

        image = get_image_from_camera(camera, raw, scale, min_value, max_value, colormap_name)

        response.set_header('Content-type', 'image/png')
        return image

    @app.error(405)
    def method_not_allowed(res):
        if request.method == 'OPTIONS':
            new_res = bottle.HTTPResponse()
            new_res.set_header('Access-Control-Allow-Origin', '*')
            new_res.set_header('Access-Control-Allow-Methods', 'PUT, GET, POST, DELETE, OPTIONS')
            new_res.set_header('Access-Control-Allow-Headers', 'Origin, Accept, Content-Type')
            return new_res
        res.headers['Allow'] += ', OPTIONS'
        return request.app.default_error_handler(res)

    @app.hook('after_request')
    def enable_cors():
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
        response.headers[
            'Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

    @app.error(500)
    def error_handler_500(error):
        response.content_type = 'application/json'
        response.status = 200

        return json.dumps({"status": "error",
                           "message": str(error.exception)})

    try:
        bottle.run(app=app, host=host, port=port)
    finally:
        # Close the external processor when terminating the web server.
        instance_manager.stop_all_cameras()


def _pick_unused_port():
    # Recipe from: http://code.activestate.com/recipes/531822-pick-unused-port/
    # There is a chance that the port returned by this code might be taken before the calling code can bind to this port
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    _, port = s.getsockname()
    s.close()
    return port

#
# def get_client(address):
#     """ Factory method for cam client """
#     import cam.client
#     return cam.client.CamClient(address)
