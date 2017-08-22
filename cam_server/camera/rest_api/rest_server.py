import json
import logging

import bottle
from bottle import request, response

from cam_server import config
from cam_server.instance_management import rest_api
from cam_server.pipeline.data_processing.functions import get_png_from_image

_logger = logging.getLogger(__name__)


def register_rest_interface(app, instance_manager, interface_prefix=None):
    """
    Get the rest api server.
    :param app: Bottle app to register the interface to.
    :param instance_manager: Manager for camera instances.
    :param interface_prefix: Prefix to put before commands, and after api prefix.
    """

    if interface_prefix is None:
        interface_prefix = config.CAMERA_REST_INTERFACE_PREFIX

    api_root_address = config.API_PREFIX + interface_prefix

    # Register instance management API.
    rest_api.register_rest_interface(app, instance_manager, api_root_address)

    @app.get(api_root_address)
    def get_camera_list():
        """
        Return the list of available cameras.
        :return:
        """
        return {"state": "ok",
                "status": "List of available cameras.",
                "cameras": instance_manager.get_camera_list()}

    @app.get(api_root_address + "/<camera_name>")
    def get_camera_stream(camera_name):
        """
        Get the camera stream address.
        :param camera_name: Name of the camera.
        :return:
        """
        return {"state": "ok",
                "status": "Stream address for camera %s." % camera_name,
                "stream": instance_manager.get_camera_stream(camera_name)}

    @app.get(api_root_address + "/<camera_name>/is_online")
    def is_camera_online(camera_name):
        online = True
        status = "Camera %s is online." % camera_name

        camera = instance_manager.config_manager.load_camera(camera_name)
        try:
            camera.verify_camera_online()
        except Exception as e:
            online = False
            status = str(e)

        return {"state": "ok",
                "status": status,
                "online": online}

    @app.get(api_root_address + '/<camera_name>/config')
    def get_camera_config(camera_name):
        """
        Get cam_server config.
        :param camera_name: Name of the cam_server to retrieve the config for.
        :return: Camera config.
        """
        return {"state": "ok",
                "status": "Camera %s configuration retrieved." % camera_name,
                "config": instance_manager.config_manager.get_camera_config(camera_name).get_configuration()}

    @app.post(api_root_address + '/<camera_name>/config')
    def set_camera_config(camera_name):
        """
        Set the camera settings.
        :param camera_name: Name of the camera to change the config for.
        :return: New config.
        """

        instance_manager.config_manager.save_camera_config(camera_name, request.json)

        return {"state": "ok",
                "status": "Camera %s configuration saved." % camera_name,
                "config": instance_manager.config_manager.get_camera_config(camera_name).get_configuration()}

    @app.delete(api_root_address + '/<camera_name>/config')
    def delete_camera_config(camera_name):
        """
        Delete camera settings.
        :param camera_name: Name of the camera to delete the config for.
        """

        instance_manager.config_manager.delete_camera_config(camera_name)

        return {"state": "ok",
                "status": "Camera %s configuration deleted." % camera_name}

    @app.get(api_root_address + '/<camera_name>/geometry')
    def get_camera_geometry(camera_name):
        """
        Return cam_server geometry. This geometry can change when the cam_server is rebooted
        therefor this is a special call.
        """
        width, height = instance_manager.config_manager.get_camera_geometry(camera_name)

        return {"state": "ok",
                "status": "Geometry of camera %s retrieved." % camera_name,
                "geometry": [width, height]}

    @app.get(api_root_address + '/<camera_name>/image')
    def get_camera_image(camera_name):
        """
        Return a camera image in PNG format. URL parameters available:
        raw, scale=[float], min_value=[float], max_value[float], colormap[string].
        Colormap: See http://matplotlib.org/examples/color/colormaps_reference.html
        :param camera_name: Name of the camera to grab the image from.
        :return: PNG image.
        """

        camera = instance_manager.config_manager.load_camera(camera_name)
        raw = 'raw' in request.params
        scale = float(request.params["scale"]) if "scale" in request.params else None
        min_value = float(request.params["min_value"]) if "min_value" in request.params else None
        max_value = float(request.params["max_value"]) if "max_value" in request.params else None
        colormap_name = request.params.get("colormap")

        # Retrieve a single image from the camera.
        camera.connect()
        image_raw_bytes = camera.get_image(raw=raw)
        camera.disconnect()

        image = get_png_from_image(image_raw_bytes, scale, min_value, max_value, colormap_name)

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

        return json.dumps({"state": "error",
                           "status": str(error.exception)})
