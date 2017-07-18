import json
import logging

import bottle
from bottle import request, response

from cam_server import config
from cam_server.camera.utils import get_image_from_camera

_logger = logging.getLogger(__name__)


def start_rest_interface(host, port, instance_manager):
    """
    Start the rest api server.
    :param host: Interface to start the rest server on.
    :param port: Port to start the rest server on.
    :param instance_manager: Manager for camera instances.
    """

    api_root_address = config.API_PREFIX + "/cam_server"
    app = bottle.Bottle()

    @app.get(api_root_address)
    def list_cameras():
        """
        List all the cameras running on the server.
        """
        return {"state": "ok",
                "status": "List of cameras retrieved.",
                'cameras': instance_manager.config_manager.get_camera_list()}

    @app.delete(api_root_address)
    def stop_all_cameras():
        """
        Stop all the cameras running on the server.
        """
        instance_manager.stop_all_cameras()

        return {"state": "ok",
                "status": "All camera streams have been stopped."}

    @app.get(api_root_address + "/info")
    def get_server_info():
        """
        Return the current camera server instance info.
        """
        return {"state": "ok",
                "status": "Server info retrieved.",
                "info": instance_manager.get_info()}

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

    @app.delete(api_root_address + "/<camera_name>")
    def stop_camera(camera_name):
        """
        Stop a specific camera.
        :param camera_name: Name of the camera.
        """
        instance_manager.stop_camera(camera_name)

        return {"state": "ok",
                "status": "Camera '%s' stopped." % camera_name}

    @app.get(api_root_address + '/<camera_name>/config')
    def get_camera_config(camera_name):
        """
        Get cam_server config.
        :param camera_name: Name of the cam_server to retrieve the config for.
        :return: Camera config.
        """
        return {"state": "ok",
                "status": "Camera %s configuration retrieved." % camera_name,
                "config": instance_manager.config_manager.get_camera_config(camera_name)}

    @app.post(api_root_address + '/<camera_name>/config')
    def set_camera_config(camera_name):
        """
        Set the camera settings.
        :param camera_name: Name of the camera to change the config for.
        :return: New config.
        """

        instance_manager.config_manager.save_camera_config(camera_name, json.request)

        return {"state": "ok",
                "status": "Camera %s configuration saved." % camera_name,
                "config": instance_manager.config_manager.get_camera_config(camera_name)}

    @app.get(api_root_address + '/<camera_name>/geometry')
    def get_camera_geometry(camera_name):
        """
        Return cam_server geometry. This geometry can change when the cam_server is rebooted
        therefor this is a special call.
        """
        width, height = instance_manager.config_manager.get_camera_geometry(camera_name)

        return {"state": "ok",
                "status": "Geometry of camera %s retrieved." % camera_name,
                "width": width,
                "height": height}

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

        return json.dumps({"state": "error",
                           "status": str(error.exception)})

    try:
        bottle.run(app=app, host=host, port=port)
    finally:
        # Close the external processor when terminating the web server.
        instance_manager.stop_all_cameras()
