import base64
import json
import logging
import pickle
import bottle
from bottle import request, response

from cam_server import config
from cam_server.instance_management import rest_api
from cam_server.pipeline.data_processing.functions import get_png_from_image
from cam_server.utils import register_logs_rest_interface

_logger = logging.getLogger(__name__)


def register_rest_interface(app, instance_manager, interface_prefix=None):
    """
    Get the rest api server.
    :param app: Bottle app to register the interface to.
    :param instance_manager: Manager for camera instances.
    :param interface_prefix: Prefix to put before commands, and after api prefix.
    """

    if interface_prefix is None:
        interface_prefix = config.PIPELINE_REST_INTERFACE_PREFIX

    api_root_address = config.API_PREFIX + interface_prefix

    # Register instance management API.
    rest_api.register_rest_interface(app, instance_manager, api_root_address)

    # Register logs  API.
    register_logs_rest_interface(app,config.API_PREFIX + config.LOGS_INTERFACE_PREFIX)

    @app.get(api_root_address)
    def get_pipeline_list():
        """
        Return the list of available pipelines.
        :return:
        """
        return {"state": "ok",
                "status": "List of available cameras.",
                "pipelines": list(instance_manager.get_pipeline_list())}

    @app.post(api_root_address)
    def create_pipeline_from_config():
        pipeline_config = request.json
        user_instance_id = request.query.decode().get("instance_id")

        instance_id, stream_address = instance_manager.create_pipeline(configuration=pipeline_config,
                                                                       instance_id=user_instance_id)

        # TODO: Remove dependency on instance.

        return {"state": "ok",
                "status": "Stream address for pipeline %s." % instance_id,
                "instance_id": instance_id,
                "stream": stream_address,
                "config": instance_manager.get_instance_configuration(instance_id)}

    @app.post(api_root_address + '/<pipeline_name>')
    def create_pipeline_from_name(pipeline_name):
        params = request.query.decode()
        user_instance_id = params.get("instance_id")
        additional_config = params.get("additional_config")
        if additional_config:
            additional_config = json.loads(additional_config)

        instance_id, stream_address = instance_manager.create_pipeline(pipeline_name=pipeline_name,
                                                                       configuration = additional_config,
                                                                       instance_id=user_instance_id)

        # TODO: Remove dependency on instance.

        return {"state": "ok",
                "status": "Stream address for pipeline %s." % instance_id,
                "instance_id": instance_id,
                "stream": stream_address,
                "config": instance_manager.get_instance_configuration(instance_id)}

    @app.get(api_root_address + '/instance/<instance_id>')
    def get_instance_stream(instance_id):
        stream_address = instance_manager.get_instance_stream(instance_id)

        return {"state": "ok",
                "status": "Stream address for pipeline %s." % instance_id,
                "stream": stream_address}

    @app.post(api_root_address + '/instance/')
    def get_instance_stream_from_config():
        pipeline_config = request.json

        instance_id, stream_address = instance_manager.get_instance_stream_from_config(configuration=pipeline_config)

        # TODO: Remove dependency on instance.

        return {"state": "ok",
                "status": "Stream address for pipeline %s." % instance_id,
                "instance_id": instance_id,
                "stream": stream_address,
                "config": instance_manager.get_instance_configuration(instance_id)}

    @app.get(api_root_address + '/instance/<instance_id>/info')
    def get_instance_info(instance_id):

        # TODO: Remove dependency on instance.

        return {"state": "ok",
                "status": "Pipeline instance %s info retrieved." % instance_id,
                "info": instance_manager.get_instance_info(instance_id)}

    @app.get(api_root_address + '/instance/<instance_id>/exitcode')
    def get_instance_exit_code(instance_id):
        # TODO: Remove dependency on instance.

        return {"state": "ok",
                "status": "Pipeline instance %s info retrieved." % instance_id,
                "exitcode": instance_manager.get_instance_exit_code(instance_id)}

    @app.get(api_root_address + '/instance/<instance_id>/config')
    def get_instance_config(instance_id):

        # TODO: Remove dependency on instance.

        return {"state": "ok",
                "status": "Pipeline instance %s info retrieved." % instance_id,
                "config": instance_manager.get_instance_configuration(instance_id)}

    @app.post(api_root_address + '/instance/<instance_id>/config')
    def set_instance_config(instance_id):
        config_updates = request.json

        if not config_updates:
            raise ValueError("Config updates cannot be empty.")
        _logger.info("Setting instance '%s' config: %s" % (instance_id, str(config_updates)))
        instance_manager.update_instance_config(instance_id, config_updates)

        # TODO: Remove dependency on instance.

        return {"state": "ok",
                "status": "Pipeline instance %s configuration changed." % instance_id,
                "config": instance_manager.get_instance_configuration(instance_id)}

    @app.get(api_root_address + '/<pipeline_name>/config')
    def get_pipeline_config(pipeline_name):

        # TODO: Remove dependency on config_manager.

        return {"state": "ok",
                "status": "Pipeline %s configuration retrieved." % pipeline_name,
                "config": instance_manager.config_manager.get_pipeline_config(pipeline_name)}

    @app.post(api_root_address + '/<pipeline_name>/config')
    def set_pipeline_config(pipeline_name):

        # TODO: Remove dependency on config_manager.
        _logger.info("Setting pipeline '%s' config: %s" % (pipeline_name, str(request.json)))
        instance_manager.save_pipeline_config(pipeline_name, request.json)

        return {"state": "ok",
                "status": "Pipeline %s configuration saved." % pipeline_name,
                "config": instance_manager.config_manager.get_pipeline_config(pipeline_name)}

    @app.delete(api_root_address + '/<pipeline_name>/config')
    def delete_pipeline_config(pipeline_name):

        # TODO: Remove dependency on config_manager.

        instance_manager.config_manager.delete_pipeline_config(pipeline_name)

        return {"state": "ok",
                "status": "Pipeline %s configuration deleted." % pipeline_name}

    @app.post(api_root_address + '/camera/<camera_name>/background')
    def collect_background_on_camera(camera_name):
        number_of_images = request.query.decode().get("n_images", config.PIPELINE_DEFAULT_N_IMAGES_FOR_BACKGROUND)

        try:
            number_of_images = int(number_of_images)
        except ValueError:
            raise ValueError("n_images must be a number.")

        background_id = instance_manager.collect_background(camera_name, number_of_images)

        return {"state": "ok",
                "status": "Background collected on camera %s." % camera_name,
                "background_id": background_id}

    @app.get(api_root_address + '/camera/<camera_name>/background')
    def get_latest_background_for_camera(camera_name):

        # TODO: Remove dependency to instance_manager

        background_id = instance_manager.background_manager.get_latest_background_id(camera_name)

        return {"state": "ok",
                "status": "Latest background for camera %s." % camera_name,
                "background_id": background_id}

    @app.get(api_root_address + '/camera')
    def get_camera_list():

        # TODO: Remove dependency on cam_server_client.

        return {"state": "ok",
                "status": "List of available cameras.",
                "cameras": instance_manager.cam_server_client.get_cameras()}


    @app.get(api_root_address + '/background/<background_name>/image')
    def get_background_image(background_name):
        """
        Return a background file in PNG format. URL parameters available:
        scale=[float], min_value=[float], max_value[float], colormap[string].
        Colormap: See http://matplotlib.org/examples/color/colormaps_reference.html
        :param background_name: Background file name.
        :return: PNG image.
        """

        scale = float(request.params["scale"]) if "scale" in request.params else None
        min_value = float(request.params["min_value"]) if "min_value" in request.params else None
        max_value = float(request.params["max_value"]) if "max_value" in request.params else None
        colormap_name = request.params.get("colormap")

        image_raw_bytes = instance_manager.background_manager.get_background(background_name)

        image = get_png_from_image(image_raw_bytes, scale, min_value, max_value, colormap_name)

        response.set_header('Content-type', 'image/png')
        return image

    @app.get(api_root_address + '/background/<background_name>/image_bytes')
    def get_background_image_bytes(background_name):
        """
        Return the bytes of aa a background file.
        :param background_name: Background file name.
        :return: JSON with details and byte stream.
        """
        image_bytes = instance_manager.background_manager.get_background(background_name)

        base64_bytes = base64.b64encode(image_bytes)
        image_shape = image_bytes.shape
        image_dtype = image_bytes.dtype.descr[0][1]
        return {"state": "ok",
                "status": "Background file '%s'." % background_name,
                "image": {"bytes": base64_bytes.decode("utf-8"),
                          "shape": image_shape,
                          "dtype": image_dtype}}


    @app.put(api_root_address + '/background/<background_name>/image_bytes')
    def set_background_image_bytes(background_name):
        """
        Sets the bytes of a background file.
        :param background_name: Background file name.
        :param image_bytes: Contents of file
        :return:
        """
        try:
            #Big arrays: request.body is BufferedRandom
            data = request.body.raw.readall()
        except:
            #request.body is  BytesIO
            data = request.body.getvalue()
        arr = pickle.loads(data)
        instance_manager.background_manager.save_background(background_name, arr, False)
        return {"state": "ok",
                "status": "Background image %s stored." % background_name,
                }

    @app.get(api_root_address + '/script')
    def get_script_list():

        # TODO: Remove dependency on cam_server_client.

        return {"state": "ok",
                "status": "List of available cameras.",
                "scripts": instance_manager.user_scripts_manager.get_scripts()}

    @app.get(api_root_address + '/script/<script_name>/script_bytes')
    def get_script(script_name):
        """
        Return the bytes of aa a script file.
        :param background_name: script file name.
        :return: JSON with details and byte stream.
        """
        script = instance_manager.user_scripts_manager.get_script(script_name)
        return {"state": "ok",
                "status": "Script file '%s'." % script_name,
                "script": script,
            }


    @app.put(api_root_address + '/script/<script_name>/script_bytes')
    def set_script(script_name):
        """
        Return the bytes of aa a script file.
        :param background_name: script file name.
        :return:
        """
        data = request.body.read()
        script = data.decode("utf-8")
        instance_manager.save_script(script_name, script)
        return {"state": "ok",
                "status": "Script file %s stored." % script_name,
                }

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
