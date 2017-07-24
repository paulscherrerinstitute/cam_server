import json
import logging

from bottle import request

from cam_server import config
from cam_server.instance_management import rest_api

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

    # TODO: Figure out where and on what to do background collection.
    # @app.post(api_root_address + '/instance/<instance_id>/background')
    # def collect_background(instance_id):
    #     number_of_images = 10
    #     if "number_of_images" in request.json:
    #         number_of_images = request.json["number_of_images"]
    #
    #     background_file = camera_configuration_directory + '/' + camera_name + '_background.npy'
    #     instance.start(None, config_file, number_of_images, background_file)
    #
    #     return json.dumps({"instance_id": default_instance_name})

    @app.post(api_root_address)
    def create_pipeline():
        pipeline_config = request.json
        instance_id, stream_address = instance_manager.create_pipeline_instance(configuration=pipeline_config)

        {"state": "ok",
         "status": "Stream address for pipeline %s." % instance_id,
         "instance_id": instance_id,
         "stream": stream_address,
         "config": instance_manager.get_instance(instance_id).get_parameters()}

    @app.post(api_root_address + '/<pipeline_name>')
    def create_pipeline_from_config(pipeline_name):
        instance_id, stream_address = instance_manager.create_pipeline_instance(pipeline_name=pipeline_name)

        return {"state": "ok",
                "status": "Stream address for pipeline %s." % instance_id,
                "instance_id": instance_id,
                "stream": stream_address,
                "config": instance_manager.get_instance(instance_id).get_parameters()}

    @app.get(api_root_address + '/instance/<instance_id>')
    def get_instance(instance_id):
        pipeline = instance_manager.get_instance(instance_id)

        return {"state": "ok",
                "status": "Stream address for pipeline %s." % instance_id,
                "instance_id": instance_id,
                "stream": pipeline.stream_address,
                "config": pipeline.get_parameters()}

    @app.get(api_root_address + '/instance/<instance_id>/info')
    def get_instance_info(instance_id):
        return {"state": "ok",
                "status": "Pipeline instance %s info retrieved." % instance_id,
                "info": instance_manager.get_instance(instance_id).get_info()}

    @app.get(api_root_address + '/instance/<instance_id>/config')
    def get_instance_config(instance_id):
        return {"state": "ok",
                "status": "Pipeline instance %s info retrieved." % instance_id,
                "config": instance_manager.get_instance(instance_id).get_config()}

    @app.post(api_root_address + '/instance/<instance_id>/config')
    def set_instance_config(instance_id):
        configuration = request.json
        # TODO: This call is to coupled. Remove the validation logic from the rest api.
        # TODO: Get instance current config and update it.
        instance_manager.config_manager.validate_pipeline_config(configuration)

        pipeline = instance_manager.get_instance(instance_id)
        pipeline.set_parameter(configuration)

        return {"state": "ok",
                "status": "Pipeline instance %s configuration changed." % instance_id,
                "config": pipeline.get_config()}

    @app.get(api_root_address + '/<pipeline_name>/config')
    def get_pipeline_config(pipeline_name):
        return {"state": "ok",
                "status": "Pipeline %s configuration retrieved." % pipeline_name,
                "config": instance_manager.config_manager.get_pipeline_config(pipeline_name)}

    @app.post(api_root_address + '/<pipeline_name>/config')
    def set_pipeline_config(pipeline_name):

        instance_manager.config_manager.save_pipeline_config(pipeline_name, json.request)

        return {"state": "ok",
                "status": "Pipeline %s configuration saved." % pipeline_name,
                "config": instance_manager.config_manager.get_pipeline_config(pipeline_name)}
