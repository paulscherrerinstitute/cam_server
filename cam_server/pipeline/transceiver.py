from cam_server.pipeline.types import custom
from cam_server.pipeline.types import processing
from cam_server.pipeline.types import store
from cam_server.pipeline.types import stream
from cam_server.pipeline.types import fanout
from cam_server.pipeline.types import fanin

from logging import getLogger

from cam_server import config

_logger = getLogger(__name__)


pipeline_name_to_pipeline_function_mapping = {
    config.PIPELINE_TYPE_PROCESSING: processing.run,
    config.PIPELINE_TYPE_STORE: store.run,
    config.PIPELINE_TYPE_STREAM: stream.run,
    config.PIPELINE_TYPE_CUSTOM: custom.run,
    config.PIPELINE_TYPE_FANOUT: fanout.run,
    config.PIPELINE_TYPE_FANIN: fanin.run,
    config.PIPELINE_TYPE_SCRIPT: None
}

def get_builtin_pipelines():
    return list(pipeline_name_to_pipeline_function_mapping.keys())

def get_pipeline_function(pipeline_type_name):
    builtin_pipelines = get_builtin_pipelines()
    if pipeline_type_name not in pipeline_name_to_pipeline_function_mapping:
        raise ValueError("pipeline_type '%s' not present in mapping. Available: %s." %
                         (pipeline_type_name, builtin_pipelines) )
    return pipeline_name_to_pipeline_function_mapping[pipeline_type_name]
