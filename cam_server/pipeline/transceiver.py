from cam_server.pipeline.types import custom
from cam_server.pipeline.types import processing
from cam_server.pipeline.types import store
from cam_server.pipeline.types import stream

from logging import getLogger

from cam_server import config

_logger = getLogger(__name__)


pipeline_name_to_pipeline_module_mapping = {
    config.PIPELINE_TYPE_PROCESSING: processing,
    config.PIPELINE_TYPE_STORE: store,
    config.PIPELINE_TYPE_STREAM: stream,
    config.PIPELINE_TYPE_CUSTOM: custom,
    config.PIPELINE_TYPE_SCRIPT: config.PIPELINE_TYPE_SCRIPT
}

def get_builtin_pipelines():
    return list(pipeline_name_to_pipeline_module_mapping.keys())

def get_pipeline_function(pipeline_type_name):
    builtin_pipelines = get_builtin_pipelines()
    if pipeline_type_name not in builtin_pipelines:
        raise ValueError("pipeline_type '%s' not present in mapping. Available: %s." %
                         (pipeline_type_name, builtin_pipelines) )
    if pipeline_type_name == config.PIPELINE_TYPE_SCRIPT:
        return Exception("Cannot evaluated script pipelines")
    return pipeline_name_to_pipeline_module_mapping[pipeline_type_name].run
