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
    config.PIPELINE_TYPE_CUSTOM: custom
}


def get_pipeline_function(pipeline_type_name):
    if pipeline_type_name not in pipeline_name_to_pipeline_module_mapping:
        raise ValueError("pipeline_type '%s' not present in mapping. Available: %s." %
                         (pipeline_type_name, list(pipeline_name_to_pipeline_module_mapping.keys())))

    return pipeline_name_to_pipeline_module_mapping[pipeline_type_name].run
