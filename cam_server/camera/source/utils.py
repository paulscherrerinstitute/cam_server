from imp import load_source
from importlib import import_module
from logging import getLogger

from cam_server.camera.source.area_detector import AreaDetector
from cam_server.camera.source.bsread import CameraBsread, CameraBsreadSim
from cam_server.camera.source.epics import CameraEpics
from cam_server.camera.source.simulation import CameraSimulation
from cam_server.camera.source.stream import CameraStream

_logger = getLogger(__name__)

source_type_to_source_class_mapping = {
    "epics": CameraEpics,
    "simulation": CameraSimulation,
    "bsread": CameraBsread,
    "area_detector": AreaDetector,
    "bsread_simulation": CameraBsreadSim,
    "stream": CameraStream,
    "script": None
}

_user_scripts_manager = None

def is_builtin_source(source_type_name):
    return source_type_name in source_type_to_source_class_mapping

def get_source_class(camera_config):
    source_type_name = camera_config.get_source_type()
    if source_type_name not in source_type_to_source_class_mapping:
        raise ValueError("source_type '%s' not present in source class mapping. Available: %s." %
                         (source_type_name, list(source_type_to_source_class_mapping.keys())))
    if source_type_name == "script":
        name = ""
        try:
            name = camera_config.parameters.get("class")
            _logger.info("Importing source: %s." % (name))
            module_name = name  # 'mod'
            if '/' in name:
                mod = load_source(module_name, name)
            else:
                if _user_scripts_manager and _user_scripts_manager.exists(name):
                    mod = load_source(module_name, _user_scripts_manager.get_path(name))
                else:
                    mod = import_module("cam_server.camera.source." + str(name))
            cls = getattr(mod, name)
            if cls is not None:
                return cls
        except:
            _logger.exception("Could not import function: %s." % (str(name)))

        raise ValueError("Invalid custom source")

    return source_type_to_source_class_mapping[source_type_name]
