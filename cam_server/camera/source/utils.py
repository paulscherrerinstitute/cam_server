from cam_server.camera.source.bsread import CameraBsread, CameraBsreadSim
from cam_server.camera.source.epics import CameraEpics
from cam_server.camera.source.simulation import CameraSimulation

source_type_to_source_class_mapping = {
    "epics": CameraEpics,
    "simulation": CameraSimulation,
    "bsread": CameraBsread,
    "bsread_simulation": CameraBsreadSim
}


def get_source_class(source_type_name):
    if source_type_name not in source_type_to_source_class_mapping:
        raise ValueError("source_type '%s' not present in source class mapping. Available: %s." %
                         (source_type_name, list(source_type_to_source_class_mapping.keys())))

    return source_type_to_source_class_mapping[source_type_name]
