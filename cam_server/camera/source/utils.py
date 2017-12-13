import numpy

from cam_server.camera.source.bsread import CameraBsread
from cam_server.camera.source.epics import CameraEpics
from cam_server.camera.source.simulation import CameraSimulation

source_type_to_source_class_mapping = {
    "epics": CameraEpics,
    "simulation": CameraSimulation,
    "bsread": CameraBsread
}


def get_source_class(source_type_name):
    if source_type_name not in source_type_to_source_class_mapping:
        raise ValueError("source_type '%s' not present in source class mapping. Available: %s." %
                         (source_type_name, list(source_type_to_source_class_mapping.keys())))

    return source_type_to_source_class_mapping[source_type_name]


def transform_image(image, camera_config):

    if camera_config.parameters["mirror_x"]:
        image = numpy.fliplr(image)

    if camera_config.parameters["mirror_y"]:
        image = numpy.flipud(image)

    image = numpy.rot90(image, camera_config.parameters["rotate"])

    return numpy.ascontiguousarray(image)
