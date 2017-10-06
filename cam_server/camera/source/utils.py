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


def get_calibrated_axis(calibration, height, width):
    """
    Get x and y axis in nm based on calculated origin from the reference markers
    The coordinate system looks like this:
           +|
    +       |
    -----------------
            |       -
           -|
    Parameters
    ----------
    width       image with in pixel
    height      image height in pixel
    Returns
    -------
    (x_axis, y_axis)
    """

    if not calibration:
        x_axis = numpy.linspace(0, width - 1, width, dtype='f')
        y_axis = numpy.linspace(0, height - 1, height, dtype='f')

        return x_axis, y_axis

    def _calculate_center():
        center_x = int(((lower_right_x - upper_left_x) / 2) + upper_left_x)
        center_y = int(((lower_right_y - upper_left_y) / 2) + upper_left_y)

        return center_x, center_y

    def _calculate_pixel_size():
        size_y = reference_marker_height / (lower_right_y - upper_left_y)
        size_y *= numpy.cos(vertical_camera_angle * numpy.pi / 180)

        size_x = reference_marker_width / (lower_right_x - upper_left_x)
        size_x *= numpy.cos(horizontal_camera_angle * numpy.pi / 180)

        return size_x, size_y

    upper_left_x, upper_left_y, lower_right_x, lower_right_y = calibration["reference_marker"]

    reference_marker_height = calibration["reference_marker_height"]
    vertical_camera_angle = calibration["angle_vertical"]
    reference_marker_width = calibration["reference_marker_width"]
    horizontal_camera_angle = calibration["angle_horizontal"]

    # Derived properties
    origin_x, origin_y = _calculate_center()
    pixel_size_x, pixel_size_y = _calculate_pixel_size()  # pixel size in nanometer

    x_axis = numpy.linspace(0, width - 1, width, dtype='f')
    x_axis -= origin_x
    x_axis *= (-pixel_size_x)  # we need the minus to invert the axis
    y_axis = numpy.linspace(0, height - 1, height, dtype='f')
    y_axis -= origin_y
    y_axis *= (-pixel_size_y)  # we need the minus to invert the axis

    return x_axis, y_axis
