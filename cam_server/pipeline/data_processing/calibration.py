import numpy


class Calibration:
    def __init__(self, reference_marker=(0, 0, 100, 100), reference_marker_width=100, reference_marker_height=100,
                 horizontal_camera_angle=0.0, vertical_camera_angle=0.0):

        # Tuple of the pixel coordinates of the reference markers
        # (upper_left_x, upper_left_y, lower_right_x, lower_right_y)
        self.reference_marker = reference_marker
        self.reference_marker_width = reference_marker_width  # x distance between marker in nanometer
        self.reference_marker_height = reference_marker_height  # y distance between marker in nanometer

        self.horizontal_camera_angle = horizontal_camera_angle
        self.vertical_camera_angle = vertical_camera_angle

        # Derived properties
        self.origin = (0, 0)  # (x, y) - center in pixel coordinates
        self.pixel_size_x = 0  # pixel size in nanometer
        self.pixel_size_y = 0  # pixel size in nanometer

        # Calculate properties
        self._calculate_center()
        self._calculate_pixel_size()

        # self.x_axis, self.y_axis = self._get_x_y_axis(width, height)

    def _calculate_center(self):  # origin
        upper_left_x, upper_left_y, lower_right_x, lower_right_y = self.reference_marker

        center_x = int(((lower_right_x - upper_left_x) / 2) + upper_left_x)
        center_y = int(((lower_right_y - upper_left_y) / 2) + upper_left_y)

        self.origin = (center_x, center_y)

    def _calculate_pixel_size(self):

        upper_left_x, upper_left_y, lower_right_x, lower_right_y = self.reference_marker

        self.pixel_size_y = self.reference_marker_height / (lower_right_y - upper_left_y)
        self.pixel_size_y *= numpy.cos(self.vertical_camera_angle * numpy.pi / 180)

        self.pixel_size_x = self.reference_marker_width / (lower_right_x - upper_left_x)
        self.pixel_size_x *= numpy.cos(self.horizontal_camera_angle * numpy.pi / 180)

    def get_x_y_axis(self, width, height):
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
        origin_x, origin_y = self.origin
        x_axis = numpy.linspace(0, width - 1, width, dtype='f')
        x_axis -= origin_x
        x_axis *= (-self.pixel_size_x)  # we need the minus to invert the axis

        y_axis = numpy.linspace(0, height - 1, height, dtype='f')
        y_axis -= origin_y
        y_axis *= (-self.pixel_size_y)  # we need the minus to invert the axis

        return x_axis, y_axis
