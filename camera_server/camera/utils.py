import tempfile
from logging import getLogger

import numpy
import scipy
from matplotlib import cm

from camera_server import config

_logging = getLogger(__name__)


def get_image_from_camera(camera, raw, scale, min_value, max_value, colormap_name):
    camera.connect()
    image = camera.get_image(raw=raw)
    camera.disconnect()

    if scale:
        shape_0 = int(image.shape[0] * scale)
        shape_1 = int(image.shape[1] * scale)
        sh = shape_0, image.shape[0] // shape_0, shape_1, image.shape[1] // shape_1
        image = image.reshape(sh).mean(-1).mean(1)

    if min_value:
        image -= min_value
        image[image < 0] = 0

    if max_value:
        image[image > max_value] = max_value

    try:
        colormap_name = colormap_name or config.DEFAULT_CAMERA_IMAGE_COLORMAP
        # Available colormaps http://matplotlib.org/examples/color/colormaps_reference.html
        colormap = getattr(cm, colormap_name)

        # http://stackoverflow.com/questions/10965417/how-to-convert-numpy-array-to-pil-image-applying-matplotlib-colormap
        # normalize image to range 0.0-1.0
        image *= 1.0 / image.max()

        image = numpy.uint8(colormap(image) * 255)
    except:
        raise ValueError("Unable to apply colormap '%s'. "
                         "See http://matplotlib.org/examples/color/colormaps_reference.html for available colormaps." %
                         colormap_name)

    n_image = scipy.misc.toimage(image)

    tmp_file = tempfile.TemporaryFile()

    # https://github.com/python-pillow/Pillow/issues/1211
    # We do not use any compression for speed reasons
    # n_image.save('your_file.png', compress_level=0)
    n_image.save(tmp_file, 'png', compress_level=0)
    # n_image.save(tmp_file, 'jpeg', compress_level=0)  # jpeg seems to be faster

    tmp_file.seek(0)
    content = tmp_file.read()
    tmp_file.close()

    return content
