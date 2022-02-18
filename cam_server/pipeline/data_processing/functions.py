import io
from logging import getLogger

import numpy
import scipy
import scipy.misc
import scipy.optimize
import scipy.ndimage
from PIL import Image

from matplotlib import cm

from cam_server import config

_logging = getLogger(__name__)


def subtract_background(image, background_image):
    # We do not want negative numbers int the image.
    if background_image is not None:
        if image.shape != background_image.shape:
            raise RuntimeError("Invalid background_image size %s compared to image %s" % (background_image.shape,
                                                                                          image.shape))
        image = enforce_image_writable(image)
        mask_for_zeros = (background_image > image)
        numpy.subtract(image, background_image, image)
        image[mask_for_zeros] = 0
    return image

def subtract_background_signed(image, background_image):
    # We do not want negative numbers int the image.
    image = image.astype("int32")
    if background_image is not None:
        image = enforce_image_writable(image)
        numpy.subtract(image, background_image, image)
    return image


def is_number(var):
    try:
        float(var)
        return True
    except:
        return False


def rotate(image, degrees, order = 1, mode = "0.0"):
    if mode == "ortho":
        output = numpy.rot90(image, int(degrees/90))
    else:
        output = scipy.ndimage.rotate(image, float(degrees), reshape=False, order=order,
                             mode="constant" if is_number(mode) else mode,
                             cval=float(mode) if is_number(mode) else 0.0, prefilter=True)
    return output

def get_region_of_interest(image, offset_x, size_x, offset_y, size_y):
    return image[offset_y:offset_y + size_y, offset_x:offset_x + size_x]


def apply_threshold(image, threshold=1):
    image=enforce_image_writable(image)
    image[image < int(threshold)] = 0


def get_min_max(image):
    return numpy.nanmin(image), numpy.nanmax(image)


def get_x_y_profile(image):
    x_profile = image.sum(0)
    y_profile = image.sum(1)
    return x_profile, y_profile


def get_intensity(profile):
    return profile.sum()


def find_index(axis, item):
    """ Find the index of the axis value that corresponds to the passed value/item"""

    left_bigger_right = axis[0] > axis[1]  # if true axis looks like this [5, 4, 3, 2, 1, 0]

    # Descending order -> [9, 8, 7, 6]
    if left_bigger_right:
        # Item value 10 -> go to first section.
        if item > axis[0]:
            return 0

        # Item value 5 -> go to last section.
        if item < axis[-1]:
            return len(axis) - 1

        # Negate the array and number to search from the right.
        return numpy.searchsorted(-axis, -item)

    # Ascending order -> [6, 7, 8, 9]
    else:
        # Item value 5 -> go to first section.
        if item < axis[0]:
            return 0

        # Item value 10 -> go to last section.
        if item > axis[-1]:
            return len(axis) - 1

        insert_index = numpy.searchsorted(axis, item)

        # If the value is the same as the array value at the given index, use this index directly.
        if insert_index < len(axis) and axis[insert_index] == item:
            return insert_index

        # Otherwise return the previous section index.
        return insert_index - 1


def get_good_region_profile(profile, threshold=0.3, gfscale=1.8):
    profile_min = profile.min()
    threshold_value = (profile.max() - profile_min) * threshold + profile_min

    # The center of the good region is defined by the index of the max value of the profile
    index_maximum = profile.argmax()

    index_start = index_maximum
    index_end = index_maximum

    for i in range(index_maximum, 0, -1):
        if profile[i] < threshold_value:
            index_start = i
            break

    for i in range(index_maximum, profile.shape[0]):
        if profile[i] < threshold_value:
            index_end = i
            break

    # Extend the good region based on gfscale
    gf_extend = (index_end - index_start) * gfscale - (index_end - index_start)

    index_start -= gf_extend / 2
    index_end += gf_extend / 2

    index_start = index_start if index_start > 0 else 0
    index_end = index_end if index_end < profile.size - 1 else profile.size - 1

    return int(index_start), int(index_end)  # Start and end index of the good region


def gauss_fit(profile, axis):
    if axis.shape[0] != profile.shape[0]:
        raise RuntimeError("Invalid axis passed %d %d" % (axis.shape[0], profile.shape[0]))

    center_of_mass = (axis * profile).sum() / profile.sum()
    center_of_mass_2 = (axis * axis * profile).sum() / profile.sum()
    rms = numpy.sqrt(numpy.abs(center_of_mass_2 - center_of_mass * center_of_mass))

    offset, amplitude, center, standard_deviation = _gauss_fit(axis, profile)
    gauss_function = _gauss_function(axis, offset, amplitude, center, standard_deviation)

    return gauss_function, offset, amplitude, center, abs(standard_deviation), center_of_mass, rms


def _gauss_function(x, offset, amplitude, center, standard_deviation):
    # return offset + amplitude * numpy.exp(-(numpy.power((x - center), 2) / (2 * numpy.power(standard_deviation, 2))))
    return offset + amplitude * numpy.exp(-(x - center) ** 2 / (2 * standard_deviation ** 2))

def _gauss_fit(axis, profile, center_of_mass=None):

    offset = profile.min()  # Minimum is good estimation of offset
    amplitude = profile.max() - offset  # Max value is a good estimation of amplitude

    if center_of_mass:
        center = center_of_mass  # Center of mass is a good estimation of center (mu)
    else:
        center = axis[profile.argmax()]

    surface = numpy.trapz((profile - offset), x=axis)
    # standard_deviation = surface / ((amplitude - offset) * numpy.sqrt(2 * numpy.pi))
    standard_deviation = surface / (amplitude * numpy.sqrt(2 * numpy.pi))

    try:
        # It shows up that fastest fitting is when sampling period is around sigma value
        optimal_parameter, _ = scipy.optimize.curve_fit(_gauss_function, axis, profile.astype("float32"),
                                                        p0=[offset, amplitude, center, standard_deviation])
    except BaseException as e:
        # print(e)
        # logging.info("COULD NOT CONVERGE!")
        # Make sure return always as same type
        optimal_parameter = numpy.array([offset, amplitude, center, standard_deviation]).astype("float64")

    return optimal_parameter


def slice_image(image, number_of_slices=1, vertical=False):
    """
    :param image:
    :param number_of_slices:
    :param vertical:            if vertical the axis to use is y, if not vertical the axis to use is x
    :return:
    """

    if vertical:
        image = image.T  # transpose

    slice_size = image.shape[0] / number_of_slices
    slices = numpy.empty((number_of_slices, image.shape[1]))

    for i in range(number_of_slices):
        slices[i] = image[i * slice_size:(i + 1) * slice_size, :].sum(0)

    return slices


def calculate_slices(axis, center, standard_deviation, scaling=2, number_of_slices=9):
    """ Calculate index list for slices based on the given axis """

    if number_of_slices % 2 == 0:
        raise ValueError("Number of slices must be odd.")

    size_slice = scaling * standard_deviation / number_of_slices

    index_center = find_index(axis, center)
    index_half_slice = find_index(axis, center + size_slice / 2)
    n_pixel_half_slice = abs(index_half_slice - index_center)

    if n_pixel_half_slice < 1:
        _logging.info('Calculated number of pixel of a slice size [%d] is less than 1 - default to 1',
                      n_pixel_half_slice)
        n_pixel_half_slice = 1

    n_pixel_slice = 2 * n_pixel_half_slice

    # Add middle slice - located half/half on center
    start_index = index_center - n_pixel_half_slice
    end_index = index_center + n_pixel_half_slice

    list_slices_indexes = []
    slice_length = None

    number_of_elements_axis = len(axis)

    if start_index >= 0 and end_index < number_of_elements_axis:

        list_slices_indexes.append(start_index)
        list_slices_indexes.append(end_index)

        # The slice length is the difference in axis value from the start to the end of the axis.
        slice_length = abs(axis[start_index] - axis[end_index])

        # We subtract 1 because we already added the middle slice.
        counter_slices = number_of_slices - 1

        # Calculate outer slices
        while counter_slices > 0:
            start_index -= n_pixel_slice
            end_index += n_pixel_slice
            if start_index < 0 or end_index >= number_of_elements_axis:
                _logging.info('Stopping slice calculation as they are out of range ...')
                # Start index cannot be smaller than 0 and end index cannot e larger than len(axis)
                break
            list_slices_indexes.insert(0, start_index)
            list_slices_indexes.append(end_index)

            counter_slices -= 2

    return list_slices_indexes, n_pixel_half_slice, slice_length


def get_x_slices_data(image, x_axis, y_axis, x_center, x_standard_deviation, scaling=2, number_of_slices=11):
    """
    Calculate slices and their statistics
    :return: <center [x,y]>, <standard deviation>, <intensity>
    """

    list_slices, n_pixel_half_slice, slice_length = calculate_slices(x_axis, x_center, x_standard_deviation, scaling,
                                                                     number_of_slices)

    slice_data = []

    for i in range(len(list_slices) - 1):
        if list_slices[i] < image.shape[-1] and list_slices[i + 1] < image.shape[-1]:
            # slices are within good region
            slice_n = image[:, list_slices[i]:list_slices[i + 1]]

            slice_y_profile = slice_n.sum(1)
            pixel_intensity = slice_n.sum()

            # Does x need to be the middle of slice? - currently it is
            center_x = x_axis[list_slices[i] + n_pixel_half_slice]

            gauss_function, offset, amplitude, center_y, standard_deviation, _, _ = gauss_fit(slice_y_profile, y_axis)
            slice_data.append(([center_x, center_y], standard_deviation, pixel_intensity))
        else:
            _logging.info('Drop slice')

    return slice_data, slice_length


def get_y_slices_data(image, x_axis, y_axis, y_center, y_standard_deviation, scaling=2, number_of_slices=11):
    """
    Calculate slices and their statistics
    :return: <center [x,y]>, <standard deviation>, <intensity>
    """

    list_slices, n_pixel_half_slice, slice_length = calculate_slices(y_axis, y_center, y_standard_deviation, scaling,
                                                                     number_of_slices)

    slice_data = []

    for i in range(len(list_slices) - 1):
        if list_slices[i] < image.shape[0] and list_slices[i + 1] < image.shape[0]:
            # slices are within good region
            slice_n = image[list_slices[i]:list_slices[i + 1], :]

            slice_x_profile = slice_n.sum(0)
            pixel_intensity = slice_n.sum()

            gauss_function, offset, amplitude, center_x, standard_deviation, _, _ = gauss_fit(slice_x_profile, x_axis)

            # Does x need to be the middle of slice? - currently it is
            slice_data.append(([center_x, y_axis[list_slices[i] + n_pixel_half_slice]], standard_deviation,
                               pixel_intensity))
        else:
            _logging.info('Drop slice')

    return slice_data, slice_length


def _linear_function(x, slope, offset):
    return slope * x + offset


def linear_fit(x, y):  # x/y arrays
    # offset = 0.0
    # slope = 0.1
    # optimal_parameter, covariance = scipy.optimize.curve_fit(_linear_function, x, y, p0=[slope, offset])
    optimal_parameter, covariance = scipy.optimize.curve_fit(_linear_function, x, y)  # No initial guesses

    return optimal_parameter  # returns [slope, offset]


def _quadratic_function(x, a, b, c):
    return a * x ** 2 + b * x + c


def quadratic_fit(x, y):
    optimal_parameter, covariance = scipy.optimize.curve_fit(_quadratic_function, x, y)

    return optimal_parameter


def get_png_from_image(image_raw_bytes, scale=None, min_value=None, max_value=None, colormap_name=None):
    """
    Generate an image from the provided camera.
    :param image_raw_bytes: Image bytes to turn into PNG
    :param scale: Scale the image.
    :param min_value: Min cutoff value.
    :param max_value: Max cutoff value.
    :param colormap_name: Colormap to use. See http://matplotlib.org/examples/color/colormaps_reference.html
    :return: PNG image.
    """

    image_raw_bytes = image_raw_bytes.astype("float64")

    if scale:
        shape_0 = int(image_raw_bytes.shape[0] * scale)
        shape_1 = int(image_raw_bytes.shape[1] * scale)
        sh = shape_0, image_raw_bytes.shape[0] // shape_0, shape_1, image_raw_bytes.shape[1] // shape_1
        image_raw_bytes = image_raw_bytes.reshape(sh).mean(-1).mean(1)

    if min_value:
        image_raw_bytes -= min_value
        image_raw_bytes[image_raw_bytes < 0] = 0

    if max_value:
        image_raw_bytes[image_raw_bytes > max_value] = max_value

    try:
        colormap_name = colormap_name or config.DEFAULT_CAMERA_IMAGE_COLORMAP
        # Available colormaps http://matplotlib.org/examples/color/colormaps_reference.html
        colormap = getattr(cm, colormap_name)

        # http://stackoverflow.com/questions/10965417/how-to-convert-numpy-array-to-pil-image-applying-matplotlib-colormap
        # normalize image to range 0.0-1.0
        image_raw_bytes *= 1.0 / image_raw_bytes.max()

        image = numpy.uint8(colormap(image_raw_bytes) * 255)
    except:
        raise ValueError("Unable to apply colormap '%s'. "
                         "See http://matplotlib.org/examples/color/colormaps_reference.html for available colormaps." %
                         colormap_name)

    n_image = Image.fromarray(image)

    """
    import tempfile
    #n_image = scipy.misc.toimage(image)
    tmp_file = tempfile.TemporaryFile()

    # https://github.com/python-pillow/Pillow/issues/1211
    # We do not use any compression for speed reasons
    # n_image.save('your_file.png', compress_level=0)
    n_image.save(tmp_file, 'png', compress_level=0)
    # n_image.save(tmp_file, 'jpeg', compress_level=0)  # jpeg seems to be faster

    tmp_file.seek(0)
    content = tmp_file.read()
    tmp_file.close()
    """

    with io.BytesIO() as output:
        n_image.save(output, "png", compress_level=0)
        content = output.getvalue()

    return content


def get_fwhm(x, y):
    return get_fw(x, y, 0.5)


def get_fw(x, y, threshold=0.5):
    try:
        ymax, ymin =  numpy.amax(y),  numpy.amin(y)
        hm =  (ymax - ymin) * threshold
        max_index, l_index, r_index = numpy.argmax(y), 0, len(x)-1

        for i in range(max_index-1, 0, -1):
            if (y[i] - ymin) <= hm:
                l_index = i
                break
        for i in range(max_index+1, len(x), 1):
            if (y[i] - ymin) <= hm:
                r_index = i
                break
        fwhm = abs(x[l_index] - x[r_index])
        return fwhm
    except:
        return 0.0


def _gauss_deriv(x, offset, amplitude, center, standard_deviation):
    fac = numpy.exp(-(x - center) ** 2 / (2 * standard_deviation ** 2))
    result = numpy.empty((4, x.size), dtype=x.dtype)
    result[0, :] = 1.0
    result[1, :] = fac
    result[2, :] = amplitude * fac * (x - center) / (standard_deviation ** 2)
    result[3, :] = amplitude * fac * ((x - center) ** 2) / (standard_deviation ** 3)
    return result

def gauss_fit_psss(profile, axis, **kwargs):
    if axis.shape[0] != profile.shape[0]:
        raise RuntimeError("Invalid axis passed %d %d" % (axis.shape[0], profile.shape[0]))

    offset = kwargs.get('offset', profile.min())  # Minimum is good estimation of offset
    amplitude = kwargs.get('amplitude', profile.max() - offset)  # Max value is a good estimation of amplitude
    center = kwargs.get('center', numpy.dot(axis,
                                            profile) / profile.sum())  # Center of mass is a good estimation of center (mu)
    # Consider gaussian integral is amplitude * sigma * sqrt(2*pi)
    standard_deviation = kwargs.get('standard_deviation', numpy.trapz((profile - offset), x=axis) / (
                amplitude * numpy.sqrt(2 * numpy.pi)))
    maxfev = kwargs.get('maxfev', 20)  # the default is 100 * (N + 1), which is over killing

    # If user requests fitting to be skipped, return the estimated parameters.
    if kwargs.get('skip', False):
        return offset, amplitude, center, abs(standard_deviation)

    try:
        optimal_parameter, _ = scipy.optimize.curve_fit(
            _gauss_function, axis, profile.astype("float64"),
            p0=[offset, amplitude, center, standard_deviation],
            jac=_gauss_deriv,
            col_deriv=1,
            maxfev=maxfev)
    except BaseException as e:
        #Make sure return always as same type
        optimal_parameter = numpy.array([offset, amplitude, center, standard_deviation]).astype("float64")

    offset, amplitude, center, standard_deviation = optimal_parameter
    return offset, amplitude, center, abs(standard_deviation)

def binning(image, x_axis, y_axis, bx, by, mean = False):
    sy, sx = (len(y_axis), len(x_axis)) if (image is None) else image.shape
    _sx, _sy = int(sx/bx) * bx, int(sy/by) * by
    if (sx != _sx) or (sy != _sy):
        sy, sx = _sy, _sx
        if image is not None:
            image = image[:sy,:sx]
        if x_axis is not None:
            x_axis = x_axis[:sx]
        if y_axis is not None:
            y_axis = y_axis[:sy]
    if x_axis is not None:
        type = x_axis.dtype
        x_axis = x_axis.reshape(int(sx / bx), int(bx)).mean(axis=(1)).astype(type, copy=False)
    if y_axis is not None:
        type = y_axis.dtype
        y_axis = y_axis.reshape(int(sy / by), int(by)).mean(axis=(1)).astype(type, copy=False)
    if image is not None:
        image = image.reshape(int(sy / by), int(by), int(sx / bx), int(bx))
        if mean:
            image = image.mean(axis=(1, 3))
        else:
            image = image.sum(axis=(1, 3))
            numpy.clip(image, 0, 0xFFFF, out=image)
        image = image.astype("uint16", copy=False)
    return image, x_axis, y_axis

def chunk_copy(image, max_chunk = 2000000):
    """
    Copies an image in slices so that each slice is never bigger than the hugepage size(2MB).
    This increases enormously performance.
    Server CPU consumption on a Mac, processing images at 5Hz, 16bits, with height = 960:
        width cpu(%)
        1280   95
        1100   95
        1093   95
        1092    5
        1050    5
        1000    5
    On Linux servers the outcome of frames bigger than the hugepage size is worse.
    NumPy is spawning the copying on all available cores, making the CPU consumption
    increase to 2.4K%. Probably a locking between the different threads.

    :param image: 
    :return: 
    """
    row_size = image.shape[1] * image.itemsize
    rows = len(image)
    chunk_rows = max_chunk // row_size if row_size else 0
    buffer = numpy.empty_like(image)
    pos = 0
    while pos < rows:
        next = max(min(pos+chunk_rows, rows), pos+1)
        buffer[pos:next] = image[pos:next]
        pos = next
    return buffer


def copy_image(image):
    if config.CHUNK_COPY_IMAGES:
        return chunk_copy(image)
    return numpy.array(image)


def enforce_image_writable(image):
    if not image.flags['WRITEABLE']:
        return copy_image(image)
    return image