from logging import getLogger

from cam_server.pipeline.data_processing import functions

import json

import numpy
import scipy.signal
import scipy.optimize

import epics


_logger = getLogger(__name__)

output_pv, center_pv, fwhm_pv, ymin_pv, ymax_pv, axis_pv  = None, None, None, None, None, None
roi = [0, 0]
initialized = False


def initialize(parameters):
    global ymin_pv, ymax_pv, axis_pv, output_pv, center_pv, fwhm_pv

    epics_pv_name_prefix = parameters["camera_name"]
    output_pv_name = epics_pv_name_prefix + ":SPECTRUM_Y"
    center_pv_name = epics_pv_name_prefix + ":SPECTRUM_CENTER"
    fwhm_pv_name = epics_pv_name_prefix + ":SPECTRUM_FWHM"
    ymin_pv_name = epics_pv_name_prefix + ":SPC_ROI_YMIN"
    ymax_pv_name = epics_pv_name_prefix + ":SPC_ROI_YMAX"
    axis_pv_name = epics_pv_name_prefix + ":SPECTRUM_X"
    epics.ca.clear_cache()

    output_pv = epics.PV(output_pv_name)
    center_pv = epics.PV(center_pv_name)
    fwhm_pv = epics.PV(fwhm_pv_name)

    ymin_pv = epics.PV(ymin_pv_name)
    ymax_pv = epics.PV(ymax_pv_name)
    axis_pv = epics.PV(axis_pv_name)
    ymin_pv.wait_for_connection()
    ymax_pv.wait_for_connection()
    axis_pv.wait_for_connection()


def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata=None):
    global roi, initialized
    global ymin_pv, ymax_pv, axis_pv, output_pv, center_pv, fwhm_pv

    if not initialized:
        initialize(parameters)
        initialized = True

    processed_data = dict()

    epics_pv_name_prefix = parameters["camera_name"]

    if ymin_pv and ymin_pv.connected:
        roi[0] = ymin_pv.value
    if ymax_pv and ymax_pv.connected:
        roi[1] = ymax_pv.value
    if axis_pv and axis_pv.connected:
        axis = axis_pv.value
    else:
        axis = None

    if axis is None or len(axis) != image.shape[1]:
        _logger.warning("Invalid energy axis")
        return None

    #processed_data[epics_pv_name_prefix + ":processing_parameters"] = json.dumps({"roi": roi, "background": parameters['background']})
    parameters["roi"] = roi
    processed_data[epics_pv_name_prefix + ":processing_parameters"] = json.dumps(parameters)


    processing_image = image
    nrows, ncols = processing_image.shape

    """
    # validate background data
    background_image = parameters.get('background_data')
    if isinstance(background_image, numpy.ndarray):
        if background_image.shape != processing_image.shape:
            background_image = None
    else:
        background_image = None
    """

    # crop the image in y direction
    ymin, ymax = roi
    if nrows >= ymax > ymin >= 0:
        if (nrows != ymax) or (ymin != 0):
            processing_image = processing_image[int(ymin):int(ymax), :]
        """
        if background_image is not None:
            background_image = background_image[ymin:ymax, :]

    # remove the background and collapse in y direction to get the spectrum
    if background_image is not None:
        spectrum = functions.get_spectrum(processing_image, background_image)
    else:
        """
    spectrum = processing_image.sum(0, 'uint32')

    # smooth the spectrum with savgol filter with 51 window size and 3rd order polynomial
    smoothed_spectrum = scipy.signal.savgol_filter(spectrum, 51, 3)

    # check wether spectrum has only noise. the average counts per pixel at the peak
    # should be larger than 1.5 to be considered as having real signals.
    minimum, maximum = smoothed_spectrum.min(), smoothed_spectrum.max()
    amplitude = maximum - minimum
    skip = True
    if amplitude > nrows * 1.5:
        skip = False
    # gaussian fitting
    offset, amplitude, center, sigma = functions.gauss_fit_psss(smoothed_spectrum[::2], axis[::2],
            offset=minimum, amplitude=amplitude, skip=skip)

    # outputs
    processed_data[epics_pv_name_prefix + ":SPECTRUM_Y"] = spectrum
    processed_data[epics_pv_name_prefix + ":SPECTRUM_X"] = axis
    processed_data[epics_pv_name_prefix + ":SPECTRUM_CENTER"] = center
    processed_data[epics_pv_name_prefix + ":SPECTRUM_FWHM"] = 2.355 * sigma

    if output_pv and output_pv.connected:
        output_pv.put(processed_data[epics_pv_name_prefix + ":SPECTRUM_Y"])
        _logger.debug("caput on %s for pulse_id %s", output_pv, pulse_id)

    if center_pv and center_pv.connected:
        center_pv.put(processed_data[epics_pv_name_prefix + ":SPECTRUM_CENTER"])

    if fwhm_pv and fwhm_pv.connected:
        fwhm_pv.put(processed_data[epics_pv_name_prefix + ":SPECTRUM_FWHM"])

    return processed_data
