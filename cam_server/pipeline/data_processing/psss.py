from logging import getLogger

from cam_server.pipeline.data_processing import functions
from cam_server.utils import create_thread_pvs, epics_lock
from collections import defaultdict, deque


import json

import numpy as np
import numpy
import scipy.signal
import scipy.optimize
import numba
import time
from threading import Thread

numba.set_num_threads(4)

_logger = getLogger(__name__)

channel_names = None
output_pv, center_pv, fwhm_pv, ymin_pv, ymax_pv, axis_pv = None, None, None, None, None, None
roi = [0, 0]
initialized = False
sent_pid = -1
nrows = 1
axis = None
buffer = deque(maxlen=5)


@numba.njit(parallel=False)
def get_spectrum(image, background):
    y = image.shape[0]
    x = image.shape[1]

    profile = np.zeros(x, dtype=np.float64)

    for i in numba.prange(y):
        for j in range(x):
            profile[j] += image[i, j] - background[i, j]
    return profile


def update_PVs(buffer, output_pv_name, center_pv_name, fwhm_pv_name,com_pv_name, std_pv_name):
    [output_pv, center_pv, fwhm_pv, com_pv, std_pv]  = create_thread_pvs([output_pv_name, center_pv_name, fwhm_pv_name,com_pv_name, std_pv_name])

    while True:
        time.sleep(0.1)
        try:
            rec = buffer.popleft()
        except:
            continue
        try:
            if output_pv and output_pv.connected and (rec[0] is not None):
                output_pv.put(rec[0])
            if center_pv and center_pv.connected and (rec[1] is not None):
                center_pv.put(rec[1])
            if fwhm_pv and fwhm_pv.connected and (rec[2] is not None):
                fwhm_pv.put(rec[2])
            if com_pv and com_pv.connected and (rec[3] is not None):
                com_pv.put(rec[3])
            if std_pv and std_pv.connected and (rec[4] is not None):
                std_pv.put(rec[4])
        except:
            _logger.exception("Error updating channels")

def initialize(params):
    global ymin_pv, ymax_pv, axis_pv, output_pv, center_pv, fwhm_pv, buffer
    global channel_names, spectra_buffer
    # fit.recompile()
    camera_name = params["camera_name"]
    output_pv_name = camera_name + ":SPECTRUM_Y"
    center_pv_name = camera_name + ":SPECTRUM_CENTER"
    fwhm_pv_name = camera_name + ":SPECTRUM_FWHM"
    ymin_pv_name = camera_name + ":SPC_ROI_YMIN"
    ymax_pv_name = camera_name + ":SPC_ROI_YMAX"
    axis_pv_name = camera_name + ":SPECTRUM_X"
    com_pv_name = camera_name + ":SPECTRUM_COM"
    std_pv_name = camera_name + ":SPECTRUM_STD"
    thread = Thread(target=update_PVs, args=(buffer, output_pv_name, center_pv_name, fwhm_pv_name,com_pv_name, std_pv_name))
    thread.start()
    channel_names = [ymin_pv_name, ymax_pv_name, axis_pv_name]


def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata=None, background=None):
    global roi, initialized, sent_pid, nrows, axis
    global channel_names, buffer

    if not initialized:
        initialize(parameters)
        initialized = True
    [ymin_pv, ymax_pv, axis_pv] = create_thread_pvs(channel_names)
    processed_data = dict()
    camera_name = parameters["camera_name"]

    if ymin_pv and ymin_pv.connected:
        roi[0] = ymin_pv.value
    if ymax_pv and ymax_pv.connected:
        roi[1] = ymax_pv.value
    if axis_pv and axis_pv.connected:
        axis = axis_pv.value
    else:
        axis = None

    if axis is None:
        _logger.warning("Energy axis not connected");
        return None

    if len(axis) < image.shape[1]:
        _logger.warning("Energy axis length %d < image width %d", len(axis), image.shape[1])
        return None

    # match the energy axis to image width
    axis = axis[:image.shape[1]]

    processing_image = image.astype(np.float32) - np.float32(parameters["pixel_bkg"])
    nrows, ncols = processing_image.shape

    # validate background data if passive mode (background subtraction handled here)
    background_image = parameters.pop('background_data', None)
    if isinstance(background_image, np.ndarray):
        background_image = background_image.astype(np.float32)
        if background_image.shape != processing_image.shape:
            _logger.info("Invalid background shape: %s instead of %s" % (
                str(background_image.shape), str(processing_image.shape)))
            background_image = None
    else:
        background_image = None

    processed_data[camera_name + ":processing_parameters"] = json.dumps(
        {"roi": roi, "background": None if (background_image is None) else parameters.get('image_background')})

    # crop the image in y direction
    ymin, ymax = int(roi[0]), int(roi[1])
    if nrows >= ymax > ymin >= 0:
        if (nrows != ymax) or (ymin != 0):
            processing_image = processing_image[ymin: ymax, :]
            if background_image is not None:
                background_image = background_image[ymin:ymax, :]

    # remove the background and collapse in y direction to get the spectrum
    if background_image is not None:
        spectrum = get_spectrum(processing_image, background_image)
    else:
        spectrum = np.sum(processing_image, axis=0)

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
                                                                offset=minimum, amplitude=amplitude, skip=skip,
                                                                maxfev=10)

    smoothed_spectrum_normed = smoothed_spectrum / np.sum(smoothed_spectrum)
    spectrum_com = np.sum(axis * smoothed_spectrum_normed)
    spectrum_std = np.sqrt(np.sum((axis - spectrum_com) ** 2 * smoothed_spectrum_normed))
    import signal
    # outputs
    processed_data[camera_name + ":SPECTRUM_Y"] = spectrum
    processed_data[camera_name + ":SPECTRUM_X"] = axis
    processed_data[camera_name + ":SPECTRUM_CENTER"] = np.float64(center)
    processed_data[camera_name + ":SPECTRUM_FWHM"] = np.float64(2.355 * sigma)
    processed_data[camera_name + ":SPECTRUM_COM"] = spectrum_com
    processed_data[camera_name + ":SPECTRUM_STD"] = spectrum_std

    if epics_lock.acquire(False):
        try:
            if pulse_id > sent_pid:
                sent_pid = pulse_id
                buffer.append( (processed_data[camera_name + ":SPECTRUM_Y"], processed_data[camera_name + ":SPECTRUM_CENTER"],
                                processed_data[camera_name + ":SPECTRUM_FWHM"], processed_data[camera_name + ":SPECTRUM_COM"],
                                processed_data[camera_name + ":SPECTRUM_STD"]))
        finally:
            epics_lock.release()
    return processed_data





