import sys
import os

from cam_server.pipeline.data_processing import functions

# Emittance measurement. Method determines the vertical electron
# beam size using vertically polarized synchrotron radiation in
# the visible to uv range. Images are acquired by the pi-polarization
# method.

import numpy as np
from scipy import interpolate
from scipy.signal import find_peaks
import math
from logging import getLogger

_logger = getLogger(__name__)


class PpolPeakValley():
    def __init__(self):
        self.x, self.y = self.ppol_interpol()

    def ppol_interpol(self, plot_flag=False):
        ipv = [32.976, 20.848, 15.306, 12.121, 10.0499899, 8.595, 7.5176, 6.6871, 6.0276, 5.491, 5.0471, 4.673, 4.353,
               4.0778, 3.837, 3.439, 3.122, 2.865, 2.652, 2.473, 2.32]
        sig = [3.659, 5.175, 6.338, 7.318, 8.182, 8.963, 9.681, 10.350, 10.978, 11.572, 12.136, 12.676, 13.194, 13.692,
               14.172, 15.087, 15.950, 16.769, 17.549, 18.296, 19.014]
        sig2 = [element * element for element in sig]
        x = np.linspace(0, 999999, 999999)
        x = [(1 + val) / 1000000 * max(ipv) for val in x]
        interpfunc = interpolate.interp1d(ipv, sig2, kind='quadratic')
        inter_x = interpfunc(x)
        sqrt_val = [math.sqrt(val) for val in inter_x]
        y = [0]
        y.extend(sqrt_val)
        y.append(max(sig))
        xf = [0]
        xf.extend(x)
        xf.append(max(ipv))
        return xf, y

    def get_emittance(self, ratio):
        emittance = 0
        for i in range(0, len(self.x)):
            if self.x[i] > ratio:
                self.y[i - 1]
                return self.y[i - 1]


# For peak search - delta(h) to max peak value
DELTA_HEIGHT = 400  # 400
BG_XRANGE_LOW = [340, 400]  # 100 160 [340, 400]
BG_XRANGE_HIGH = [460, 520]  # 840 900 [460, 520]

PEAK_SEARCH_REL_RANGE = [-1, 2]
VALLEY_SEARCH_REL_RANGE = [-2, 3]
ppol = PpolPeakValley()

# abreviations
BX1 = BG_XRANGE_LOW[0]
BX2 = BG_XRANGE_LOW[1]
BX3 = BG_XRANGE_HIGH[0]
BX4 = BG_XRANGE_HIGH[1]
plr = PEAK_SEARCH_REL_RANGE
vlr = VALLEY_SEARCH_REL_RANGE

ydata = []


def calculate_emittance(image, fit_pars):
    global ydata  # array of indexes
    DELTA_HEIGHT = fit_pars['delta_height']
    BG_XRANGE_LOW = fit_pars['bg_range_low']
    BG_XRANGE_HIGH = fit_pars['bg_range_high']
    PEAK_SEARCH_REL_RANGE = fit_pars['peak_search_rel_range']
    VALLEY_SEARCH_REL_RANGE = fit_pars['valley_search_rel_range']

    w, h = len(image[0]), len(image)
    if len(ydata) != h:
        H = []
        for i in range(0, h):
            H.append(i)
        ydata = H[:int(h)]

    peak_array = [None] * 2
    proj_peak_array = [None] * 2
    peak_value = [None] * 2
    peak_bg = [None] * 2
    image -= image.min()
    projy = np.sum(image, axis=1)
    projx = np.sum(image, axis=0)
    # find peaks
    max_element = np.amax(projy)
    max_indices = np.where(projy == max_element)
    peaks, _ = find_peaks(projy, height=(max_element - DELTA_HEIGHT))

    _logger.debug("max indices /peaks " + str(max_indices) + " " + str(peaks))

    if len(peaks) != 2:
        mess = "Too few peaks found! " if len(peaks) < 2 else \
            "Too many peaks found "
        _logger.debug(mess + str(peaks))
        peaks_buffer = []
        for val in peaks:
            ### COMMENTED BY ALEX
            # if val > 567 and val < 590:
            peaks_buffer.append(val)

        # if len(peaks_buffer) ==3:
        #    peaks = [None] * 2
        #    peaks[0] = peaks_buffer[0]
        #    peaks[1] = peaks_buffer[2]
        if len(peaks_buffer) != 2:
            return (-1.0, -2.0)
        else:
            peaks = peaks_buffer

    if (peaks[1] - peaks[0]) < 6:
        _logger.debug("Peaks are too close: " + str(peaks[1] - peaks[0]))
        raise Exception("Peaks are too close")

    # peaks =[569, 577]
    # Distance to minimum
    min_element = np.amin(projy[peaks[0]:peaks[1]])
    min_indices = np.where(projy == min_element)

    min_idx_value = 0  # min_indices[0][0]
    for val in min_indices[0]:
        if val > peaks[0] and val < peaks[1]:
            min_idx_value = val
            break

    if min_idx_value == 0:
        raise Exception("min_idx_value == 0")

    for i in range(0, len(peak_array)):
        peak_array[i] = ydata[peaks[i] + plr[0]: peaks[i] + plr[1]]
    valley_array = ydata[min_idx_value + vlr[0]: min_idx_value + vlr[1]]

    # print("peaks", peaks, flush=True)
    # print(np.subtract(peaks, h)*(-1))
    # print("projections peak, valley", projy[(peaks[0]-1):(peaks[1]+2)], projy[valley_array])
    # print(peak_array, valley_array, flush=True)
    # x_bg_center = min_indices[0][0]

    # background
    bg_y1 = projy[BX1: BX2]
    bg_y2 = projy[BX3: BX4]

    bg_yS = np.concatenate((bg_y1, bg_y2))
    bg_xS = list(range(BX1, BX2)) + list(range(BX3, BX4))

    bg_x = []
    bg_y = []

    for x, y in zip(bg_xS, bg_yS):
        ### COMMENTED BY ALEX
        # if y > 800 and y < 1000:
        bg_x.append(x)
        bg_y.append(y)

    # print(bg_x, flush=True)
    # print(bg_y, flush=True)
    # fit
    poly_bg = np.polyfit(bg_x, bg_y, deg=1)
    array_bg = np.linspace(0, h, 10400)
    val_bg = np.polyval(poly_bg, array_bg)

    for i in range(0, len(proj_peak_array)):
        proj_peak_array[i] = projy[peaks[i] + plr[0]: peaks[i] + plr[1]]
    # proj_peak_array[1] = projy[peaks[1]-1 : peaks[1]+2]
    proj_valley = projy[min_idx_value + vlr[0]: min_idx_value + vlr[1]]

    # peaks
    for i in range(0, 2):
        poly = np.polyfit(peak_array[i], proj_peak_array[i], deg=2)
        idx = -poly[1] / 2 / poly[0]
        peak_value[i] = np.polyval(poly, idx)
        peak_bg[i] = val_bg[int(idx)]

    # valley
    poly2 = np.polyfit(valley_array, proj_valley, deg=2)
    # Only works for deg=2
    minv_idx = -poly2[1] / 2 / poly2[0]

    poly = np.polyfit(valley_array, proj_valley, deg=4)

    valley_subarray = np.linspace(valley_array[0], valley_array[-1], 800)
    poly_array = np.polyval(poly, valley_subarray)
    valley_fitted_value = min(poly_array)

    # print("peak value", peak_value, "valley_fitted value", valley_fitted_value, flush=True)

    valley_bg = val_bg[int(minv_idx)]
    # print("valley background", valley_bg, "peak background", peak_bg[0], peak_bg[1], flush=True)

    ### COMMENTED BY ALEX
    # if valley_fitted_value <  valley_bg:
    #    valley_fitted_value = min( projy[valley_array])
    #    if valley_fitted_value <  valley_bg:
    #        raise Exception("valley_fitted_value <  valley_bg")
    #        #return

    ratio_corrected = 2 * (valley_fitted_value - valley_bg) / (
        abs((peak_value[0] - peak_bg[0]) + (peak_value[1] - peak_bg[1])))

    idx = 0 if abs(peak_value[0] - peak_bg[0]) > abs(peak_value[1] - peak_bg[1]) else 1
    ratio_max = (valley_fitted_value - valley_bg) / abs(peak_value[idx] - peak_bg[idx])

    emittance = ppol.get_emittance(ratio_corrected)
    emittance2 = ppol.get_emittance(ratio_max)

    _logger.debug("ratio=%f emittance %f ratio2=%f emittance2 %f" % (ratio_corrected, emittance, ratio_max, emittance2))
    return (emittance, emittance2)


def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata=None):
    channel_prefix = parameters["camera_name"]
    emittance = emittance2 = float("NaN")
    status = "Ok"
    try:
        ret = calculate_emittance(image, parameters['fit'])
        if ret is not None:
            emittance, emittance2 = ret

    except Exception as e:
        status = "Error: "
        exc_type, exc_obj, exc_tb = sys.exc_info()
        while exc_tb is not None:
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            status = status + fname + " line " + str(exc_tb.tb_lineno) + ": " + str(e) + " | "
            exc_tb = exc_tb.tb_next

    ret = {}
    ret[channel_prefix + ":emmitance"] = emittance
    ret[channel_prefix + ":emmitance2"] = emittance2
    ret[channel_prefix + ":status"] = str(status)
    return ret
