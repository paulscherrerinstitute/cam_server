import json
from collections import deque
from logging import getLogger

import numpy as np
_logger = getLogger(__name__)

background = deque(maxlen=4)

DEFAULT_ROI_SIGNAL = None
DEFAULT_ROI_BACKGROUND = None


def get_roi_x_profile(image, roi):
    offset_x, size_x, offset_y, size_y = roi
    roi_image = image[offset_y:offset_y + size_y, offset_x:offset_x + size_x]

    return roi_image.sum(0)


def find_edge(data, step_length=50, edge_type='falling', refinement=1):
    # refine data
    def _interp(fp, xp, x):  # utility function to be used with apply_along_axis
        return np.interp(x, xp, fp)

    data_length = data.shape[1]
    refined_data = np.apply_along_axis(
        _interp,
        axis=1,
        arr=data,
        x=np.arange(0, data_length - 1, refinement),
        xp=np.arange(data_length),
    )

    # prepare a step function and refine it
    step_waveform = np.ones(shape=(step_length,))
    if edge_type == 'rising':
        step_waveform[: int(step_length / 2)] = -1
    elif edge_type == 'falling':
        step_waveform[int(step_length / 2) :] = -1

    step_waveform = np.interp(
        x=np.arange(0, step_length - 1, refinement),
        xp=np.arange(step_length),
        fp=step_waveform,
    )

    # find edges
    xcorr = np.apply_along_axis(np.correlate, 1, refined_data, v=step_waveform, mode='valid')
    edge_position = np.argmax(xcorr, axis=1).astype(float) * refinement
    xcorr_amplitude = np.amax(xcorr, axis=1)

    # correct edge_position for step_length
    edge_position += np.floor(step_length / 2)

    return {'edge_pos': edge_position, 'xcorr_ampl': xcorr_amplitude}


def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters):

    processed_data = dict()

    image_property_name = parameters["camera_name"]
    roi_signal = parameters.get("roi_signal", DEFAULT_ROI_SIGNAL)
    roi_background = parameters.get("roi_background", DEFAULT_ROI_BACKGROUND)

    processed_data[image_property_name + ".processing_parameters"] = json.dumps({"roi_signal": roi_signal,
                                                                                 "roi_background": roi_background})

    # output = {}

    if roi_signal:
        signal_profile = get_roi_x_profile(image, roi_signal)
        processed_data[image_property_name + ".roi_signal_x_profile"] = signal_profile

        # if pulse_id % 4 == 0:
        #     # fel shot
        #     if background:
        #         avg_background = sum(background) / len(background)
        #         output = find_edge(signal_profile - avg_background)
        #     else:
        #         output['edge_pos'] = np.nan
        #         output['xcorr_ampl'] = np.nan
        # else:
        #     background.append(signal_profile)
        #     output['edge_pos'] = np.nan
        #     output['xcorr_ampl'] = np.nan
        #
        # processed_data[image_property_name + ".edge_position"] = output['edge_pos']
        # processed_data[image_property_name + ".cross_correlation_amplitude"] = output['xcorr_ampl']

    if roi_background:
        processed_data[image_property_name + ".roi_background_x_profile"] = get_roi_x_profile(image, roi_background)

    return processed_data
