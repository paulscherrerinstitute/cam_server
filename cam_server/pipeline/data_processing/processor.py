import json
from logging import getLogger

from cam_server.pipeline.data_processing import functions

_logger = getLogger(__name__)


def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata=None):

    # Add return values
    return_value = dict()

    image_threshold = parameters.get("image_threshold")
    if image_threshold is not None and image_threshold > 0:
        functions.apply_threshold(image, image_threshold)

    image_region_of_interest = parameters.get("image_region_of_interest")
    if image_region_of_interest:
        offset_x, size_x, offset_y, size_y = image_region_of_interest
        image = functions.get_region_of_interest(image, offset_x, size_x, offset_y, size_y)

        # Apply roi to geometry x_axis and y_axis
        x_axis = x_axis[offset_x:offset_x + size_x]
        y_axis = y_axis[offset_y:offset_y + size_y]

    (min_value, max_value) = functions.get_min_max(image)

    (x_profile, y_profile) = functions.get_x_y_profile(image)

    x_fit = functions.gauss_fit(x_profile, x_axis)
    y_fit = functions.gauss_fit(y_profile, y_axis)

    x_fit_gauss_function, x_fit_offset, x_fit_amplitude, x_fit_mean, x_fit_standard_deviation, x_center_of_mass, x_rms = x_fit
    y_fit_gauss_function, y_fit_offset, y_fit_amplitude, y_fit_mean, y_fit_standard_deviation, y_center_of_mass, y_rms = y_fit

    x_fwhm = functions.get_fwhm(x_axis, x_profile)
    y_fwhm = functions.get_fwhm(y_axis, y_profile)

    # Could be also y_profile.sum() -> it should give the same result.
    intensity = x_profile.sum()

    # Add return values
    return_value["x_axis"] = x_axis
    return_value["y_axis"] = y_axis
    return_value["image"] = image
    return_value["width"] = image.shape[1]
    return_value["height"] = image.shape[0]
    return_value["timestamp"] = timestamp
    return_value["min_value"] = min_value
    return_value["max_value"] = max_value
    return_value["x_profile"] = x_profile
    return_value["y_profile"] = y_profile
    return_value["intensity"] = intensity
    return_value["x_fwhm"] = x_fwhm
    return_value["y_fwhm"] = y_fwhm

    # Needed for config tr  aceability.
    return_value["processing_parameters"] = json.dumps(parameters)

    # TODO Provide - Center of mass of profile
    # TODO Provide - RMS of profile
    return_value["x_center_of_mass"] = x_center_of_mass
    return_value["x_rms"] = x_rms
    return_value["y_center_of_mass"] = y_center_of_mass
    return_value["y_rms"] = y_rms

    # Fitting results
    return_value["x_fit_gauss_function"] = x_fit_gauss_function
    return_value["x_fit_offset"] = x_fit_offset
    return_value["x_fit_amplitude"] = x_fit_amplitude
    return_value["x_fit_standard_deviation"] = x_fit_standard_deviation
    return_value["x_fit_mean"] = x_fit_mean

    return_value["y_fit_gauss_function"] = y_fit_gauss_function
    return_value["y_fit_offset"] = y_fit_offset
    return_value["y_fit_amplitude"] = y_fit_amplitude
    return_value["y_fit_standard_deviation"] = y_fit_standard_deviation
    return_value["y_fit_mean"] = y_fit_mean

    image_good_region = parameters.get("image_good_region")
    if image_good_region:
        try:

            def initialize_good_region_values():
                # Initialize the good region parameters.
                return_value["good_region"] = None
                return_value["gr_x_axis"] = None
                return_value["gr_y_axis"] = None
                return_value["gr_x_profile"] = None
                return_value["gr_y_profile"] = None

                return_value["gr_x_fit_gauss_function"] = None
                return_value["gr_x_fit_offset"] = None
                return_value["gr_x_fit_amplitude"] = None
                return_value["gr_x_fit_standard_deviation"] = None
                return_value["gr_x_fit_mean"] = None

                return_value["gr_y_fit_gauss_function"] = None
                return_value["gr_y_fit_offset"] = None
                return_value["gr_y_fit_amplitude"] = None
                return_value["gr_y_fit_standard_deviation"] = None
                return_value["gr_y_fit_mean"] = None
                return_value["gr_intensity"] = None

                slices = parameters.get("image_slices")
                if slices:
                    return_value["slice_orientation"] = slices["orientation"]
                    initialize_slices_values(slices["number_of_slices"])

            def initialize_slices_values(number_of_slices):
                return_value["slice_amount"] = number_of_slices
                return_value["slice_length"] = None

                for i in range(number_of_slices):
                    return_value["slice_%s_center_x" % i] = None
                    return_value["slice_%s_center_y" % i] = None
                    return_value["slice_%s_standard_deviation" % i] = None
                    return_value["slice_%s_intensity" % i] = None

                return_value["coupling"] = None
                return_value["coupling_slope"] = None
                return_value["coupling_offset"] = None

            # Good region and slices should be None if cannot be calculated.
            initialize_good_region_values()

            threshold = image_good_region["threshold"]
            gfscale = image_good_region["gfscale"]

            # Get the good region
            good_region_x_start, good_region_x_end = functions.get_good_region_profile(x_profile, threshold, gfscale)
            good_region_y_start, good_region_y_end = functions.get_good_region_profile(y_profile, threshold, gfscale)

            # Clip good region
            good_region = image[good_region_y_start:good_region_y_end, good_region_x_start:good_region_x_end]

            good_region_x_axis = x_axis[good_region_x_start:good_region_x_end]
            good_region_y_axis = y_axis[good_region_y_start:good_region_y_end]

            # Get profiles of the good region
            (good_region_x_profile, good_region_y_profile) = functions.get_x_y_profile(good_region)

            # Could be also good_region_y_profile.sum() -> it should give the same result.
            good_region_intensity = good_region_x_profile.sum()

            # Fit the profiles
            good_region_x_fit = functions.gauss_fit(good_region_x_profile, good_region_x_axis)
            good_region_y_fit = functions.gauss_fit(good_region_y_profile, good_region_y_axis)

            gr_x_fit_gauss_function, gr_x_fit_offset, gr_x_fit_amplitude, gr_x_fit_mean, gr_x_fit_standard_deviation, _, _ = good_region_x_fit
            gr_y_fit_gauss_function, gr_y_fit_offset, gr_y_fit_amplitude, gr_y_fit_mean, gr_y_fit_standard_deviation, _, _ = good_region_y_fit

            # Add return values
            return_value["good_region"] = [good_region_x_start, good_region_y_start, good_region_x_end,
                                           good_region_y_end]
            return_value["gr_x_axis"] = good_region_x_axis
            return_value["gr_y_axis"] = good_region_y_axis
            return_value["gr_x_profile"] = good_region_x_profile
            return_value["gr_y_profile"] = good_region_y_profile

            return_value["gr_x_fit_gauss_function"] = gr_x_fit_gauss_function
            return_value["gr_x_fit_offset"] = gr_x_fit_offset
            return_value["gr_x_fit_amplitude"] = gr_x_fit_amplitude
            return_value["gr_x_fit_standard_deviation"] = gr_x_fit_standard_deviation
            return_value["gr_x_fit_mean"] = gr_x_fit_mean

            return_value["gr_y_fit_gauss_function"] = gr_y_fit_gauss_function
            return_value["gr_y_fit_offset"] = gr_y_fit_offset
            return_value["gr_y_fit_amplitude"] = gr_y_fit_amplitude
            return_value["gr_y_fit_standard_deviation"] = gr_y_fit_standard_deviation
            return_value["gr_y_fit_mean"] = gr_y_fit_mean
            return_value["gr_intensity"] = good_region_intensity

            image_slices = parameters.get("image_slices")

            if image_slices:

                scale = image_slices["scale"]
                slice_number = image_slices["number_of_slices"]
                orientation = image_slices["orientation"]

                # Adjust the number of slices to be odd.
                if slice_number % 2 == 0:
                    # Add a middle slice if number of slices is even - as middle slice is half/half on center
                    _logger.info('Add additional middle slice')
                    slice_number += 1
                    initialize_slices_values(slice_number)

                try:
                    x_slice_data, x_slice_length = functions.get_x_slices_data(good_region, good_region_x_axis,
                                                                               good_region_y_axis, gr_x_fit_mean,
                                                                               gr_x_fit_standard_deviation,
                                                                               scaling=scale,
                                                                               number_of_slices=slice_number)

                    y_slice_data, y_slice_length = functions.get_y_slices_data(good_region, good_region_x_axis,
                                                                               good_region_y_axis, gr_y_fit_mean,
                                                                               gr_y_fit_standard_deviation,
                                                                               scaling=scale,
                                                                               number_of_slices=slice_number)

                    if orientation == "vertical":
                        slice_data = y_slice_data
                        slice_length = y_slice_length

                    elif orientation == "horizontal":
                        slice_data = x_slice_data
                        slice_length = x_slice_length

                    else:
                        raise ValueError("Invalid slice orientation '%s'." % orientation)

                    return_value["slice_length"] = slice_length

                    # Add return values
                    counter = 0
                    for data in slice_data:
                        return_value["slice_%s_center_x" % counter] = data[0][0]
                        return_value["slice_%s_center_y" % counter] = data[0][1]
                        return_value["slice_%s_standard_deviation" % counter] = data[1]
                        return_value["slice_%s_intensity" % counter] = data[2]
                        counter += 1

                    # Calculate x/y coupling, always out of horizontal (x axis) slices.
                    x = []
                    y = []
                    for data in x_slice_data:
                        x.append(float(data[0][0]))  # x
                        y.append(float(data[0][1]))  # y

                    slope, offset = functions.linear_fit(x, y)
                    return_value["coupling"] = slope * (gr_x_fit_standard_deviation ** 2)
                    return_value["coupling_slope"] = slope
                    return_value["coupling_offset"] = offset

                except:  # Except for slices
                    _logger.debug('Unable to apply slices')

        except:  # Except for good region
            _logger.debug('Unable to detect good region')

    return return_value
