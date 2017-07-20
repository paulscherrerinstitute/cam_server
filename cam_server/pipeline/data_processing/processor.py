from logging import getLogger

from cam_server.pipeline.data_processing import functions

_logger = getLogger(__name__)


def process_image(image, timestamp, x_axis, y_axis, parameter):

    # Add return values
    return_value = dict()

    image = image
    if parameter.subtract_background:
        image = functions.subtract_background(image, parameter.background_image)

    if parameter.apply_threshold:
        image = functions.apply_threshold(image, parameter.threshold)

    if parameter.apply_region_of_interest:
        offset_x, size_x, offset_y, size_y = parameter.region_of_interest
        image = functions.get_region_of_interest(image, offset_x, size_x, offset_y, size_y)

        # TODO To be optimized
        # Apply roi to geometry x_axis and y_axis
        x_axis = x_axis[offset_x:offset_x + size_x]
        y_axis = y_axis[offset_y:offset_y + size_y]

    (min_value, max_value) = functions.get_min_max(image)

    (x_profile, y_profile) = functions.get_x_y_profile(image)

    x_fit = functions.gauss_fit(x_profile, x_axis)
    y_fit = functions.gauss_fit(y_profile, y_axis)
    x_fit_gauss_function, x_fit_offset, x_fit_amplitude, x_fit_mean, x_fit_standard_deviation, x_center_of_mass, x_rms = x_fit
    y_fit_gauss_function, y_fit_offset, y_fit_amplitude, y_fit_mean, y_fit_standard_deviation, y_center_of_mass, y_rms = y_fit

    # Add return values
    return_value["x_axis"] = x_axis
    return_value["y_axis"] = y_axis
    return_value["image"] = image
    return_value["timestamp"] = timestamp
    return_value["min_value"] = min_value
    return_value["max_value"] = max_value
    return_value["x_profile"] = x_profile
    return_value["y_profile"] = y_profile

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

    if parameter.apply_good_region:
        try:
            # Get the good region
            good_region_x_start, good_region_x_end = functions.get_good_region_profile(x_profile,
                                                                                       parameter.good_region_threshold,
                                                                                       parameter.good_region_gfscale)
            good_region_y_start, good_region_y_end = functions.get_good_region_profile(y_profile,
                                                                                       parameter.good_region_threshold,
                                                                                       parameter.good_region_gfscale)

            # Clip good region
            good_region = image[good_region_y_start:good_region_y_end, good_region_x_start:good_region_x_end]

            good_region_x_axis = x_axis[good_region_x_start:good_region_x_end]
            good_region_y_axis = y_axis[good_region_y_start:good_region_y_end]

            # Get profiles of the good region
            (good_region_x_profile, good_region_y_profile) = functions.get_x_y_profile(good_region)

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

            if parameter.apply_slices:

                try:
                    x_slice_data = functions.get_x_slices_data(good_region, good_region_x_axis, good_region_y_axis, gr_x_fit_mean,
                                                               gr_x_fit_standard_deviation,
                                                               scaling=parameter.slices_scale,
                                                               number_of_slices=parameter.number_of_slices)

                    y_slice_data = functions.get_y_slices_data(good_region, good_region_x_axis, good_region_y_axis,
                                                               gr_y_fit_mean,
                                                               gr_y_fit_standard_deviation,
                                                               scaling=parameter.slices_scale,
                                                               number_of_slices=parameter.number_of_slices)

                    # Add return values
                    counter = 0
                    for data in y_slice_data:
                        return_value["slice_%s_center_x" % counter] = data[0][0]
                        return_value["slice_%s_center_y" % counter] = data[0][1]
                        return_value["slice_%s_standard_deviation" % counter] = data[1]
                        return_value["slice_%s_intensity" % counter] = data[2]
                        counter += 1

                    # Calculate x/y coupling
                    x = []
                    y = []
                    for data in x_slice_data:
                        x.append(float(data[0][0]))  # x
                        y.append(float(data[0][1]))  # y

                    slope, offset = functions.linear_fit(x, y)
                    print(slope)

                    # TODO need to calculate: slope * sigma(x)^2
                    return_value["coupling"] = slope * (gr_x_fit_standard_deviation ** 2)

                except:  # Except for slices
                    _logger.exception('Unable to apply slices')

        except:  # Except for good region
            _logger.exception('Unable to detect good region')

    return return_value