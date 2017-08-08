from datetime import datetime
from logging import getLogger

import numpy
from bsread import source, SUB

_logger = getLogger(__name__)


def get_host_port_from_stream_address(stream_address):
    source_host, source_port = stream_address.rsplit(":", maxsplit=1)
    source_host = source_host.split("//")[1]

    return source_host, int(source_port)


def update_pipeline_config(current_config, config_updates):
    def update_subsection(section_name):
        if config_updates.get(section_name) is not None:
            old_section = current_config.get(section_name)

            if old_section:
                old_section.update(config_updates.get(section_name))
                config_updates[section_name] = old_section

    update_subsection("camera_calibration")
    update_subsection("image_good_region")
    update_subsection("image_slices")

    current_config.update(config_updates)

    return current_config


def collect_background(camera_name, stream_address, n_images, background_manager):

    try:

        host, port = get_host_port_from_stream_address(stream_address)
        accumulator_image = None

        with source(host=host, port=port, mode=SUB) as stream:
            for _ in range(n_images):

                data = stream.receive()
                image = data.data.data["image"].value

                if accumulator_image is None:
                    accumulator_image = numpy.array(image)
                else:
                    accumulator_image += image

        background_id = camera_name + datetime.now().strftime("_%Y%m%d_%H%M%S_%f")
        background_image = accumulator_image / n_images

        background_manager.save_background(background_id, background_image)

        return background_id

    except:
        _logger.exception("Error while collecting background.")
        raise

