from datetime import datetime

import numpy
from bsread import source, SUB


def get_host_port_from_stream_address(stream_address):
    source_host, source_port = stream_address.rsplit(":", maxsplit=1)
    source_host = source_host.split("//")[1]

    return source_host, int(source_port)


def collect_background(camera_name, stream_address, n_images, background_manager):

    host, port = get_host_port_from_stream_address(stream_address)
    accumulator_image = None

    with source(host=host, port=port, mode=SUB) as stream:
        for _ in range(n_images):

            data = stream.receive()
            print(data)
            image = data.data.data["image"].value

            if accumulator_image is None:
                accumulator_image = numpy.array(image)
            else:
                accumulator_image += image

    background_id = camera_name + datetime.now().strftime("_%Y%m%d_%H%M%S_%f")
    background_image = accumulator_image / n_images

    background_manager.save_background(background_id, background_image)

    return background_id

