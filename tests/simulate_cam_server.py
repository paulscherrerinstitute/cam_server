import time
from mflow import PULL, PUSH, PUB
from bsread.sender import sender
from cam_server.camera.configuration import CameraConfig
from cam_server.camera.source.simulation import CameraSimulation
from cam_server.pipeline.configuration import PipelineConfig
from cam_server.pipeline.data_processing.processor import process_image

# Size of simulated image.
image_size_x = 1280
image_size_y = 960

# Select compression options. Only this 2 are available in Python - cam_server uses "bitshuffle_lz4".
compression = "bitshuffle_lz4"
# compression = "none"

# Stream configuration. cam_server uses PUB for the output stream.
output_stream_port = 9999
output_stream_mode = PUB
# output_stream_mode = PUSH

simulated_camera = CameraSimulation(camera_config=CameraConfig("simulation"),
                                    size_x=image_size_x, size_y=image_size_y)
x_axis, y_axis = simulated_camera.get_x_y_axis()
x_size, y_size = simulated_camera.get_geometry()

# Documentation: https://github.com/datastreaming/cam_server#pipeline_configuration
pipeline_parameters = {
    "camera_name": "simulation"
}

pipeline_config = PipelineConfig("test_pipeline", pipeline_parameters)
parameters = pipeline_config.get_configuration()

image_number = 0

with sender(port=output_stream_port, mode=output_stream_mode) as output_stream:
    # Get simulated image.
    image = simulated_camera.get_image()

    # Generate timestamp.
    timestamp = time.time()

    # Pass data to processing pipeline.
    processed_data = process_image(image, timestamp, x_axis, y_axis, pipeline_parameters)

    # Set height and width.
    processed_data["width"] = processed_data["image"].shape[1]
    processed_data["height"] = processed_data["image"].shape[0]

    print("Sending image number: ", image_number)
    image_number += 1

    output_stream.send(data=processed_data, timestamp=timestamp)
