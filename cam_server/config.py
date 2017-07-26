########################
# Interface parameters #
########################

# API prefix.
API_PREFIX = "/api/v1"
# Camera server prefix.
CAMERA_REST_INTERFACE_PREFIX = "/cam"
# Pipeline server prefix
PIPELINE_REST_INTERFACE_PREFIX = "/pipeline"
# Default logging level.
DEFAULT_LOGGING_LEVEL = "WARNING"
# How many seconds do we wait before disconnecting a stream without clients.
MFLOW_NO_CLIENTS_TIMEOUT = 3

###################
# Camera settings #
###################

# Each camera config gets assigned one port. 1000 cameras per server should be enough.
CAMERA_STREAM_PORT_RANGE = (10100, 11100)
# Default folder for cam_server configs.
DEFAULT_CAMERA_CONFIG_FOLDER = "configuration"
# Default colormap to use when getting an image from the camera.
DEFAULT_CAMERA_IMAGE_COLORMAP = "rainbow"

# We have only 2 channels: Image and timestamp. Header compression is not really needed.
CAMERA_BSREAD_DATA_HEADER_COMPRESSION = None
# Compression here might be a good idea. Use "bitshuffle_lz4" or None.
CAMERA_BSREAD_IMAGE_COMPRESSION = None

# Wait for a maximum of 500ms before exiting the send.
CAMERA_SEND_TIMEOUT = 500

#####################
# Pipeline settings #
#####################

# Every time you open a pipeline it gets the next port. 1000 rotating ports should be enough.
PIPELINE_STREAM_PORT_RANGE = (11100, 12100)
# Receive timeout for the pipeline.
PIPELINE_RECEIVE_TIMEOUT = 500

################
# IPC settings #
################

# Time to wait for the process to execute the requested action.
PROCESS_COMMUNICATION_TIMEOUT = 6
# Interval used when polling the state from the process.
PROCESS_POLL_INTERVAL = 0.1
