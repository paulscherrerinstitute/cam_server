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
MFLOW_NO_CLIENTS_TIMEOUT = 10

###################
# Camera settings #
###################

# Each camera config gets assigned one port.
CAMERA_STREAM_PORT_RANGE = (10100, 10201)
# Default folder for camera configs.
DEFAULT_CAMERA_CONFIG_FOLDER = "configuration/camera"
# Default colormap to use when getting an image from the camera.
DEFAULT_CAMERA_IMAGE_COLORMAP = "rainbow"

# We have only 2 channels: Image and timestamp. Header compression is not really needed.
CAMERA_BSREAD_DATA_HEADER_COMPRESSION = "none"
# Compression here might be a good idea. Use "bitshuffle_lz4" or None.
CAMERA_BSREAD_IMAGE_COMPRESSION = "none"

#####################
# Pipeline settings #
#####################

# Every time you open a pipeline it gets the next port.
PIPELINE_STREAM_PORT_RANGE = (11100, 11201)
# Default folder for camera configs.
DEFAULT_PIPELINE_CONFIG_FOLDER = "configuration/pipeline"
# Where to store the backgrounds by default.
DEFAULT_BACKGROUND_CONFIG_FOLDER = "configuration/background"
# Maximum time to wait before aborting the receive.
PIPELINE_RECEIVE_TIMEOUT = 1000

################
# IPC settings #
################

# Time to wait for the process to execute the requested action.
PROCESS_COMMUNICATION_TIMEOUT = 6
# Interval used when polling the state from the process.
PROCESS_POLL_INTERVAL = 0.1
