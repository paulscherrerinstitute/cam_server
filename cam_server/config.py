########################
# Interface parameters #
########################

from cam_server_client.config import *

# Default logging level.
DEFAULT_LOGGING_LEVEL = "INFO"
# How many seconds do we wait before disconnecting a stream without clients.
MFLOW_NO_CLIENTS_TIMEOUT = 10
# Default web server used by Bottle
DEFAULT_WEB_SERVER = "wsgiref"

###################
# Camera settings #
###################

# Each camera config gets assigned one port.
CAMERA_STREAM_PORT_RANGE = (10100, 10301)
# Default folder for camera configs.
DEFAULT_CAMERA_CONFIG_FOLDER = "configuration/camera_config"
# Default colormap to use when getting an image from the camera.
DEFAULT_CAMERA_IMAGE_COLORMAP = "rainbow"

# We have only 2 channels: Image and timestamp. Header compression is not really needed.
CAMERA_BSREAD_DATA_HEADER_COMPRESSION = None
# Compression here might be a good idea. Use "bitshuffle_lz4" or None.
CAMERA_BSREAD_IMAGE_COMPRESSION = None
# Compression for scalar attributes.
CAMERA_BSREAD_SCALAR_COMPRESSION = None
# Default interval for simulation camera.
DEFAULT_CAMERA_SIMULATION_INTERVAL = 0.1

# Number of image format error before rising exception
FORMAT_ERROR_COUNT = 10

#Format for creating shared memory files for IPC protocol.
IPC_FEEDS_FOLDER = "/tmp"

#####################
# Pipeline settings #
#####################

# Every time you open a pipeline it gets the next port.
PIPELINE_STREAM_PORT_RANGE = (11100, 11201)
# Default folder for camera configs.
DEFAULT_PIPELINE_CONFIG_FOLDER = "configuration/pipeline_config"
# Where to store the backgrounds by default.
DEFAULT_BACKGROUND_CONFIG_FOLDER = "configuration/background_config"
# Where to store the user pipelene scripts.
DEFAULT_USER_SCRIPT_FOLDER = "configuration/user_scripts"
# How many days background files other than latest are kept (default -1 means forever)
DEFAULT_BACKGROUND_FILES_DAYS_TO_LIVE = -1
# Temporary storage.
DEFAULT_TEMP_FOLDER = "temp"
# Maximum time to wait before aborting the receive.
PIPELINE_RECEIVE_TIMEOUT = 1000
# Default number of images to collect when acquiring the background.
PIPELINE_DEFAULT_N_IMAGES_FOR_BACKGROUND = 10

#Default mFlow=ZMQ parameters
PIPELINE_DEFAULT_QUEUE_SIZE = 10
PIPELINE_DEFAULT_MODE = "PUB"
PIPELINE_DEFAULT_BLOCK =True

#Default buffer sizes for BS data gathering
BSREAD_IMAGE_BUFFER_SIZE_DEFAULT = 1000
BSREAD_DATA_BUFFER_SIZE_DEFAULT = 1000
BSREAD_IMAGE_BUFFER_SIZE_MAX = 5000
BSREAD_DATA_BUFFER_SIZE_MAX = 10000

BSREAD_FORMAT_ERROR_TIMEOUT = 10 #s

#Pipeline types
PIPELINE_TYPE_PROCESSING = "processing"
PIPELINE_TYPE_STORE = "store"
PIPELINE_TYPE_STREAM = "stream"


ABORT_ON_ERROR = True

################
# IPC settings #
################

# Time to wait for the process to execute the requested action.
PROCESS_COMMUNICATION_TIMEOUT = 10
# Interval used when polling the state from the process.
PROCESS_POLL_INTERVAL = 0.1

####################
# General settings #
####################

TIME_FORMAT = "%Y-%m-%d %H:%M:%S UTC%z"

##################
# EPICS settings #
##################

EPICS_TIMEOUT = None

EPICS_PV_SUFFIX_STATUS = ":INIT"
EPICS_PV_SUFFIX_WIDTH = ":WIDTH"
EPICS_PV_SUFFIX_HEIGHT = ":HEIGHT"
EPICS_PV_SUFFIX_IMAGE = ":FPICTURE"
EPICS_PV_SUFFIX_STREAM_ADDRESS = ":BSREADCONFIG"

################
# ZMQ settings #
################

ZMQ_RECEIVE_TIMEOUT = 1000


#########################
# Manager rest settings #
#########################

DEFAULT_SERVER_CLIENT_TIMEOUT = None
DEFAULT_SERVER_INFO_TIMEOUT = 2000
