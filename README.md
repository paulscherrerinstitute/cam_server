[![Build Status](https://travis-ci.org/paulscherrerinstitute/cam_server.svg?branch=master)](https://travis-ci.org/datastreaming/cam_server) [![Build status](https://ci.appveyor.com/api/projects/status/0vyk18qxnqk2cmvx?svg=true)](https://ci.appveyor.com/project/Babicaa/cam-server)

# Camera and Pipeline server
Cam server is an epics - bsread interface that converts epics enabled camera into a bs_read stream. In addition it
also provides a processing pipeline and a REST interface to control both the cameras and the pipeline.

**WARNING**: Please note that for normal users, only **PipelineClient** should be used. CamClient is used by the
underlying infrastructure to provide camera images to the pipeline server.

# Table of content
1. [Quick start (Get stream from screen panel)](#quick_start)
2. [Build](#build)
    1. [Conda setup](#conda_setup)
    2. [Local build](#local_build)
    3. [Docker build](#docker_build)
3. [Basic concepts](#basic_concepts)
    1. [Requesting a stream and instance management](#reqeust_a_stream)
    2. [Shared and private pipeline instances](#shared_and_private)
    3. [Configuration versioning and camera background in the pipeline server](#configuration_versioning)
4. [Configuration](#configuration)
    1. [Camera configuration](#camera_configuration)
    2. [Pipeline configuration](#pipeline_configuration)
5. [Web interface](#web_interface)
    1. [Python client](#python_client)
    2. [REST API](#rest_api)
6. [Running the servers](#running_the_servers)
    1. [Camera_server](#run_camera_server)
    2. [Pipeline server](#run_pipeline_server)
    3. [Docker Container](#run_docker_container)
7. [Production configuration](#production_configuration)
8. [Examples](#examples)
    1. [Get stream from screen panel](#quick_start)
    2. [Get the simulation camera stream](#get_simulation_camera_stream)
    3. [Get a basic pipeline with a simulated camera](#basic_pipeline)
    4. [Create a pipeline instance with background](#private_pipeline)
    5. [Read the stream for a given camera name](#read_camera_stream)
    6. [Modifying camera config](#modify_camera_config)
    7. [Modifying pipeline config](#modify_pipeline_config)
    8. [Create a new camera](#create_camera_config)
    9. [Get single message from screen_panel stream](#single_message_screen_panel)
    10. [Save camera stream to H5 file](#stream_to_h5_file)
9. [Deploy in production](#deploy_in_production)


<a id="quick_start"></a>
## Quick start (Get stream from screen panel)
The example below shows how to access the camera stream you have currently open in a screen_panel instance.
This should be one of the most common use cases.

Open screen_panel to the camera you want to stream. Configure the stream in the desired way using screen_panel.
Do not close screen_panel - let it run. When you are satisfied with the image (and calculated fields), run the script:

```python
from cam_server import PipelineClient
from cam_server.utils import get_host_port_from_stream_address
from bsread import source, SUB

# Create a pipeline client.
client = PipelineClient()

# Define the camera name you want to read. This should be the same camera you are streaming in screen panel.
camera_name = "simulation"

# Format of the instance id when screen_panel creates a pipeline.
pipeline_instance_id = camera_name + "_sp1"

# Get the stream for the pipelie instance.
stream_address = client.get_instance_stream(pipeline_instance_id)

# Extract the stream host and port from the stream_address.
stream_host, stream_port = get_host_port_from_stream_address(stream_address)

# Open connection to the stream. When exiting the 'with' section, the source disconnects by itself.
with source(host=stream_host, port=stream_port, mode=SUB) as input_stream:
    input_stream.connect()

    # Read one message.
    message = input_stream.receive()

    # Print out the received stream data - dictionary.
    print("Dictionary with data:\n", message.data.data)

    # Print out the X center of mass.
    print("X center of mass: ", message.data.data["x_center_of_mass"].value)
```

Whenever screen_panel starts start streaming a camera, it creates a pipeline named
"**[camera\_name]**_sp1". You can connect to this instance with the **get\_instance\_stream** function of the client.

All the changes to the stream (config changes) done in the screen_panel will be reflected in the stream after a couple
of images.

<a id="build"></a>
## Build

<a id="conda_setup"></a>
### Conda setup
If you use conda, you can create an environment with the cam_server library by running:

```bash
conda create -c paulscherrerinstitute --name <env_name> cam_server
```

After that you can just source you newly created environment and start using the server.

<a id="local_build"></a>
### Local build
You can build the library by running the setup script in the root folder of the project:

```bash
python setup.py install
```

or by using the conda also from the root folder of the project:

```bash
conda build conda-recipe
conda install --use-local mflow_node_processors
```

#### Requirements
The library relies on the following packages:

- requests
- bsread >=0.9.9
- bottle
- numpy
- scipy
- pyepics
- matplotlib
- pillow
- psutil
- h5py
- numba

In case you are using conda to install the packages, you might need to add the **paulscherrerinstitute** channel to
your conda config:

```
conda config --add channels paulscherrerinstitute
```

<a id="docker_build"></a>
### Docker build
**Warning**: When you build the docker image with **build.sh**, your built will be pushed to the PSI repo as the
latest cam_server version. Please use the **build.sh** script only if you are sure that this is what you want.

To build the docker image, run the build from the **docker/** folder:
```bash
./build.sh
```

Before building the docker image, make sure the latest version of the library is available in Anaconda.

**Please note**: There is no need to build the image if you just want to run the docker container.
Please see the [Docker Container](#run_docker_container) chapter.


<a id="basic_concepts"></a>
## Basic concepts

<a id="request_a_stream"></a>
### Requesting a stream and instance management

Instance management is done automatically. Once you request a stream, you have a fixed amount of time
(no clients timeout) to connect to the stream. If you do not connect in the given time, the stream will automatically
be stopped. In this case, you will need to request the stream again.

The same is true when disconnecting from the stream. You do not need to take any actions to close the pipeline or
camera stream, as they will close themselves automatically after the configured time has elapsed.

There are however 2 methods available for manual instance management:
- stop_instance
- stop_all_instances

They are not meant to be used during normal operations, but only as administrative methods if something does not work
as expected.

<a id="shared_and_private"></a>
### Shared and private pipeline instances

The difference between shared and private pipeline instances in mainly in the fact that shared instances are
read only (the pipeline config cannot be changed). This is done to prevent the interaction between different users
- if 2 people are independently viewing the same pipeline at the same time, we need to prevent to any of them to change
what the other receives. If you need to change the pipeline parameters, you need to create a private pipeline instance.

Shared instances are named after the pipeline config they are created from. For example, if you have a saved pipeline
with the name 'simulation_pipeline', the shared instance instance_id will be 'simulation_pipeline'. Private instances
can have a custom instance_id you provide, or an automatically generated one.
In any case, the given instance_id must be unique.

You can share the private pipeline as well - all you need to share is the instance_id. But you need to be
aware that anyone having your instance_id can change any parameter on the pipeline.

<a id="configuration_versioning"></a>
### Configuration versioning and camera background in the pipeline server
We have a requirement to be always able to access the original configuration with which the image was processed.

The configuration used by the pipeline for image processing is explicitly added to each bs_read message. The
configuration is added as a JSON object (string representation) in the **processing\_parameters** field.
Since this config is included with every message, there is no need to version the
processing parameters on the pipeline server. You can retrieve the processing parameters for each frame individually
(the processing parameters can change from frame to frame).

Backgrounds, on the other hand, are not included into this field - just the background id is (since the background is
a large image, it does not make sense to include it in every message). As a consequence, backgrounds can never be
deleted and need to be versioned on the pipeline server. All backgrounds need to be backed up regularly.

<a id="configuration"></a>
## Configuration
The camera and pipeline instances have their own configuration which can be read and set via the rest interface.
We currently store the configurations in files, but REDIS is planned to be used in the near future.

The expected folder structure for the configuration (which can be changed by passing parameters to the executables):

- configuration/camera_config : Folder where JSON files with camera configurations are located.
- configuration/pipeline_config : Folder where JSON files with pipeline configurations are located.
- configuration/background_config : Folder where NPY files with camera backgrounds are located.

<a id="camera_configuration"></a>
### Camera configuration
For camera configuration, all fields must be specified, and there is no defaulting in case some are missing.

#### Configuration parameters

- **name**: Name of the camera.
- **source**: Source of the camera (PV prefix, bsread stream)
- **source_type**: Type of the source (available: epics, bsread, simulation)
- **mirror\_x**: Mirror camera image over X axis.
- **mirror\_y**: Mirror camera image over Y axis.
- **rotate**: how many times to rotate the camera image by 90 degrees.
- **camera\_calibration** (Default _None_): Info on how to convert the camera pixels into engineering units.
    - reference_marker (Default _[0, 0, 100, 100]_): Reference markers placement.
    - reference_marker_width (Default _100.0_): Width of reference markers.
    - reference_marker_height (Default _100.0_): Height of reference markers.
    - angle_horizontal (Default _0.0_): Horizontal angle.
    - angle_vertical (Default _0.0_): Vertical angle.
- **protocol**: (Default _tcp_): ZMQ transport protocol: _tcp_ or _ipc_. 
- **alias**: List of aliases for this camera (alternative ways to refer to the camera, must be unique).
- **group**: List of camera groups this camera belongs to (so cameras can be listed by group).

##### Source type
cam_server can connect to different type of sources. The type of source you select defines the meaning of the
**source** field in the configuration:

- source_type = "epics" : Connect to an Epics camera. The 'source' field is the camera prefix.
- source_type = "bsread" : Connect to a bsread stream. The 'source' field is the stream address.
- source_type = "simulation": Generate simulated images. The 'source' can be anything, but it must NOT be None.


##### Configuration parameters for source\_type = _'bsread'_  
- **connections** (Default _1_): Number of ZMQ connections to the camera. More connections can increase the throughput.
- **buffer_size** (Default _connections * 5_): If greater than 0, then receivers and sender are threaded, and this value 
  defines the size of the message buffer. Irrelevant if connections < 2/


#### Example
```json
{
  "name": "example_4",
  "source": "EPICS_example_4",
  "source_type": "epics",
  "mirror_x": true,
  "mirror_y": false,
  "rotate": 4,

  "camera_calibration": {
    "reference_marker": [ 0, 0, 100, 100 ],
    "reference_marker_width": 100.0,
    "reference_marker_height": 100.0,
    "angle_horizontal": 0.0,
    "angle_vertical": 0.0
  }
}
```

<a id="pipeline_configuration"></a>
### Pipeline configuration

Configuration changes can for the pipeline can be incremental - you need to specify only the fields that you want
to change. A valid configuration must have only the **camera_name** specified, all other fields will be
defaulted to **None** (or False, in the case of "image_background_enable"). The complete configuration used for the
pipeline is added to the output bsread stream in the
**processing\_parameters** field.

#### Configuration parameters

- **pipeline\_type** (Default _'processing_):
    - _'processing'_, _'store'_, _'stream'_, _'custom'_, _'script'_, _'fanout'_ or _'fanin'_
- **camera\_name** : Name of the camera to use as a pipeline source.
- **function** (Default _None_):
    - Redefine processing script (function name or file name implementing processing function).     
    - If pipeline_type = 'processing', set function = _'transparent'_ for no calculations.
    - If pipeline\_type = _'processing'_, the processing script must implement the function:
        - def process_image(image, pulse_id, timestamp, x_axis, y_axis, parameters, bsdata):
    - If pipeline\_type = _'stream'_, the processing script must implement the function:
        - def process(stream_data, pulse_id, timestamp, parameters):
    - In both cases the processing function should return a OrderedDict with the values to stream out.        
- **reload** (Default _False_):
    - If True reloads the processing function. For performance reasons the function is not reloaded by default. 
      Should be set to True only for testing new functions. If is automatically reset after the function is reloaded.
- **include** (Default _None_):
    - If defined, provides list of names of fields requested in the pipeline output stream (any other not listed is removed). 
- **exclude** (Default _None_):
    - If defined, provides list of names of fields to be removed from the pipeline output stream. 
- **camera_timeout** (Default _10.0_):
    - If no message received in camera_timeout seconds, pipeline attempts to reconnect to the camera
      stream. If reconnection is not possible, the pipeline will stop. 
      If null or non positive then the timeout handling is disabled.
- **create_header** (Default _'on_change'_): Message header creation strategy ('once', 'always' or 'on_change').
    - **once**: Header is created only in the first message. Assume types and shapes never change.
      This has the best  performance but the  pipeline function must make sure channel types don't change, otherwise the pipeline 
      will break.
    - **always**: Creates header for each message. It is the most flexible, as the message is always consistent with 
      the header. Suited for pipelines with changing channels. There is a cost of regenerating the header each time.      
    - **on_change**: A compromise of the options above. Inspect channels types and shapes and recreate the header only if there is any change. 
- **mode** (Default _PUB_, but  _PUSH_ for __fanout__ pipelines): Output type ('PUB', 'PUSH' or 'FILE').
    - For stream modes (PUB or PUSH), the following parameters are valid:
        - **queue_size** (Default _10_): stream High Water Mark.
        - **block** (Default _True_ for PUSH, _False_ for PUB): define if stream sending is blocking.   
        - **no_client_timeout** (Default _10_): Timeout to close the pipeline if no client is connected.
          A not positive number disable this monitoring 
          (the pipeline is kept running even if there is no connected client). 
        - **buffer_size** (Default _None_): If defined, sets the size of a message buffer. 
          In this case the messages are not sent immediately but buffered and processed in a different thread.
          Used to receive also messages generated before the stream was started, together with 
          _"mode":"PUSH"_ and _"queue_size":1_.  
    - For FILE mode, the following parameters are valid:
        - **file**: File name.
        - **layout** (Default _'DEFAULT'_): Output file layout ('DEFAULT' or 'FLAT').
        - **localtime** (Default _True_): Create datasets to store each channel local time (in addition to global timestamp).
        - **change** (Default _False_): If True then supports change in arrays dimensions (creating new datasets in the same file).
- **pid_range** (Default _None_): Pulse ID range to be processed [start_pid, stop_pid]. 
  The list can be updated dynamically. While start_pid=0 writing is disabled. 
  If stop_pid is reached then the instance shuts itself down, as no other pulse id wil be processed.  
- **records** (Default _None_): Number of messages to be sent. 
  The list cannot be updated dynamically in FILE mode: datasets are created with fix size. 
  If the number of records is reached then the instance shuts itself down.  
- **paused** (Default _False_): While set to _True_ the messages processing is stopped (received messages are dumped).
- **input\_pipeline** : Name of a pipeline to be used as a pipeline source (instead a camera instance).
        Creates the pipeline and sets the "input_stream" parameter.
- **input\_stream** : Name of a bsread stream to be used as a pipeline source (instead a camera instance).
- **input\_mode** (Default _SUB_, but  _PULL_ for __fanin__ pipelines): bsread stream mode (_SUB_ or _PULL_).
- **output\_pipeline** : Name of a pipeline to be used as a pipeline destination (in worker pipelines sending data to fan-in pipelines).
        Creates the pipeline and sets the "output_stream" parameter.
- **output\_stream** : Address to be used as a pipeline destination (in worker pipelines sending data to fan-in pipelines).
- **downsampling** (Default _None_):
    - If defined the incoming stream is downsampled by the given factor.
- **processing_threads** (Default _None_): Number of  processing threads. If greater than 0 then the processing is parallelized.
- **abort_on_error** (Default _True_): If true (default) the pipeline stops upon errors during processing.
- **abort_on_timeout** (Default _False_): If true the pipeline stops in source timeout.
- **stream_timeout** (Default _10_): Timeout of the source stream, defined in seconds.
- **enforce_pid** (Default _False_): If true pulse id is monotonic (excluding messages having smaller pulse id  than the last one send).

  
##### Configuration parameters for pipeline\_type = _'processing'_  
- **image\_background** (Default _None_): Background to subtract from the original image.
- **image\_background_enable** (Default _False_): Enable or disable the image_background subtraction.
   If set to _"passive"_ then the background is fetched and, instead of being applied to the image, is sent to the processing
   function, within the parameters, with the key name "background_data". 
- **image\_threshold** (Default _None_): Minimum value of each pixel. Pixels below the threshold are converted to 0.
- **image\_region\_of\_interest** (Default _None_): Crop the image before processing.
- **image\_good\_region** (Default _None_): Good region to use for fits and slices.
    - threshold (Default _0.3_): Threshold to apply on each pixel.
    - gfscale (Default _1.8_): Scale to extend the good region.
- **image\_slices** (Default _None_):
    - number_of_slices (Default _1_): Desired number of slices.
    - scale (Default _2.0_): Good region scale in for slicing purposes.
    - orientation (Default _vertical_): Orientation of the slices. Can be 'vertical' or 'horizontal'.
- **rotation** (Default _None_):
    - Rotation to be applied to image prior to running the pipeline:
        - angle (Default 0.0): Rotation angle in degrees.
        - order (Default 1): Order of the interpolation (1 to 5)
        - mode (Default '0.0'): Rotation mode. By default preserves image shape and fills empty pixels with zeros.
            - Rotation modes that  preserve image shape and axis scaling (the mode indicates how empty pixels are filled):
              _reflect_, _nearest_, _mirror_, _wrap_, or a constant.                            
            - _ortho_: orthogonal mode, preserve image pixels and adapt axis scaling. The angle must be a multiple of 90:                            
- **averaging** (Default _None_):
    - If defined specifies the size of the image buffer to be averaged.
        If number is positive, generates only one output when the buffer is full and clears it  (frame rate is reduced).
        If number is negative, generates outputs continuously, averaging the last images  (frame rate is sustained).
- **max_frame_rate** (Default _None_):
    - If defined determines the maximum desired frame rate generated by the pipeline.
- **bsread_address** (Default _None_): Source of bsread data to be merged with camera data. 
- **bsread_channels** (Default _None_): Channel names of bsread to be merged with camera data. 
  If defined and bsread_address is not, then reads from the dispatcher.
- **bsread_mode** (Default _None_): "PULL"(default if bsread_address is defined ) or "SUB" (default if bsread_address is not defined)
- **bsread_image_buf** (Default _1000_): Size of image buffer to merge with bsread data.
- **bsread_data_buf** (Default _1000_): Size of data buffer to merge with image data. 
- **copy** (Default _False_): If true the received image is copied before the pipeline is processed. The received image is read-only. 
        Operations in-place in the default pipeline create a copy of the image if needed (thresholding and background subtraction).
        If custom pipelines change the received image in-place they must make a copy of it before - or set this parameter to true. 
        
##### Configuration parameters for pipeline\_type = _'stream'_    
- **bsread_address** (Default _None_): Source of bsread data. 
- **bsread_channels** (Default _None_): Channel names of bsread data. 
  Must be defined if bsread_address is not - in this case reading from the dispatcher.
- **bsread_mode** (Default _None_): "PULL"(default if bsread_address is defined ) or "SUB" (default if bsread_address is not defined )

##### Configuration parameters for pipeline\_type = _'fanin'_    
- **pid_buffer** (Default _0_): Buffer size for reordering pulse ids received from different worker pipelines.

    
#### Example
```json
{
  "camera_name": "simulation",

  "image_background": null,
  "image_background_enable": false,
  "image_threshold": 0.5,
  "image_region_of_interest": [0, 100, 0, 100],

  "image_good_region": {
    "threshold": 0.9,
    "gfscale": 3
  },

  "image_slices": {
    "number_of_slices": 1,
    "scale": 1.0,
    "orientation": "vertical"
  }
}
```

<a id="web_interface"></a>
## Web interface
The cam_server is divided into 2 parts:

- Camera server (default: localhost:8888)
- Pipeline server (default: localhost:8889)

Operations on both server are accessible via the REST api and also via a python client class (that calls the REST api).

As an end user you should interact mostly with the pipeline, with the exception of cases where you need the raw
image coming directly from the camera - even in this case, it is advisable to create a new pipeline without processing
and use the stream from the pipeline. In this way, we lower the network load on the system (only one instance of
camera stream, that is shared by many pipelines).

All request (with the exception of **get\_camera\_image**) return a JSON with the following fields:
- **state** - \["ok", "error"\]
- **status** - What happened on the server or error message, depending on the state.
- Optional request specific field - \["cameras", "geometry", "info", "stream", "config", "pipelines",
"instance_id", "background_id"\]

For more information on what each command does check the **API** section in this document.

<a id="python_client"></a>
### Python client

There are 2 classes available to communicate with the camera and pipeline server. They are basically just wrappers
around REST API calls (see next chapters).

#### CamClient

**WARNING**: Please note that for normal users, only **PipelineClient** should be used. CamClient is used by the
underlying infrastructure to provide camera images to the pipeline server.

Import and create a cam client instance:
```python
from cam_server import CamClient
client = CamClient()
```

Class definition:
```
class CamClient()

  __init__(self, address='http://sf-daqsync-01:8888/')
    :param address: Address of the cam API, e.g. http://localhost:10000

  delete_camera_config(self, camera_name)
      Delete config of camera.
      :param camera_name: Camera to set the config to.
      :return: Actual applied config.

  get_address(self)
      Return the REST api endpoint address.

  get_camera_config(self, camera_name)
      Return the cam configuration.
      :param camera_name: Name of the cam.
      :return: Camera configuration.

  get_camera_geometry(self, camera_name)
      Get cam geometry.
      :param camera_name: Name of the cam.
      :return: Camera geometry.

  get_camera_image(self, camera_name)
      Return the cam image in PNG format.
      :param camera_name: Camera name.
      :return: server_response content (PNG).

  def get_camera_image_bytes(self, camera_name):
      Return the cam image bytes.
      :param camera_name: Camera name.
      :return: JSON with bytes and metadata.

  get_instance_stream(self, camera_name)
      Get the camera stream address.
      :param camera_name: Name of the camera to get the address for.
      :return: Stream address.

  get_cameras(self)
      List existing cameras.
      :return: Currently existing cameras.

  get_server_info(self)
      Return the info of the cam server instance.
      For administrative purposes only.
      :return: Status of the server

  set_camera_config(self, camera_name, configuration)
      Set config on camera.
      :param camera_name: Camera to set the config to.
      :param configuration: Config to set, in dictionary format.
      :return: Actual applied config.

  stop_all_instances(self)
      Stop all the cameras on the server.
      :return: Response.

  stop_instance(self, camera_name)
      Stop the camera.
      :param camera_name: Name of the camera to stop.
      :return: Response.

```

#### PipelineClient
Import and create a pipeline client instance:
```python
from cam_server import PipelineClient
client = PipelineClient()
```

Class definition:
```
class PipelineClient(builtins.object)

  __init__(self, address='http://sf-daqsync-01:8889/')
      :param address: Address of the pipeline API, e.g. http://localhost:10000

  collect_background(self, camera_name, n_images=None)
      Collect the background image on the selected camera.
      :param camera_name: Name of the camera to collect the background on.
      :param n_images: Number of images to collect the background on.
      :return: Background id.

  create_instance_from_config(self, configuration, instance_id=None)
      Create a pipeline from the provided config. Pipeline config can be changed.
      :param configuration: Config to use with the pipeline.
      :param instance_id: User specified instance id. GUID used if not specified.
      :return: Pipeline instance stream.

  create_instance_from_name(self, pipeline_name, instance_id=None)
      Create a pipeline from a config file. Pipeline config can be changed.
      :param pipeline_name: Name of the pipeline to create.
      :param instance_id: User specified instance id. GUID used if not specified.
      :return: Pipeline instance stream.

  delete_pipeline_config(self, pipeline_name)
      Delete a pipeline config.
      :param pipeline_name: Name of pipeline config to delete.

  get_cameras(self)
      List available cameras.
      :return: Currently available cameras.

  get_instance_config(self, instance_id)
      Return the instance configuration.
      :param instance_id: Id of the instance.
      :return: Pipeline configuration.

  get_instance_info(self, instance_id)
      Return the instance info.
      :param instance_id: Id of the instance.
      :return: Pipeline instance info.

  get_instance_stream(self, instance_id)
      Return the instance stream. If the instance does not exist, it will be created.
      Instance will be read only - no config changes will be allowed.
      :param instance_id: Id of the instance.
      :return: Pipeline instance stream.

  get_latest_background(self, camera_name)
      Return the latest collected background for a camera.
      :param camera_name: Name of the camera to return the background.
      :return: Background id.

  get_pipeline_config(self, pipeline_name)
      Return the pipeline configuration.
      :param pipeline_name: Name of the pipeline.
      :return: Pipeline configuration.

  get_pipelines(self)
      List existing pipelines.
      :return: Currently existing cameras.

  get_server_info(self)
      Return the info of the cam server instance.
      For administrative purposes only.
      :return: Status of the server

  save_pipeline_config(self, pipeline_name, configuration)
      Set config of the pipeline.
      :param pipeline_name: Pipeline to save the config for.
      :param configuration: Config to save, in dictionary format.
      :return: Actual applied config.

  set_instance_config(self, instance_id, configuration)
      Set config of the instance.
      :param instance_id: Instance to apply the config for.
      :param configuration: Config to apply, in dictionary format.
      :return: Actual applied config.

  stop_all_instances(self)
      Stop all the pipelines on the server.

  stop_instance(self, instance_id)
      Stop the pipeline.
      :param instance_id: Name of the pipeline to stop.

  get_instance_message(self, instance_id):

      Get a single message from a stream instance.
      :param instance_id: Instance id of the stream.
      :return: Message from the stream.

```

<a id="rest_api"></a>
### REST API

#### Camera server API

**WARNING**: Please note that for normal users, only the **Pipeline server API** should be used. The Camera server API
is used by the underlying infrastructure to provide camera images to the pipeline server.

In the API description, localhost and port 8888 are assumed. Please change this for your specific case.

* `GET localhost:8888/api/v1/cam` - get the list of available cameras.
    - Response specific field: "cameras" - List of cameras.

* `GET localhost:8888/api/v1/cam/<camera_name>` - get the camera stream.
    - Response specific field: "stream" - Stream address.

* `GET localhost:8888/api/v1/cam/<camera_name>/config` - get camera config.
    - Response specific field: "config" - configuration JSON.

* `POST localhost:8888/api/v1/cam/<camera_name>/config` - set camera config.
    - Response specific field: "config" configuration JSON.

* `DELETE localhost:8888/api/v1/cam/<camera_name>/config` - delete the camera config.
    - Response specific field: None

* `GET localhost:8888/api/v1/cam/<camera_name>/geometry` - get the geometry of the camera.
    - Response specific field: "geometry" - \[width, height\] of image

* `GET localhost:8888/api/v1/cam/<camera_name>/image` - get one PNG image of the camera.
    - Returns a PNG image

* `GET localhost:8888/api/v1/cam/<camera_name>/image_bytes` - get one PNG image of the camera.
    - Returns JSON with Base64, UTF-8 image bytes and metadata.

* `GET localhost:8888/api/v1/cam/info` - return info on the camera manager.
    - Response specific field: "info" - JSON with instance info.

* `DELETE localhost:8888/api/v1/cam` - stop all camera instances.
    - Response specific field: None

* `DELETE localhost:8888/api/v1/cam/<camera_name>` - stop the camera instance.
    - Response specific field: None

#### Pipeline server API

In the API description, localhost and port 8889 are assumed. Please change this for your specific case.

* `GET localhost:8889/api/v1/pipeline` - get the list of available pipelines.
    - Response specific field: "pipelines" - List of pipelines.

* `POST localhost:8889/api/v1/pipeline?instance_id=id` - create a pipeline by passing its config as a JSON payload.
    - Query parameter: instance_id (optional) - request a specific instance id. Must be unique.
    - Response specific field: "instance_id", "stream", "config"

* `POST localhost:8889/api/v1/pipeline/<pipeline_name>?instance_id=id` - create a pipeline from a named config.
    - Query parameter: instance_id (optional) - request a specific instance id. Must be unique.
    - Response specific field: "instance_id", "stream", "config"

* `GET localhost:8889/api/v1/pipeline/instance/<instance_id>` - get pipeline instance stream address.
    - Response specific field: "stream" - Stream address of the pipeline instance.

* `GET localhost:8889/api/v1/pipeline/instance/<instance_id>/info` - get pipeline instance info.
    - Response specific field: "info" - JSON with instance info.

* `GET localhost:8889/api/v1/pipeline/instance/<instance_id>/config` - get pipeline instance config.
    - Response specific field: "config" - JSON instance config.

* `POST localhost:8889/api/v1/pipeline/instance/<instance_id>/config` - set pipeline instance config - JSON payload.
    - Response specific field: "config" - JSON instance config.

* `GET localhost:8889/api/v1/pipeline/<pipeline_name>/config` - get named pipeline config.
    - Response specific field: "config" - JSON named pipeline config.

* `POST localhost:8889/api/v1/pipeline/<pipeline_name>/config` - set named pipeline config - JSON payload.
    - Response specific field: "config" - JSON named pipeline config.

* `GET localhost:8889/api/v1/pipeline/camera` - return the list of available cameras.
    - Response specific field: "cameras" - List of cameras.

* `POST localhost:8889/api/v1/pipeline/camera/<camera_name>/background?n_images=10` - collect background for the camera.
    - Query parameter: n_images (optional) = how many images to average for the background.
    - Response specific field: "background_id" - ID of the acquired background.

* `GET localhost:8889/api/v1/pipeline/camera/<camera_name>/background` - return latest background for camera.
    - Response specific field: "background_id" - ID of the latest background for the camera.

* `GET localhost:8889/api/v1/pipeline/info` - return info on the pipeline manager.
    - Response specific field: "info" -  JSON with instance info.

* `DELETE localhost:8889/api/v1/pipeline` - stop all pipeline instances.
    - Response specific field: None

* `DELETE localhost:8889/api/v1/pipeline/<instance_id>` - stop the pipeline instance.
    - Response specific field: None

<a id="running_the_servers"></a>
## Running the servers

The scripts for running the existing server are located under the **cam\_server/** folder.

The two servers are:

- **Camera server** (start_camera_server.py): Converts epics cameras into bsread cameras.
- **Pipeline server** (start_pipeline_server.py): Processes cameras in bsread format.

You can also use the docker container directly - it setups and starts both servers.

Before you can run the servers, you need to have (and specify where you have it) the cameras, pipelines and background
configurations. In this repo, you can find the test configurations inside the **tests/** folder. To use the
production configuration, see **Production configuration** chapter below.

<a id="run_camera_server"></a>
### Camera server

```bash
usage: start_camera_server.py [-h] [-p PORT] [-i INTERFACE] [-b BASE]
                              [-n HOSTNAME]
                              [--log_level {CRITICAL,ERROR,WARNING,INFO,DEBUG}]

Camera acquisition server

optional arguments:
  -h, --help            show this help message and exit
  -p PORT, --port PORT  Server cam_port
  -i INTERFACE, --interface INTERFACE
                        Hostname interface to bind to
  -b BASE, --base BASE  (Camera) Configuration base directory
  -n HOSTNAME, --hostname HOSTNAME
                        Hostname to use when returning the stream address.
  --log_level {CRITICAL,ERROR,WARNING,INFO,DEBUG}
                        Log level to use.
```

<a id="run_pipeline_server"></a>
### Pipeline server

```bash
usage: start_pipeline_server.py [-h] [-c CAM_SERVER] [-p PORT] [-i INTERFACE]
                                [-b BASE] [-g BACKGROUND_BASE] [-u USER_SCRIPTS_BASE]  [-n HOSTNAME]
                                [--log_level {CRITICAL,ERROR,WARNING,INFO,DEBUG}]

Pipeline processing server

optional arguments:
  -h, --help            show this help message and exit
  -c CAM_SERVER, --cam_server CAM_SERVER
                        Cam server rest api address.
  -p PORT, --port PORT  Server port
  -i INTERFACE, --interface INTERFACE
                        Hostname interface to bind to
  -b BASE, --base BASE  (Pipeline) Configuration base directory
  -g BACKGROUND_BASE, --background_base BACKGROUND_BASE
  -n HOSTNAME, --hostname HOSTNAME
                        Hostname to use when returning the stream address.
  --log_level {CRITICAL,ERROR,WARNING,INFO,DEBUG}
                        Log level to use.

```

<a id="run_docker_container"></a>
### Docker container
To execute the application inside a docker container, you must first start it (from the project root folder):
```bash
docker run --net=host -it -v /CURRENT_DIR/tests:/configuration docker.psi.ch:5000/cam_server
```

**WARNING**: Docker needs (at least on OSX) a full path for the -v option. Replace the **CURRENT\_DIR** with your
actual path.

This will map the test configuration (-v option) to the /configuration folder inside the container.
If you need to have the production configuration, see the next chapter.

Once in the container bash, you can start the two servers:
```bash
camera_server & pipeline_server &
```

<a id="production_configuration"></a>
## Production configuration

The production configurations are not part of this repository but are available on:
- https://github.com/paulscherrerinstitute/cam_server/cam_server_configuration

You can download it using git:
```bash
git clone https://github.com/paulscherrerinstitute/cam_server/cam_server_configuration.git
```

And later, when you start the docker container, map the configuration using the **-v** parameter:
```bash
docker run --net=host -it -v /CURRENT_DIR/cam_server_configuration/configuration:/configuration docker.psi.ch:5000/cam_server
```

**WARNING**: Docker needs (at least on OSX) a full path for the -v option. Replace the **CURRENT\_DIR** with your
actual path.

<a id="examples"></a>
## Examples

<a id="get_simulation_camera_stream"></a>
### Get the simulation camera stream

**WARNING**: This example should not be used by normal users. CamClient is used by the
underlying infrastructure to provide camera images to the pipeline server. See PipelineClient for a
user oriented client.

This is just an example on how you can retrieve the raw image from the camera. You **should not** do this for
the normal use case. See next example for a more common use.

```python
from cam_server import CamClient
from cam_server.utils import get_host_port_from_stream_address
from bsread import source, SUB

# Initialize the client.
camera_client = CamClient()

# Get stream address of simulation camera. Stream address in format tcp://hostname:port.
camera_stream_address = camera_client.get_instance_stream("simulation")

# Extract the stream hostname and port from the stream address.
camera_host, camera_port = get_host_port_from_stream_address(camera_stream_address)

# Subscribe to the stream.
with source(host=camera_host, port=camera_port, mode=SUB) as stream:
    # Receive next message.
    data = stream.receive()

image_width = data.data.data["width"].value
image_height = data.data.data["height"].value
image_bytes = data.data.data["image"].value

print("Image size: %d x %d" % (image_width, image_height))
print("Image data: %s" % image_bytes)
```

<a id="basic_pipeline"></a>
### Get a basic pipeline with a simulated camera

In contrast with the example above, where we just request a camera stream, we have to create a pipeline instance
in this example. We create a pipeline instance by specifying which camera - simulation in our example - to use as
the pipeline source.

By not giving any additional pipeline parameters, the image will in fact be exactly the same as the one in the
previous example. Even if requesting a raw camera image, it is still advisable to use the PipelineClient (and create
an empty pipeline as in the example below) because the CamClient might change and be moved to a different server out
of your reach.

```python
from cam_server import PipelineClient
from cam_server.utils import get_host_port_from_stream_address
from bsread import source, SUB

# Initialize the client.
pipeline_client = PipelineClient()

# Setup the pipeline config. Use the simulation camera as the pipeline source.
pipeline_config = {"camera_name": "simulation"}

# Create a new pipeline with the provided configuration. Stream address in format tcp://hostname:port.
instance_id, pipeline_stream_address = pipeline_client.create_instance_from_config(pipeline_config)

# Extract the stream hostname and port from the stream address.
pipeline_host, pipeline_port = get_host_port_from_stream_address(pipeline_stream_address)

# Subscribe to the stream.
with source(host=pipeline_host, port=pipeline_port, mode=SUB) as stream:
    # Receive next message.
    data = stream.receive()

image_width = data.data.data["width"].value
image_height = data.data.data["height"].value
image_bytes = data.data.data["image"].value

print("Image size: %d x %d" % (image_width, image_height))
print("Image data: %s" % image_bytes)
```

<a id="private_pipeline"></a>
### Create a pipeline instance with background

This example is the continuation of the previous example. Before creating our own pipeline, we collect the
background for the simulation camera and apply it to the pipeline.

```python
from cam_server import PipelineClient

camera_name = "simulation"

# Initialize the client.
pipeline_client = PipelineClient()

# Collect the background for the given camera.
background_id = pipeline_client.collect_background(camera_name)

# Setup the pipeline config. Use the simulation camera as the pipeline source, and the collected background.
pipeline_config = {"camera_name": camera_name,
                   "background_id": background_id}

# Create a new pipeline with the provided configuration. Stream address in format tcp://hostname:port.
instance_id, pipeline_stream_address = pipeline_client.create_instance_from_config(pipeline_config)

# TODO: Continue as in the example above.
```

<a id="read_camera_stream"></a>
### Read the stream for a given camera name

```python
from cam_server import PipelineClient
from cam_server.utils import get_host_port_from_stream_address
from bsread import source, SUB

# Define the camera name you want to read.
camera_name = "SAROP21-PPRM102"

# First create the pipeline for the selected camera.
client = PipelineClient()
instance_id, stream_address = client.create_instance_from_config({"camera_name": camera_name})

# Extract the stream host and port from the stream_address.
stream_host, stream_port = get_host_port_from_stream_address(stream_address)

# Check if your instance is running on the server.
if instance_id not in client.get_server_info()["active_instances"]:
    raise ValueError("Requested pipeline is not running.")

# Open connection to the stream. When exiting the 'with' section, the source disconnects by itself.
with source(host=stream_host, port=stream_port, mode=SUB) as input_stream:
    input_stream.connect()

    # Read one message.
    message = input_stream.receive()

    # Print out the received stream data - dictionary.
    print("Dictionary with data:\n", message.data.data)

    # Print out the X center of mass.
    print("X center of mass: ", message.data.data["x_center_of_mass"].value)

```

<a id="modify_camera_config"></a>
### Modifying camera config
When modifying the camera config, the changes are applied immediately. As soon as you call **set\_camera\_config** the
changes will be reflected in the camera stream within a couple of frames.

```python
from cam_server import CamClient

# Initialize the camera client.
cam_client = CamClient()

# Print the list of available cameras.
print(cam_client.get_cameras())

# TODO: Put the name of the camera you want to modify.
camera_to_modify = "test_camera"

# Retrieve the camera config.
camera_config = cam_client.get_camera_config(camera_to_modify)

# Change the mirror_x setting.
camera_config["mirror_x"] = False
# Change the camera_calibration setting.
camera_config["camera_calibration"] = {
    "reference_marker": [ 0, 0, 100, 100 ],
    "reference_marker_width": 100.0,
    "reference_marker_height": 100.0,
    "angle_horizontal": 0.0,
    "angle_vertical": 0.0
}

# Save the camera configuration.
cam_client.set_camera_config(camera_to_modify, camera_config)

# You can also save the same (or another) config under a different camera name.
cam_client.set_camera_config("camera_to_delete", camera_config)

# And also delete camera configs.
cam_client.delete_camera_config("camera_to_delete")
```

<a id="modify_pipeline_config"></a>
### Modifying pipeline config
Please note that modifying the pipeline config works differently than modifying the camera config.
When using **save\_pipeline\_config**, the changes do not affect existing pipelines. You need to restart or recreate
the pipeline for the changes to be applied.

You can however modify an existing instance config, by calling **set\_instance\_config**. The changes will be
reflected in the pipeline stream within a couple of frames.

```python
from cam_server import PipelineClient

# Initialize the pipeline client.
pipeline_client = PipelineClient()

# Print the list of available pipelines.
print(pipeline_client.get_pipelines())

# TODO: Put the name of the pipeline you want to modify.
pipeline_to_modify = "test_pipeline"

# Retrieve the camera config.
pipeline_config = pipeline_client.get_pipeline_config(pipeline_to_modify)

# Change the image threshold.
pipeline_config["image_threshold"] = 0.5
# Change the image region of interest.
pipeline_config["image_region_of_interest"] = [0, 100, 0, 100]

# Save the camera configuration.
pipeline_client.save_pipeline_config(pipeline_to_modify, pipeline_config)

# You can also save the same (or another) config under a different camera name.
pipeline_client.save_pipeline_config("pipeline_to_delete", pipeline_config)

# And also delete camera configs.
pipeline_client.delete_pipeline_config("pipeline_to_delete")
```

<a id="create_camera_config"></a>
### Create a new camera
This example shows how to create a new camera config.

```python
from cam_server import CamClient

# Initialize the camera client.
cam_client = CamClient()

# Specify the desired camera config.
camera_config = {
  "name": "camera_example_3",
  "source": "EPICS:CAM1:EXAMPLE",
  "source_type": "epics",
  "mirror_x": False,
  "mirror_y": False,
  "rotate": 0,

  "camera_calibration": {
    "reference_marker": [ 0, 0, 100, 100 ],
    "reference_marker_width": 100.0,
    "reference_marker_height": 100.0,
    "angle_horizontal": 0.0,
    "angle_vertical": 0.0
  }
}

# Specify the new camera name.
new_camera_name = "new_camera_name"

# Save the camera configuration.
cam_client.set_camera_config(new_camera_name, camera_config)

# In case you need to, delete the camera config you just added.
# cam_client.cam_client.delete_camera_config(new_camera_name)
```

<a id="single_message_screen_panel"></a>
### Get single message from screen_panel stream
You should have the screen_panel open, with the camera you want to acquire running. We will connect to the same
stream instance as the screen_panel uses, which means that all the changes done in the screen panel will also be
reflected in our acquisition (you can configure what you want to acquire in the screen panel).

```python
from cam_server import PipelineClient

# Instantiate the pipeline client.
pipeline_client = PipelineClient()

# Name of the camera we want to get a message from.
camera_name = "simulation"
# Screen panel defines the instance name as: [CAMERA_NAME]_sp1
instance_name = camera_name + "_sp1"

# Get the data.
data = pipeline_client.get_instance_message(instance_name)
```

<a id="stream_to_h5_file"></a>
### Save camera stream to H5 file
```python
from bsread import h5, SUB
from cam_server import PipelineClient

camera_name = "simulation"
file_name = "output.h5"
n_messages = 10

client = PipelineClient()

instance_id, stream_address = client.create_instance_from_config({"camera_name": camera_name})

# The output file 'output.h5' has 10 images from the simulation camera stream.
h5.receive(source=stream_address, file_name=file_name, mode=SUB, n_messages=n_messages)
```

#### Commandline

Following script can be used to dump the camera stream to an hdf5 file using the `bs h5` tool. The process to use this script is to

1) Open ScreenPanel and select desired camera
2) (Optional) Configure all the calculations you are interested in in the ScreenPanel
3) Use the script as follows (if you name the script `record_camera`): `record_camera <camera_name> <filename>`

```bash
#!/bin/bash

if (( $# != 2 )); then
    echo "usage: $0 <camera> <filename>"
    exit -1
fi

CAMERA_NAME=$1
FILENAME=$2

STREAM=$(curl -s http://sf-daqsync-01:8889/api/v1/pipeline/instance/${CAMERA_NAME}_sp1 | sed -e 's/.*"stream": "\([^"]*\)".*/\1/')
echo $STREAM
bs h5 -s $STREAM -m sub $FILENAME
```

<a id="deploy_in_production"></a>
## Deploy in production

Before deploying in production, make sure the latest version was tagged in git (this triggers the Travis build) and
that the Travis build completed successfully (the new cam_server package in available in anaconda). After this 2 steps,
you need to build the new version of the docker image (the docker image checks out the latest version of cam_server
from Anaconda). The docker image version and the cam_server version should always match - If they don't, something went
wrong.

### Production configuration
Login to the target system, where cam_server will be running. Checkout the production configuration into the root
of the target system filesystem.

```bash
cd /
git clone https://github.com/paulscherrerinstitute/cam_server/cam_server_configuration.git
```

### Setup the cam_server as a service
On the target system, copy **docker/camera_server.service** and **docker/pipeline_server.service** into
**/etc/systemd/system**.

Then need to reload the systemctl daemon:
```bash
systemctl daemon-reload
```

### Verifying the configuration
On the target system, copy **docker/validate_configs.sh** into your home folder.
Run it to verify if the deployed configurations are valid for the current version of the cam_server.

```bash
./validate_configs.sh
```

### Run the servers
Using systemctl you then run both servers:
```bash
systemctl start camera_server.service
systemctl start pipeline_server.service
```

### Inspecting server logs
To inspect the logs for each server, use journalctl:
```bash
journalctl -u camera_server.service
journalctl -u pipeline_server.service
```

Note: The '-f' flag will make you follow the log file.
