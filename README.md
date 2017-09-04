[![Build Status](https://travis-ci.org/datastreaming/cam_server.svg?branch=master)](https://travis-ci.org/datastreaming/cam_server) [![Build status](https://ci.appveyor.com/api/projects/status/0vyk18qxnqk2cmvx?svg=true)](https://ci.appveyor.com/project/Babicaa/cam-server)

# Camera and Pipeline server
Cam server is an epics - bsread interface that converts epics enabled camera into a bs_read stream. In addition it 
also provides a processing pipeline and a REST interface to control both the cameras and the pipeline.

# Table of content
1. [Build](#build)
    1. [Conda setup](#conda_setup)
    2. [Local build](#local_build)
    3. [Docker build](#docker_build)
2. [Basic concepts](#basic_concepts)
    1. [Requesting a stream and instance management](#reqeust_a_stream)
    2. [Shared and private pipeline instances](#shared_and_private)
    3. [Configuration versioning and camera background in the pipeline server](#configuration_versioning)
3. [Configuration](#configuration)
    1. [Camera configuration](#camera_configuration)
    2. [Pipeline configuration](#pipeline_configuration)
4. [Web interface](#web_interface)
    1. [Python client](#python_client)
    2. [REST API](#rest_api)
5. [Running the servers](#running_the_servers)
    1. [Camera_server](#run_camera_server)
    2. [Pipeline server](#run_pipeline_server)
    3. [Docker Container](#run_docker_container)
6. [Production configuration](#production_configuration)
7. [Examples](#examples)
    1. [Get the simulation camera stream](#get_simulation_camera_stream)
    2. [Get a basic pipeline with a simulated camera](#basic_pipeline)
    3. [Create a pipeline instance with background](#private_pipeline)
    3. [Read the stream for a given camera name](#read_camera_stream)
8. [Deploy in production](#deploy_in_production)
    
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
- bsread >=0.9.3
- bottle
- numpy
- scipy
- pyepics
- matplotlib
- pillow

In case you are using conda to install the packages, you might need to add the **paulscherrerinstitute** channel to 
your conda config:

```
conda config --add channels paulscherrerinstitute
```

<a id="docker_build"></a>
### Docker build
To build the docker image run (from project root):
```bash
./docker/build.sh
```

Before building the docker image, make sure the latest version of the library is available in anaconda.

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

- configuration/camera : Folder where JSON files with camera configurations are located.
- configuration/pipeline : Folder where JSON files with pipeline configurations are located.
- configuration/background : Folder where NPY files with camera backgrounds are located.

<a id="camera_configuration"></a>
### Camera configuration
For camera configuration, all fields must be specified, and there is no defaulting in case some are missing.

#### Configuration parameters

- **name**: Name of the camera.
- **prefix**: PV prefix to connect to the camera.
- **mirror\_x**: Mirror camera image over X axis.
- **mirror\_y**: Mirror camera image over Y axis.
- **rotate**: how many times to rotate the camera image by 90 degrees.

#### Example
```json
{
  "name": "example_4",
  "prefix": "EPICS_example_4",
  "mirror_x": true,
  "mirror_y": false,
  "rotate": 4
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

- **camera\_name** : Name of the camera to use as a pipeline source.
- **camera\_calibration** (Default _None_): Info on how to convert the camera pixels into engineering units.
    - reference_marker (Default _[0, 0, 100, 100]_): Reference markers placement.
    - reference_marker_width (Default _100.0_): Width of reference markers.
    - reference_marker_height (Default _100.0_): Height of reference markers.
    - angle_horizontal (Default _0.0_): Horizontal angle.
    - angle_vertical (Default _0.0_): Vertical angle.
- **image\_background** (Default _None_): Background to subtract from the original image.
- **image\_background_enable** (Default _False_): Enable or disale the image_background subtraction.
- **image\_threshold** (Default _None_): Minimum value of each pixel. Pixels below the threshold are converted to 0.
- **image\_region\_of\_interest** (Default _None_): Crop the image before processing.
- **image\_good\_region** (Default _None_): Good region to use for fits and slices.
    - threshold (Default _0.3_): Threshold to apply on each pixel.
    - gfscale (Default _1.8_): Scale to extend the good region.
- **image\_slices** (Default _None_): 
    - number_of_slices (Default _1_): Desired number of slices.
    - scale (Default _2.0_): Good region scale in for slicing purposes.

#### Example
```json
{
  "camera_name": "simulation",

  "camera_calibration": {
    "reference_marker": [ 0, 0, 100, 100 ],
    "reference_marker_width": 100.0,
    "reference_marker_height": 100.0,
    "angle_horizontal": 0.0,
    "angle_vertical": 0.0
  },

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
    "scale": 1.0
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
Import and create a cam client instance:
```python
from cam_server import CamClient
client = CamClient()
```

Class definition:
```
class CamClient()
  
  __init__(self, address='http://0.0.0.0:8888/')
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
  
  get_camera_stream(self, camera_name)
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
  
  stop_all_cameras(self)
      Stop all the cameras on the server.
      :return: Response.
  
  stop_camera(self, camera_name)
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
  
  __init__(self, address='http://0.0.0.0:8889/')
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

```

<a id="rest_api"></a>
### REST API

#### Camera server API

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
                                [-b BASE] [-g BACKGROUND_BASE] [-n HOSTNAME]
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
- https://git.psi.ch/controls_highlevel_applications/cam_server_configuration

You can download it using git:
```bash
git clone https://git.psi.ch/controls_highlevel_applications/cam_server_configuration.git
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

This is just an example on how you can retrieve the raw image from the camera. You **should not** do this for 
the normal use case. See next example for a more common use.

```python
from cam_server import CamClient
from cam_server.utils import get_host_port_from_stream_address
from bsread import source, SUB

# Change to match your camera server
server_address = "http://0.0.0.0:8888"

# Initialize the client.
camera_client = CamClient(server_address)

# Get stream address of simulation camera. Stream address in format tcp://hostname:port.
camera_stream_address = camera_client.get_camera_stream("simulation")

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

# Change to match your pipeline server
server_address = "http://0.0.0.0:8889"

# Initialize the client.
pipeline_client = PipelineClient(server_address)

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

# Change to match your pipeline server
server_address = "http://0.0.0.0:8889"
camera_name = "simulation"

# Initialize the client.
pipeline_client = PipelineClient(server_address)

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
client = PipelineClient("http://sf-daqsync-01:8889")
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
git clone https://git.psi.ch/controls_highlevel_applications/cam_server_configuration.git
```

### Setup the cam_server as a service
On the target system, copy **docker/camera_server.service** and **docker/pipeline_server.service** into 
**/etc/systemd/system**.

Then need to reload the systemctl daemon:
```bash
systemctl daemon-reload
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