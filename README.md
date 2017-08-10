[![Build Status](https://travis-ci.org/datastreaming/cam_server.svg?branch=master)](https://travis-ci.org/datastreaming/cam_server) [![Build status](https://ci.appveyor.com/api/projects/status/0vyk18qxnqk2cmvx?svg=true)](https://ci.appveyor.com/project/Babicaa/cam-server)

# Camera and Pipeline server
Cam server is an epics - bsread interface that converts epics enabled camera into a bs_read stream. In addition it 
also provides a processing pipeline and a REST interface to control both the cameras and the pipeline.

## Conda setup
If you use conda, you can create an environment with the cam_server library by running:

```bash
conda create -c paulscherrerinstitute --name <env_name> cam_server
```

After that you can just source you newly created environment and start using the server.

## Local build
You can build the library by running the setup script in the root folder of the project:

```bash
python setup.py install
```

or by using the conda also from the root folder of the project:

```bash
conda build conda-recipe
conda install --use-local mflow_node_processors
```

## Docker build
To use the docker image run 
```bash
./docker/build.sh
```

### Requirements
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

## Basic concepts

### Requesting a stream and instance management

### Shared and private pipeline instances

### Configuration versioning and camera background

## Configuration
The camera and pipeline instances have their own configuration which can be read and set via the rest interface.
We currently store the configurations in files, but REDIS is planned to be used in the near future.

The expected folder structure for the configuration (which can be changed by passing parameters to the executables):

- configuration/camera : Folder where JSON files with camera configurations are located.
- configuration/pipeline : Folder where JSON files with pipeline configurations are located.
- configuration/background : Folder where NPY files with camera backgrounds are located.

### Camera configuration
For camera configuration, all fields must be specified, and there is no defaulting in case some are missing.

Example:
```json
{
  "name": "example_4",
  "prefix": "EPICS_example_4",
  "mirror_x": true,
  "mirror_y": false,
  "rotate": 4
}
```

### Pipeline configuration

Configuration changes can for the pipeline can be incremental - you need to specify only the fields that you want 
to change. A valid configuration must have only the **camera_name** specified, all other fields will be 
defaulted to **None**. The complete configuration used for the pipeline is added to the output bsread stream in the 
**processing\_parameters** field.

Example:
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

### Camera server API

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

### Pipeline server API

In the API description, localhost and port 8889 are assumed. Please change this for your specific case.

* `GET localhost:8889/api/v1/pipeline` - get the list of available pipelines.
    - Response specific field: "pipelines" - List of pipelines.
* `POST localhost:8889/api/v1/pipeline` - create a pipeline by passing its config as a JSON payload.
    - Response specific field: "instance_id", "stream", "config"
* `POST localhost:8889/api/v1/pipeline/<pipeline_name>` - create a pipeline from a named config.
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
* `POST localhost:8889/api/v1/pipeline/camera/<camera_name>/background` - collect background for the camera.
    - Response specific field: "background_id" - ID of the acquired background.
* `GET localhost:8888/api/v1/pipeline/info` - return info on the pipeline manager.
    - Response specific field: "info" -  JSON with instance info.
* `DELETE localhost:8888/api/v1/pipeline` - stop all pipeline instances.
    - Response specific field: None
* `DELETE localhost:8888/api/v1/pipeline/<instance_id>` - stop the pipeline instance.
    - Response specific field: None
    

## Python client

## API

## Running the servers

The scripts for running the existing server are located under the **cam\_server/** folder.

The two servers are:

- **Camera server** (start_camera_server.py): Converts epics cameras into bsread cameras.
- **Pipeline server** (start_pipeline_server.py): Processes cameras in bsread format.

You can also use the docker container directly - it setups and starts both servers.

Before you can run the servers, you need to have (and specify where you have it) the cameras, pipelines and background 
configurations. This configurations are not part of this repository but are available on:
- https://git.psi.ch/controls_highlevel_applications/cam_server_configuration

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

### Docker container

## Examples

### Get the simulation camera stream

### Get a basic pipeline with a simulated camera

### Create a private pipeline instance and collect the camera background
