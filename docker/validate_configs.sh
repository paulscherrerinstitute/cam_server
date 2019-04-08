#!/usr/bin/env bash

/usr/bin/docker kill validate_configs
/usr/bin/docker rm validate_configs
/usr/bin/docker pull paulscherrerinstitute/cam_server
/usr/bin/docker run -it --name validate_configs -v /cam_server_configuration/configuration:/configuration paulscherrerinstitute/cam_server validate_configs -c /configuration/camera_config -p /configuration/pipeline_config -b /configuration/background_config
