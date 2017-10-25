#!/usr/bin/env bash

/usr/bin/docker kill validate_configs
/usr/bin/docker rm validate_configs
/usr/bin/docker run -it --name validate_configs -v /cam_server_configuration/configuration:/configuration docker.psi.ch:5000/cam_server validate_configs -c /configuration/camera_config -p /configuration/pipeline_config -b /configuration/background_config
