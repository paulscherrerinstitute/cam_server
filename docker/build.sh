#!/bin/bash
VERSION=0.0.3
docker build --no-cache=true -t docker.psi.ch:5000/cam_server .
docker tag docker.psi.ch:5000/cam_server docker.psi.ch:5000/cam_server:$VERSION
docker push docker.psi.ch:5000/cam_server:$VERSION
docker push docker.psi.ch:5000/cam_server
