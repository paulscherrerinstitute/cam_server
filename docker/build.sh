#!/bin/bash
VERSION=3.0.0
docker build --no-cache=true -t paulscherrerinstitute/cam_server .
docker tag paulscherrerinstitute/cam_server paulscherrerinstitute/cam_server:$VERSION
docker login
docker push paulscherrerinstitute/cam_server:$VERSION
docker push paulscherrerinstitute/cam_server

