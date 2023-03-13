#!/bin/bash
VERSION= 5.5.4
#docker image prune --all --filter "until=4320h"   #delete images older than 6 months
#docker system prune
docker buildx build --platform linux/amd64  --no-cache=true --progress=plain -t paulscherrerinstitute/cam_server .
docker tag paulscherrerinstitute/cam_server paulscherrerinstitute/cam_server:$VERSION
docker login
docker push paulscherrerinstitute/cam_server:$VERSION
docker push paulscherrerinstitute/cam_server

