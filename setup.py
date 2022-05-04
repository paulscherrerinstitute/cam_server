import os
from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name="cam_server",
      version="5.2.1",
      maintainer="Paul Scherrer Institute",
      maintainer_email="daq@psi.ch",
      author="Paul Scherrer Institute",
      author_email="daq@psi.ch",
      description="BSREAD image processing pipeline and EPICS cameras converter.",

      license="GPL3",

      packages=['cam_server',
                "cam_server.camera",
                "cam_server.camera.rest_api",
                "cam_server.camera.source",
                "cam_server.instance_management",
                "cam_server.pipeline",
                "cam_server.pipeline.data_processing",
                "cam_server.pipeline.rest_api",
                "cam_server.pipeline.types",
                "cam_server_client"
            ],

      package_data={
                # If any package contains *.html, include them:
                '': ['*.html', '*.js',],
            },

      # long_description=read('Readme.md'),
      )
