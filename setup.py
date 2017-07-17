import os
from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name="cam_server",
      version="0.0.1",
      maintainer="Paul Scherrer Institute",
      maintainer_email="daq@psi.ch",
      author="Paul Scherrer Institute",
      author_email="daq@psi.ch",
      description="Camera server to convert epics enabled cameras into bsread cameras.",

      license="GPL3",

      packages=['cam_server',
                "cam_server.camera",
                "cam_server.rest_api"],

      long_description=read('Readme.md'),

      install_requires=[
          'requests',
          'bsread',
          'bottle',
          'numpy',
          'scipy',
          'pyepics',
          'matplotlib'],
      )
