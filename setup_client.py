from setuptools import setup


setup(name="cam_server",
      version="4.2.4",
      maintainer="Paul Scherrer Institute",
      maintainer_email="daq@psi.ch",
      author="Paul Scherrer Institute",
      author_email="daq@psi.ch",
      description="CamServer client classes.",

      license="GPL3",

      packages=["cam_server_client",
                ],

      zip_safe = False
      )
