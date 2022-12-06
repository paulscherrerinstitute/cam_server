LIBRARY_NAME = "mylib"
VERSION = 1.0
DESCRIPTION = 'This is a demo package'


from distutils.core import setup, Extension

import numpy

setup (name=LIBRARY_NAME,
       version=VERSION,
       description=DESCRIPTION,
       include_dirs=[numpy.get_include()],
       ext_modules=[Extension(LIBRARY_NAME, sources=[LIBRARY_NAME + '.c'])])
