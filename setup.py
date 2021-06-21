#! /usr/bin/env python
#
# Copyright (C) 2012 Mathieu Blondel

import sys
import os
from setuptools import Extension, find_packages, setup
import numpy as np
from numpy.distutils.core import setup
from distutils.version import LooseVersion


DISTNAME = 'mondrian_forest'
DESCRIPTION = "Implementation of mondrian_forest " + \
              "based on sklearn tree code."
URL = 'https://github.com/scikit-optimize/mondrian_forest'
LICENSE = 'new BSD'
VERSION = '0.0.dev0'

CYTHON_MIN_VERSION = '0.23'


message = ('Please install cython with a version >= {0} in order '
           'to build a scikit-garden development version.').format(
           CYTHON_MIN_VERSION)
try:
    import Cython
    if LooseVersion(Cython.__version__) < CYTHON_MIN_VERSION:
        message += ' Your version of Cython was {0}.'.format(
            Cython.__version__)
        raise ValueError(message)
    from Cython.Build import cythonize
except ImportError as exc:
    exc.args += (message,)
    raise

libraries = []
if os.name == 'posix':
    libraries.append('m')

extensions = []
for name in ['_tree', '_splitter', '_criterion', '_utils']:
    extensions.append(Extension(
        'mondrian_forest.tree.{}'.format(name),
        sources=['mondrian_forest/tree/{}.pyx'.format(name)],
        include_dirs=[np.get_include()],
        libraries=libraries,
        extra_compile_args=['/std:c++17','/O2','/GT'],
    ))
extensions = cythonize(extensions,compiler_directives={'language_level':'3','infer_types':True})

if __name__ == "__main__":
    setup(name=DISTNAME,
          packages=find_packages(),
          include_package_data=True,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: C',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'
             ],
          install_requires=["numpy", "scipy", "scikit-learn>=0.18", "cython"],
          setup_requires=["cython"],
          ext_modules=extensions
          )
