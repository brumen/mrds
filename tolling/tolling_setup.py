import config
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("tolling_fast", ["tolling_fast.pyx"])]
setup(name = 'Fast functions for tolling',
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules
    )

# compile using python tolling_cy_setup.py build_ext --inplace
