import config 
from distutils.core import setup
from distutils.extension import Extension
import cython 
from Cython.Distutils import build_ext

ext_modules = [Extension("pricers_fast", ["pricers_fast.pyx"])]

setup(name='Fast trivariate pricers module',
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules)

# compile using python pricers_setup.py build_ext --inplace
