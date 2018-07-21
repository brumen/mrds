# compile script for the mrds_fast module using python3
# 
# compile using python3 mrds_setup.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("mrds_fast", ["mrds_fast.pyx"])]
setup(name='mrds_fast_fct',
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules)
