import config
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("mrds_fast", ["mrds_fast.pyx"])]
setup(name='mrds_fast_fct',
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules)

# compile using python mrds_setup.py build_ext --inplace
