# compilation file for pricers_fast

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup( name        = 'Fast trivariate pricers module'
     , cmdclass    = {'build_ext': build_ext}
     , ext_modules = [Extension("pricers_fast", ["pricers_fast.pyx"])])
