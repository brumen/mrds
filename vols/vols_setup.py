import config
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("vols_fast", ["vols_fast.pyx"],
                         include_dirs=config.cython_include_dirs,
                         extra_link_args=config.cython_extra_link_args)]
    

setup(name = 'volatility functions fast',
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules
    )
