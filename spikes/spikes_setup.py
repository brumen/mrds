import config
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

if config.OS == 'MS':
    ext_modules = [Extension("spikes_fast", ["spikes_fast.pyx"], \
                             include_dirs=[ '\\\\ms\\dist\\python\\PROJ\\numpy\\1.4.1-py26\\lib\\numpy\\core\\include\\']) \
                   ]
    # '\\\\ms\dist\\python\\PROJ\\core\\2.6.4\\include\\python2.6\\', <- THIS DOESNT WORK BECAUSE in pyconfig.h _HYPOT should be set to 0
else:
    ext_modules = [Extension("spikes_fast", ["spikes_fast.pyx"])]
    

setup( \
    name = 'Spikes fast procedures', \
    cmdclass = {'build_ext': build_ext}, \
    ext_modules = ext_modules \
    )


# compile using python spikes_setup.py build_ext --inplace
