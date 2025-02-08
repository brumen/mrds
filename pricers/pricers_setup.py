# compilation file for pricers_fast

from setuptools import setup
from Cython.Distutils import build_ext
from Cython.Build import cythonize


setup(
    name='Fast trivariate pricers module',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(
        'pricers_fast.pyx',
        compiler_directives={'language_level': '3'}
    )
)
