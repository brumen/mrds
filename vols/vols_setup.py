""" Compiling the fast volatility searches.
"""

from distutils.core      import setup
from Cython.Distutils    import build_ext
from Cython.Build        import cythonize


setup( name        = 'volatility functions fast'
     , cmdclass    = {'build_ext': build_ext}
     , ext_modules = cythonize('vols_fast.pyx', compiler_directives={'language_level': '3'}) )
