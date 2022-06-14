# setup of the cython compilation for Quartic solution.

from distutils.core   import setup
from Cython.Distutils import build_ext
from Cython.Build     import cythonize


setup( name        = 'solution of the quartic equation (cython)'
     , cmdclass    = {'build_ext': build_ext}
     , ext_modules = cythonize('quartic_cy.pyx', compiler_directives={'language_level': '3'}) )

# compile using Makefile
