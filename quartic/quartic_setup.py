# setup of the cython compilation for Quartic solution.

from distutils.core      import setup
from distutils.extension import Extension
from Cython.Distutils    import build_ext

from config import cython_include_dirs, cython_extra_link_args


setup( name        = 'solution of the quartic equation (cython)'
     , cmdclass    = {'build_ext': build_ext}
     , ext_modules = [Extension( 'quartic_cy'
                               , ['quartic_cy.pyx']
                               , include_dirs    = cython_include_dirs
                               , extra_link_args = cython_extra_link_args ) ] )

# compile using python3 quartic_cy.py build_ext --inplace
