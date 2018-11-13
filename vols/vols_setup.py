from config import cython_include_dirs, cython_extra_link_args

from distutils.core      import setup
from distutils.extension import Extension
from Cython.Distutils    import build_ext


setup( name        = 'volatility functions fast'
     , cmdclass    = {'build_ext': build_ext}
     , ext_modules = [Extension( "vols_fast"
                               , sources         = ["vols_fast.pyx"]
                               , include_dirs    = cython_include_dirs
                               , extra_link_args = cython_extra_link_args) ] )
