from distutils.core      import setup
from distutils.extension import Extension
from Cython.Distutils    import build_ext

ext_modules = [ Extension('pricers_fast', ['pricers_fast.pyx'])
              , Extension('cond_prob', ['cond_prob.pyx'], include_dirs=['.'] ) ]

setup( name        = 'Fast transitional function comp.'
     , cmdclass    = {'build_ext': build_ext}
     , ext_modules = ext_modules )

# compile using python cond_prob_setup.py build_ext --inplace
