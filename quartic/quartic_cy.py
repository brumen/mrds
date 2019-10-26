from distutils.core      import setup
from distutils.extension import Extension
from Cython.Distutils    import build_ext

setup(name        = 'solution of the quartic equation (cython)',
      cmdclass    = {'build_ext': build_ext},
      ext_modules = [Extension("quartic_cy", ["quartic_cy.pyx"])] )

# compile using python quartic_cy.py build_ext --inplace
