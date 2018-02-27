# based on geant code 

cimport cython
import numpy as np
cimport numpy as np

# declarations of external functions 
cdef extern from "math.h" nogil:
    double sqrt(double)
cdef extern from "math.h" nogil:
    double atan(double)
cdef extern from "math.h" nogil:
    double atan2(double, double)
cdef extern from "math.h" nogil:
    double cos(double)
cdef extern from "math.h" nogil:
    double sin(double)
cdef extern from "math.h" nogil:
    double fabs(double)

DTYPE_POLY = np.double  # type of poly. coeff entered
ctypedef np.double_t DTYPE_POLY_t
DTYPE_ROOTS = np.complex
ctypedef np.complex_t DTYPE_ROOTS_t
DEF FLT_MAX = 3.40282347e38  # artificial constant



@cython.boundscheck(False)
def QuadRoots(np.ndarray[DTYPE_POLY_t, ndim=1] p):
    """
    roots of poly p[0] x^2 + p[1] x + p[2] = 0
    output: array of complex numbers
    """
    cdef double b, c, d2, d

    b  = -p[1]/p[0]/2.
    c  =  p[2]/p[0]
    d2 =  b*b - c

    cdef np.ndarray[DTYPE_ROOTS_t, ndim=1] r = np.zeros(2, dtype=DTYPE_ROOTS)

    if (d2 >= 0.):
        d         = sqrt(d2)
        r[0].real = b-d 
        r[1].real = b+d
    else:
        d       = sqrt(-d2);
        r[0].real = b #+ d * j
        r[0].imag = d
        r[1].real = b
        r[1].imag = -d

    return r


@cython.boundscheck(False)
def CubicRoots(np.ndarray[DTYPE_POLY_t, ndim=1] p_raw):
    """
    # Roots of poly p[0] x^3 + p[1] x^2...+p[3]=0
    # x=r[1][k] + i r[2][k]  k=1,...,3
    # Assumes 0<arctan(x)<pi/2 for x>0
    """

    cdef np.ndarray[DTYPE_ROOTS_t, ndim=1] r = np.zeros(3, dtype=DTYPE_ROOTS)
    cdef np.ndarray[DTYPE_POLY_t, ndim=1] p = np.zeros(4, dtype=DTYPE_POLY)

    cdef double x, t, b, c, d
    cdef int k

    # normalization
    p[1] = p_raw[1]/p_raw[0]
    p[2] = p_raw[2]/p_raw[0]
    p[3] = p_raw[3]/p_raw[0]
    p[0] = 1.

    x = p[1]/3.0
    t = x*p[1]
    b = 0.5*(x*(t/1.5 - p[2]) + p[3])
    t = (t - p[2])/3.0
    c = t*t*t
    d = b*b - c

    if d >= 0.:
        d = (sqrt(d) + fabs(b))**(1.0/3.0)
        if d != 0.:
            if b > 0.:
                b = -d
            else:
                b = d
            c =  t/b

        d = sqrt(0.75)*(b - c)
        b += c
        c = -0.5*b - x

        r[1].real = c
        r[1].imag = d

        if ((b > 0. and  x <= 0.) or (b < 0. and x > 0.)):
            r[0].real = c
            r[0].imag = -d
            r[2].real = b - x
            #r[2][3] =  0; already set to 0
        else:
            r[0].real = b - x
            # r[2][1] =  0.
            r[2].real = c
            r[2].imag = -d
    # end of 2 equal or complex roots 
    else:
        if b == 0.:
            d = atan(1.0)/1.5
        else:
            d = atan(sqrt(-d)/fabs(b))/3.0
        if b < 0.:
            b = sqrt(t)*2.
        else:
            b = -2.*sqrt(t)

        c = cos(d)*b
        t = -sqrt(0.75) * sin(d)*b - 0.5*c
        d = -t - c - x
        c -= x
        t -= x

        if fabs(c) > fabs(t):
            r[2].real = c
        else:
            r[2].real = t
            t = c

        if fabs(d) > fabs(t):
            r[1].real = d
        else:
            r[1].real = t
            t = d

        r[0].real = t

    return r


@cython.boundscheck(False)
def BiquadRoots(np.ndarray[DTYPE_POLY_t, ndim=1] p_raw):
    """
    # roots of poly p[0] x^4 + p[1] x^3...+p[4]=0
    """
    cdef np.ndarray[DTYPE_ROOTS_t, ndim=1] r = np.zeros(4, dtype=DTYPE_ROOTS)
    cdef np.ndarray[DTYPE_POLY_t, ndim=1] p = np.zeros(5, dtype=DTYPE_POLY)

    cdef double a, b, c, d, e
    cdef int i, k, j

    for k in range(1, 5):
        p[k] = p_raw[k]/p_raw[0]
    p[0] = 1.

    e = 0.25*p[1]
    b = 2*e
    c = b*b
    d = 0.75*c
    b = p[3] + b*( c - p[2] )
    a = p[2] - d
    c = p[4] + e*( e*a - p[3] )
    a = a - d

    p[1] = 0.5*a
    p[2] = (p[1]*p[1]-c)*0.25
    p[3] = b*b/(-64.0)

    if p[3] < 0.:
        r = CubicRoots(p)
        for k in range(3):
            if (r[k].imag == 0. and r[k].real > 0):
                d = r[k].real * 4.
                a = a + d

                if (a >= 0. and b >= 0.):
                    p[1] = sqrt(d)
                elif (a <= 0. and b <= 0.):
                    p[1] = sqrt(d)
                else:
                    p[1] = - sqrt(d)

                b = 0.5 * (a + b/p[1])
                p[2] = c/b
                r = QuadRoots(p)

                for i in range(2):
                    r[i+2].real = r[i].real
                    r[i+2].imag = r[i].imag

                p[1] = -p[1]
                p[2] = b
                r[0:2] = QuadRoots(p)

                for i in range(4):
                    r[i].real = r[i].real - e

    if p[2] < 0.:
        b = sqrt(c)
        d = b + b - a
        p[1] = 0.
    
        if d > 0.:
            p[1] = sqrt(d)

    else:
        if p[1] > 0.:
            b = sqrt(p[2])*2.0 + p[1]
        else:
            b = - sqrt(p[2])*2.0 + p[1]

        if b != 0.:
            p[1] = 0
        else:
            for k in range(4):
                r[k].real = -e
                r[k].imag =  0.
                
    p[2] = c/b
    r = QuadRoots(p)

    for k in range(2):
        r[k+2].real = r[k].real
        r[k+2].imag = r[k].imag

    p[1] = -p[1]
    p[2] = b
    r[0:2] = QuadRoots(p)

    for k in range(4):
        r[k].real -= e

    return r


@cython.boundscheck(False)
def QuarticRoots(np.ndarray[DTYPE_POLY_t, ndim=1] p_raw):
    
    cdef double a0, a1, a2, a3, y1
    cdef double R2, D2, E2, D, E, R = 0.
    cdef double a, b, c, d, ds

    cdef np.ndarray[DTYPE_ROOTS_t, ndim=1] r = np.zeros(4, dtype=DTYPE_ROOTS)
    cdef np.ndarray[DTYPE_POLY_t, ndim=1] p = np.zeros(5, dtype=DTYPE_POLY)
    cdef np.ndarray[DTYPE_POLY_t, ndim=1] reRoot = np.zeros(3, dtype=DTYPE_POLY)
    cdef int k, noReRoots = 0

    # if (fabs(p_raw[0]) < 1e-8:
    # this part is handled in mrds.py itself, as the calibration does not work well with small p_raw

    for k in range(3):
        reRoot[k] = FLT_MAX  # maximum double var.

    # normalization
    for k in range(1, 5):
        p[k] = p_raw[k]/p_raw[0]
    p[0] = 1.

    a3 = p[1]
    a2 = p[2]
    a1 = p[3]
    a0 = p[4]

    # resolvent cubic equation cofs:
    p[1] = -a2
    p[2] = a1 * a3 - 4. * a0
    p[3] = 4. * a2 * a0 - a1 * a1 - a3 * a3 * a0

    r[:3] = CubicRoots(p[:3])

    for k in range(3):
        if r[k].imag == 0.:  # find a real root
            noReRoots += 1
            reRoot[k] = r[k].real
        else:
            reRoot[k] = FLT_MAX

    y1 = FLT_MAX
    for k in range(3):
        if reRoot[k] < y1:
            y1 = reRoot[k]

    R2 = 0.25 * a3 * a3 - a2 + y1
    b  = 0.25 * (4. * a3 * a2 - 8. * a1 - a3 * a3 * a3)
    c  = 0.75 * a3 * a3 - 2. * a2
    a  = c - R2
    d  = 4. * y1 * y1 - 16. * a0

    # declaration of variables for later
    cdef double CD2_abs_sqrt, CD2_theta, real_CD, imag_CD, \
         CE2_theta, real_CE, imag_CE

    if R2 > 0.:
        R = sqrt(R2)
        D2 = a + b/R
        E2 = a - b/R
        if D2 >= 0.:
            D = sqrt(D2)
            r[0].real = -0.25*a3 + 0.5*R + 0.5*D
            r[1].real = -0.25*a3 + 0.5*R - 0.5*D
            r[0].imag = 0.
            r[1].imag = 0.
        else:
            D = sqrt(-D2)
            r[0].real = -0.25 * a3 + 0.5 * R
            r[1].real = -0.25 * a3 + 0.5 * R
            r[0].imag = 0.5 * D
            r[1].imag = -0.5 * D

        if E2 >= 0.:
            E = sqrt(E2)
            r[2].real = -0.25 * a3 - 0.5 * R + 0.5 * E
            r[3].real = -0.25 * a3 - 0.5 * R - 0.5 * E
            r[2].imag = 0.
            r[3].imag = 0.
        else:
            E = sqrt(-E2)
            r[2].real = -0.25 * a3 - 0.5 * R
            r[3].real = -0.25 * a3 - 0.5 * R
            r[2].imag = 0.5 * E
            r[3].imag = -0.5 * E
    elif R2 < 0.:
        R = sqrt(-R2)
        CD2_abs_sqrt = (a*a + b*b/R/R)**(1.0/4.0 )
        CD2_theta = atan2(-b/R, a)
        real_CD = CD2_abs_sqrt * cos(CD2_theta/2.)
        imag_CD = CD2_abs_sqrt * sin(CD2_theta/2.)

        r[0].real = -0.25*a3 + 0.5*real_CD
        r[1].real = -0.25*a3 - 0.5*real_CD
        r[0].imag = 0.5*R + 0.5*imag_CD
        r[1].imag = 0.5*R - 0.5*imag_CD

        CE2_theta = atan2(b/R, a)
        real_CE = CD2_abs_sqrt * cos(CE2_theta/2.)
        imag_CE = CD2_abs_sqrt * sin(CE2_theta/2.)

        r[2].real = -0.25 * a3 + 0.5 * real_CE
        r[3].real = -0.25 * a3 - 0.5 * real_CE
        r[2].imag = -0.5 * R + 0.5 * imag_CE
        r[3].imag = -0.5 * R - 0.5 * imag_CE
    else:  # R2=0 case
        if d >= 0.:
            D2 = c + sqrt(d)
            E2 = c - sqrt(d)
            if D2 >= 0.:
                D = sqrt(D2)
                r[0].real = -0.25 * a3 + 0.5 * R + 0.5 * D
                r[1].real = -0.25 * a3 + 0.5 * R - 0.5 * D
                r[0].imag = 0.
                r[1].imag = 0.
            else:
                D = sqrt(-D2)
                r[0].real = -0.25*a3 + 0.5*R
                r[1].real = -0.25*a3 + 0.5*R
                r[0].imag = 0.5*D
                r[1].imag = -0.5*D
            if E2 >= 0.:
                E = sqrt(E2)
                r[2].real = -0.25 * a3 - 0.5 * R + 0.5 * E
                r[3].real = -0.25 * a3 - 0.5 * R - 0.5 * E
                r[2].imag = 0.
                r[3].imag = 0.
            else:
                E = sqrt(-E2)
                r[2].real = -0.25*a3 - 0.5*R
                r[3].real = -0.25*a3 - 0.5*R
                r[2].imag = 0.5*E
                r[3].imag = -0.5*E
        else:
            ds = sqrt(-d)
            CD2_abs_sqrt = (c*c + ds*ds)**(1.0/4.0 )
            CE2_theta = atan2(ds, c)
            real_CD = CD2_abs_sqrt * cos(CE2_theta/2.)
            imag_CD = CD2_abs_sqrt * sin(CE2_theta/2.)

            r[0].real = -0.25 * a3 + 0.5 * real_CD
            r[1].real = -0.25 * a3 - 0.5 * real_CD
            r[0].imag = 0.5 * R + 0.5 * imag_CD
            r[1].imag = 0.5 * R - 0.5 * imag_CD

            CE2_theta = atan2(-ds, c)
            real_CE = CD2_abs_sqrt * cos(CE2_theta/2.)
            imag_CE = CD2_abs_sqrt * sin(CE2_theta/2.)

            r[2].real = -0.25*a3 + 0.5*real_CE
            r[3].real = -0.25*a3 - 0.5*real_CE
            r[2].imag = -0.5*R + 0.5*imag_CE
            r[3].imag = -0.5*R - 0.5*imag_CE

    return r
