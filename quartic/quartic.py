# based on geant code 
# comapre with quartic.cu
# write unittests
# 

import config
import numpy as np

import pycuda.autoinit
import pycuda.gpuarray as gpa

j = complex (0.,1.)


# quadratic function solution on cuda 
# pols ... matrix of Nx3 
# result is in a form Nx5 matrix, the roots are on 1,2 places 
def quadr_cuda(pols):
    nb_pols = np.shape(pols)[0]
    rts = np.zeros((nb_pols, 3, 5))

    coef_mat = gpa.to_gpu ( pols ).astype(float32)
    sol_mat  = gpa.to_gpu ( rts ).astype(float32)
    quartic_string = open ("quartic.cu").read()
    quad_sol = config.SourceModule(quartic_string).get_function ("comp_quadr")
    quad_sol (coef_mat, sol_mat, block=(1,1,1),grid=(nb_pols,1) ) # TO IMPROVE TO IMPROVE 

    return sol_mat.get()

# cubic on cuda 
# pols ... matrix of Nx5 
# result is in a form Nx5 matrix, the roots are on 1,2 places 
def cubic_cuda(pols):
    nb_pols = shape (pols)[0]
    rts = zeros ( (nb_pols,3,5) )

    coef_mat = config.gpuarray.to_gpu ( pols ).astype(float32)
    sol_mat = config.gpuarray.to_gpu ( rts ).astype(float32)
    quartic_string = open ("quartic.cu").read()
    quad_sol = config.SourceModule(quartic_string).get_function ("comp_cubic")
    quad_sol (coef_mat, sol_mat, block=(1,1,1),grid=(nb_pols,1) ) # TO IMPROVE TO IMPROVE 

    return sol_mat.get()

# quartic on cuda 
# pols ... matrix of Nx5 
# result is in a form Nx5 matrix, the roots are on 1,2,3,4 places 
def quartic_cuda(pols):
    nb_pols = shape (pols)[0]
    rts = zeros ( (nb_pols,3,5) )

    coef_mat = config.gpuarray.to_gpu ( pols ).astype(float32)
    sol_mat = config.gpuarray.to_gpu ( rts ).astype(float32)
    quartic_string = open ("quartic.cu").read()
    quad_sol = config.SourceModule(quartic_string).get_function ("comp_quartic")
    quad_sol (coef_mat, sol_mat, block=(1,1,1),grid=(nb_pols,1) ) # TO IMPROVE TO IMPROVE 

    return sol_mat.get()


def QuadRoots(p):
    """ Solves for the roots of poly p[0] x^2 + p[1] x+ p[2]=0. p[0] != 0
    """

    b  = -p[1]/p[0]/2.
    c  =  p[2]/p[0]
    d2 =  b*b - c
    
    r = np.zeros(2, dtype = complex )

    if( d2 >= 0. ):
        d         = np.sqrt(d2)
        r[0] = b-d 
        r[1] = b+d
    else:
        d       = np.sqrt(-d2)
        r[0] = b + d*j 
        r[1] = b - d*j 

    return r



# 
# Roots of poly p[0] x^3 + p[1] x^2...+p[3]=0
# Assumes 0<arctan(x)<pi/2 for x>0 - CHECK THIS CHECK THIS CHECK THIS 
#
# p[0] (leading term) _HAS_ to be 1.
def CubicRoots( p ):

    r = zeros (3, dtype = complex )
    x = p[1]/3.0
    t = x*p[1]
    b = 0.5*( x*( t/1.5 - p[2] ) + p[3] )
    t = ( t - p[2] )/3.0
    c = t*t*t
    d = b*b - c

    if( d >= 0. ):
        d = (np.sqrt(d) + abs(b) )**(1.0/3.0)
    
        # if( d != 0. ):
        #     if( b > 0. ):
        #         b = -d
        #     else:
        #         b =  d
        #     c =  t/b

        c = c * (d == 0.) + t/b * (d != 0.)
        b = b * (d == 0.) + (-d) * ( d != 0. and b>0.) + d * (d !=0. and b <= 0.)
        d =  np.sqrt(0.75) * (b - c)
        b =  b + c
        c = -0.5 * b - x
        r[1] = c + d * j # first root 
        r[0] = c - d * j
        r[2] = b - x

    else:
        if( b == 0. ):
            d =  np.arctan(1.0)/1.5
        else:
            d =  np.arctan( np.sqrt(-d)/abs(b) )/3.0

        b =  np.sqrt(t)*2.0 if b < 0 else -2.0*np.sqrt(t)

        c = np.cos(d) * b
        t = - np.sqrt(0.75) * np.sin(d) * b - 0.5 * c
        d = -t - c - x
        c =  c - x
        t =  t - x

        if abs(c) > abs(t):
            r[2] = c
        else:
            r[2] = t
            t    = c

        if abs(d) > abs(t):
            r[1] = d
        else:
            r[1] = t
            t       = d

        r[0] = t

    return r

def BiquadRoots(p_in):
    """ Roots of poly p[0] x^4 + p[1] x^3...+p[4]=0
        p[0] (leading term) _HAS_ to be 1.
    """

    e = 0.25 * p_in[1]
    b = 2. * e
    c = b * b
    d = 0.75 * c
    b = p_in[3] + b * ( c - p_in[2] )
    a = p_in[2] - d
    c = p_in[4] + e * ( e * a - p_in[3] )
    a = a - d

    p = np.zeros(4)
    p[0] = p_in[0]
    p[1] = 0.5 * a
    p[2] = ( p[1] * p[1] - c ) * 0.25
    p[3] = b * b / (-64.0)

    r = np.zeros(4, dtype=complex)

    if( p[3] < 0. ):
        r[0], r[1], r[2] = CubicRoots(p)
        for k in range (3):
            if( r[k].imag == 0. and r[k].real > 0. ):
                d = r[k].real * 4. 
                a = a + d

                if ( a >= 0. and b >= 0.):
                    p[1] =   np.sqrt(d)
                elif ( a <= 0. and b <= 0.):
                    p[1] =   np.sqrt(d)
                else:
                    p[1] = - np.sqrt(d)

                b = 0.5*( a + b/p[1] )
                p[2]    = c/b
                r[0], r[1] = QuadRoots(p)

                r[2] = r[0]
                r[3] = r[1]

                p[1]    = -p[1]
                p[2]    =  b
                r[0], r[1] = QuadRoots(p)

                for i in range (4):
                    r[i] += - e

    if p[2] < 0.:
        b    = np.sqrt(c)
        d    = b + b - a
        p[1] = 0.
    
        if( d > 0. ):
            p[1] = np.sqrt(d)

    else:
        if( p[1] > 0.):
            b =   np.sqrt(p[2])*2.0 + p[1]
        else:
            b = - np.sqrt(p[2])*2.0 + p[1]

        if( b != 0.):
            p[1] = 0.
        else:
            for k in range (4):
                r[k] = -e


    p[2]    = c/b
    r[0], r[1] = QuadRoots(p)

    r[2] = r[0]
    r[3] = r[1]

    p[1]    = -p[1]
    p[2]    =  b
    r[0], r[1] = QuadRoots(p)

    for k in range (4):
        r[k] += - e

    return r

def QuarticRoots(p_in):
    """ General quartic equation
        p[0] ... leading term _HAS_ to be 0
        Roots of poly p[0] x^4 + p[1] x^3...+p[4]=0
    """
    a3 = p_in[1]
    a2 = p_in[2]
    a1 = p_in[3]
    a0 = p_in[4]

    r = np.zeros(4, dtype=complex)

    # resolvent cubic equation cofs:
    p = np.zeros(4)
    p[0] = p_in[0]
    p[1] = - a2
    p[2] = a1 * a3 - 4. * a0
    p[3] = 4. * a2 * a0 - a1 * a1 - a3 * a3 * a0

    r[0], r[1], r[2] = CubicRoots(p)

    # finding real roots
    reRoots = np.array([r[0], r[1], r[2] ])[ np.array([r[0], r[1], r[2] ]) == \
                                         np.array([r[0], r[1], r[2] ]).real ]
    y1 = min (reRoots) 

    R2 = 0.25 * a3 * a3 - a2 + y1
    b  = 0.25 * (4. * a3 * a2 - 8. * a1 - a3 * a3 * a3)
    c  = 0.75 * a3 * a3 - 2. * a2
    a  = c - R2
    d  = 4. * y1 * y1 - 16. * a0


    if( R2 > 0.):
        R = np.sqrt(R2)
        D2 = a + b/R
        E2 = a - b/R

        if( D2 >= 0. ):
            D       = np.sqrt(D2)
            r[0] = -0.25*a3 + 0.5*R + 0.5*D
            r[1] = -0.25*a3 + 0.5*R - 0.5*D
        else:
            D       = np.sqrt(-D2)
            r[0] = -0.25 * a3 + 0.5 * R + j * 0.5 * D
            r[1] = -0.25 * a3 + 0.5 * R - j * 0.5 * D

        if( E2 >= 0. ):
            E       = np.sqrt(E2)
            r[2] = -0.25 * a3 - 0.5 * R + 0.5 * E
            r[3] = -0.25 * a3 - 0.5 * R - 0.5 * E
        else:
            E       = np.sqrt(-E2)
            r[2] = -0.25 * a3 - 0.5 * R + j * 0.5 * E
            r[3] = -0.25 * a3 - 0.5 * R - j * 0.5 * E

    elif ( R2 < 0.):
        R = np.sqrt(-R2)

        CD2_abs_sqrt = np.sqrt(np.sqrt( a*a + b*b/R/R))
        CD2_theta = arctan ( - b / R / a )
        real_CD = CD2_abs_sqrt * cos ( CD2_theta/2.) 
        imag_CD = CD2_abs_sqrt * sin ( CD2_theta/2.)

        r[0] = -0.25 * a3 + 0.5 * real_CD + j * (0.5 * R + 0.5 * imag_CD)
        r[1] = -0.25 * a3 - 0.5 * real_CD + j * (0.5 * R - 0.5 * imag_CD)

        real_CE = CD2_abs_sqrt * cos ( CD2_theta/2.)
        imag_CE = CD2_abs_sqrt * sin ( CD2_theta/2.)

        r[2] = -0.25 * a3 + 0.5 * real_CE + j * ( -0.5 * R + 0.5 * imag_CE )
        r[3] = -0.25 * a3 - 0.5 * real_CE + j * ( -0.5 * R - 0.5 * imag_CE )

    else:  # R2=0 case

        if(d >= 0.):
            D2 = c + np.sqrt(d)
            E2 = c - np.sqrt(d)

            if( D2 >= 0. ):
                D       = np.sqrt(D2)
                r[0] = -0.25 * a3 + 0.5 * R + 0.5 * D
                r[1] = -0.25 * a3 + 0.5 * R - 0.5 * D
            else:
                D       = np.sqrt(-D2)
                r[0] = -0.25*a3 + 0.5*R + j * 0.5 * D
                r[1] = -0.25*a3 + 0.5*R - j * 0.5 * D

            if( E2 >= 0. ):
                E       = np.sqrt(E2)
                r[2] = -0.25 * a3 - 0.5 * R + 0.5 * E
                r[3] = -0.25 * a3 - 0.5 * R - 0.5 * E
            else:
                E       = np.sqrt(-E2)
                r[2] = -0.25*a3 - 0.5*R + j * 0.5 * E
                r[3] = -0.25*a3 - 0.5*R - j * 0.5 * E

        else:
            ds = np.sqrt(-d)

            CD2_abs_sqrt = np.sqrt(np.sqrt( c*c + ds*ds))
            CE2_theta = arctan ( ds / c)
            real_CD = CD2_abs_sqrt * cos ( CE2_theta/2.)
            imag_CD = CD2_abs_sqrt * sin ( CE2_theta/2.)

            r[0] = -0.25 * a3 + 0.5 * real_CD + j * (0.5 * R + 0.5 * imag_CD)
            r[1] = -0.25 * a3 - 0.5 * real_CD + j * (0.5 * R - 0.5 * imag_CD)

            CE2_theta = arctan ( - ds / c)
            real_CE = CD2_abs_sqrt * cos ( CE2_theta/2.)
            imag_CE = CD2_abs_sqrt * sin ( CE2_theta/2.)

            r[2] = -0.25*a3 + 0.5*real_CE + j * ( -0.5*R + 0.5*imag_CE )
            r[3] = -0.25*a3 - 0.5*real_CE + j * ( -0.5*R - 0.5*imag_CE )

    return r
