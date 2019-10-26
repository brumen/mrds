/*
** quartic.c
** 
**
** Started on  Sun Jul  3 11:07:10 2011 brumen
** Last update Sun May 12 01:17:25 2002 Speed Blue
*/

//#include "quartic.h"

//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
// $Id: G4AnalyticalPolSolver.cc,v 1.7 2007-11-13 17:35:06 gcosmo Exp $
// GEANT4 tag $Name: geant4-09-04-patch-02 $
//

//#include <stdio.h>
//#include <math.h> 


#define FLT_MAX 3.40282347e+38F


/* int main() { */
/*   return 0; */
/* } */



//////////////////////////////////////////////////////////////////////////////
//
// Array r[3][5]  p[5]
// Roots of poly p[0] x^2 + p[1] x+p[2]=0
//
// x = r[1][k] + i r[2][k];  k = 1, 2

__device__ int QuadRoots( float p[3], float r[3][5] ) {
  float b, c, d2, d;

  b  = -p[1]/p[0]/2.;
  c  =  p[2]/p[0];
  d2 =  b*b - c;
  
  if( d2 >= 0. )
  {
    d       = sqrt(d2);
    r[1][1] = b - d;   
    r[1][2] = b + d;   
    r[2][1] = 0.; 
    r[2][2] = 0.;
  }
  else
  {
    d       = sqrt(-d2);
    r[2][1] =  d; 
    r[2][2] = -d;
    r[1][1] =  b; 
    r[1][2] =  b;
  }

  return 2; 
}

/* computing quadroots for a bunch of parameters */
__global__ void comp_quadr (float (*p_mat)[3], float (*r_mat)[3][5]) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x ;
  
  if (idx < 1000) // THIS NEEDS TO CHANGE  TO COMPLETE TO COMPLETE TO COMPLETE 
    QuadRoots (*(p_mat + idx), *(r_mat + idx) );
}


//////////////////////////////////////////////////////////////////////////////
//
// Array r[3][5]  p[5]
// Roots of poly p[0] x^3 + p[1] x^2...+p[3]=0
// x=r[1][k] + i r[2][k]  k=1,...,3
// Assumes 0<arctan(x)<pi/2 for x>0
__device__ int CubicRoots( float p[5], float r[3][5] ) {
  float x,t,b,c,d;

  /* if( p[0] != 1. ) */
  /* { */
  /*   //for(k = 1; k < 4; k++ ) { p[k] = p[k]/p[0]; } */
  /*   p[1] = p[1]/p[0]; */
  /*   p[2] = p[2]/p[0]; */
  /*   p[3] = p[3]/p[0]; */
  /*   p[0] = 1.; */
  /* } */
  // normalize this the p[0] to 1
  p[1] = p[1]/p[0];
  p[2] = p[2]/p[0];
  p[3] = p[3]/p[0];
  p[0] = 1.;
  
  x = p[1]/3.0; 
  t = x*p[1];
  b = 0.5*( x*( t/1.5 - p[2] ) + p[3] ); 
  t = ( t - p[2] )/3.0;
  c = t*t*t; 
  d = b*b - c;

  if( d >= 0. ) {
    d = cbrtf(sqrt(d) + fabs(b)); /* x^(1/3) */
    
    if( d != 0. ) {
      if( b > 0. ) 
	b = -d;
      else 
	b =  d;
      c =  t/b;
    }
    d       =  sqrt(0.75)*(b - c); 
    r[2][2] =  d; 
    b       =  b + c;
    c       = -0.5*b-x;
    r[1][2] =  c;

    if ( ( b > 0. &&  x <= 0. ) || ( b < 0. && x > 0. ) ) {
       r[1][1] =  c; 
       r[2][1] = -d; 
       r[1][3] =  b - x;
       r[2][3] =  0;
    } else {
       r[1][1] =  b - x; 
       r[2][1] =  0.; 
       r[1][3] =  c;
       r[2][3] = -d;
    }
  } // end of 2 equal or complex roots 
  else {
    if( b == 0. ) 
      d =  atan(1.0)/1.5;
    else 
      d =  atan( sqrt(-d)/fabs(b) )/3.0;

    if ( b < 0. )  
      b =  sqrt(t)*2.0;
    else 
      b = -2.0*sqrt(t);

    c =  cos(d)*b; 
    t = - sqrt(0.75)* sin(d)*b - 0.5*c;
    d = -t - c - x; 
    c =  c - x; 
    t =  t - x;

    if( fabs(c) > fabs(t) ) 
      r[1][3] = c;
    else {
      r[1][3] = t; 
      t       = c;
    }
    if( fabs(d) > fabs(t) ) 
      r[1][2] = d;
    else {
      r[1][2] = t; 
      t       = d;
    }
    r[1][1] = t;

    // for(k = 1; k < 4; k++ ) { r[2][k] = 0.; }
    r[2][1] = 0.;
    r[2][2] = 0.;
    r[2][3] = 0.;
  }
  return 0;
}

/* computing quadroots for a bunch of parameters */
__global__ void comp_cubic (float (*p_mat)[5], float (*r_mat)[3][5]) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x ;
  
  if (idx < 1000) // THIS NEEDS TO CHANGE TO COMPLETE TO COMPLETE TO COMPLETE 
    CubicRoots (*(p_mat + idx), *(r_mat + idx) );
}


//////////////////////////////////////////////////////////////////////////////
//
// Array r[3][5]  p[5]
// Roots of poly p[0] x^4 + p[1] x^3...+p[4]=0
// x=r[1][k] + i r[2][k]  k=1,...,4

__device__ int BiquadRoots( float p[5], float r[3][5] )
{
  float a, b, c, d, e;
  int i, k, j;

  if(p[0] != 1.0)
  {
    //for( k = 1; k < 5; k++) { p[k] = p[k]/p[0]; }
    p[1] = p[1]/p[0];
    p[2] = p[2]/p[0];
    p[3] = p[3]/p[0];
    p[4] = p[4]/p[0];
    p[0] = 1.;
  }
  e = 0.25*p[1];
  b = 2*e;
  c = b*b;
  d = 0.75*c;
  b = p[3] + b*( c - p[2] );
  a = p[2] - d;
  c = p[4] + e*( e*a - p[3] );
  a = a - d;

  p[1] = 0.5*a;
  p[2] = (p[1]*p[1]-c)*0.25;
  p[3] = b*b/(-64.0);

  if( p[3] < 0. )
  {
    CubicRoots(p,r);

    for( k = 1; k < 4; k++ )
    {
      if( r[2][k] == 0. && r[1][k] > 0 )
      {
        d = r[1][k]*4; 
        a = a + d;

        if     ( a >= 0. && b >= 0.) 
	  p[1] =   sqrt(d);
        else if ( a <= 0. && b <= 0.) 
	  p[1] =   sqrt(d);
        else 
	  p[1] = - sqrt(d);

        b = 0.5*( a + b/p[1] );

        p[2]    = c/b; 
        QuadRoots(p,r);

        for( i = 1; i < 3; i++ ) 
          for( j = 1; j < 3; j++ ) 
	    r[j][i+2] = r[j][i];

        p[1]    = -p[1]; 
        p[2]    =  b; 
        QuadRoots(p,r);

        //for( i = 1; i < 5; i++ ) { r[1][i] = r[1][i] - e; }
	r[1][1] = r[1][1] - e;
	r[1][2] = r[1][2] - e;
	r[1][3] = r[1][3] - e;
	r[1][4] = r[1][4] - e;
      
        return 4;
      }
    }
  }
  if( p[2] < 0. )
  {
    b    = sqrt(c); 
    d    = b + b - a;
    p[1] = 0.; 
    
    if( d > 0. ) 
      p[1] = sqrt(d);
  }
  else
  {
    if( p[1] > 0.) 
      b =   sqrt(p[2])*2.0 + p[1];
    else 
      b = - sqrt(p[2])*2.0 + p[1];

    if( b != 0.) 
      p[1] = 0;
    else
    {
      for(k = 1; k < 5; k++ )
      {
          r[1][k] = -e;
          r[2][k] =  0;
      }
      return 0;
    }
  }

  p[2]    = c/b; 
  QuadRoots(p,r);

  for( k = 1; k < 3; k++ )
    for( j = 1; j < 3; j++ ) 
      r[j][k+2] = r[j][k];

  p[1]    = -p[1]; 
  p[2]    =  b; 
  QuadRoots(p,r);

  //for( k = 1; k < 5; k++ ) { r[1][k] = r[1][k] - e; }
  r[1][1] = r[1][1] - e;
  r[1][2] = r[1][2] - e;
  r[1][3] = r[1][3] - e;
  r[1][4] = r[1][4] - e;

  return 4;
}

//////////////////////////////////////////////////////////////////////////////
// solution in [1][k=1..4]  real part
//             [2][k=1..4]  imag part
__device__ int QuarticRoots( float p[5], float r[3][5])
{
  float a0, a1, a2, a3, y1;
  float R2, D2, E2, D, E, R = 0.;
  float a, b, c, d, ds;

  float reRoot[4];
  int k, noReRoots = 0;
  
  /*for( k = 0; k < 4; k++ ) { reRoot[k] = FLT_MAX; }*/ /* DBL_MAX
						       std::numeric_limits<double>::max()
						       1.7976931348623157e+308 
						       CAN WE REPLACE
						       HERE WITH
						       FLT_MAX : 3.40282347e+38F
						    */
  reRoot[0] = FLT_MAX;
  reRoot[1] = FLT_MAX;
  reRoot[2] = FLT_MAX;
  reRoot[3] = FLT_MAX;

  if( p[0] != 1.0 )
  {
    for( k = 1; k < 5; k++) { p[k] = p[k]/p[0]; }
    p[0] = 1.;
  }
  a3 = p[1];
  a2 = p[2];
  a1 = p[3];
  a0 = p[4];

  // resolvent cubic equation cofs:

  p[1] = -a2;
  p[2] = a1*a3 - 4*a0;
  p[3] = 4*a2*a0 - a1*a1 - a3*a3*a0;

  CubicRoots(p,r);

  for( k = 1; k < 4; k++ )
  {
    if( r[2][k] == 0. ) // find a real root
    {
      noReRoots++;
      reRoot[k] = r[1][k];
    }
    else reRoot[k] = FLT_MAX; // kInfinity;  CORRECT CORRECT
			      // CORRECT 
  }
  y1 = FLT_MAX; // kInfinity;  CORRECT CORRECT CORRECT
  for( k = 1; k < 4; k++ )
  {
    if ( reRoot[k] < y1 ) { y1 = reRoot[k]; }
  }

  R2 = 0.25*a3*a3 - a2 + y1;
  b  = 0.25*(4*a3*a2 - 8*a1 - a3*a3*a3);
  c  = 0.75*a3*a3 - 2*a2;
  a  = c - R2;
  d  = 4*y1*y1 - 16*a0;

  // declaration of variables for later
  float CD2_abs_sqrt, CD2_theta, real_CD, imag_CD, CE2_theta, real_CE, imag_CE;

  if( R2 > 0.)
  {
    R = sqrt(R2);
    D2 = a + b/R;
    E2 = a - b/R;

    if( D2 >= 0. )
    {
      D       = sqrt(D2);
      r[1][1] = -0.25*a3 + 0.5*R + 0.5*D;
      r[1][2] = -0.25*a3 + 0.5*R - 0.5*D;
      r[2][1] = 0.;
      r[2][2] = 0.;
    }
    else
    {
      D       = sqrt(-D2);
      r[1][1] = -0.25*a3 + 0.5*R;
      r[1][2] = -0.25*a3 + 0.5*R;
      r[2][1] =  0.5*D;
      r[2][2] = -0.5*D;
    }
    if( E2 >= 0. )
    {
      E       = sqrt(E2);
      r[1][3] = -0.25*a3 - 0.5*R + 0.5*E;
      r[1][4] = -0.25*a3 - 0.5*R - 0.5*E;
      r[2][3] = 0.;
      r[2][4] = 0.;
    }
    else
    {
      E       = sqrt(-E2);
      r[1][3] = -0.25*a3 - 0.5*R;
      r[1][4] = -0.25*a3 - 0.5*R;
      r[2][3] =  0.5*E;
      r[2][4] = -0.5*E;
    }
  }
  else if( R2 < 0.)
  {
    R = sqrt(-R2);
    //G4complex CD2(a,-b/R); 
    //G4complex CD = sqrt(CD2);
    CD2_abs_sqrt = sqrtf (sqrtf ( a*a + b*b/R/R ) ); /* x^1/4 */
    CD2_theta = atan ( - b / R / a );
    real_CD = CD2_abs_sqrt * cos ( CD2_theta/2.);
    imag_CD = CD2_abs_sqrt * sin ( CD2_theta/2.);



    r[1][1] = -0.25*a3 + 0.5*real_CD;
    r[1][2] = -0.25*a3 - 0.5*real_CD;
    r[2][1] =  0.5*R + 0.5*imag_CD;
    r[2][2] =  0.5*R - 0.5*imag_CD;
    //G4complex CE2(a,b/R);  
    //G4complex CE = std::sqrt(CE2); 
    CE2_theta = atan ( - b/ R/ a );
    real_CE = CD2_abs_sqrt * cos ( CE2_theta/2.);
    imag_CE = CD2_abs_sqrt * sin ( CE2_theta/2.);


    r[1][3] = -0.25*a3 + 0.5*real_CE;
    r[1][4] = -0.25*a3 - 0.5*real_CE;
    r[2][3] =  -0.5*R + 0.5*imag_CE;
    r[2][4] =  -0.5*R - 0.5*imag_CE;
  }
  else // R2=0 case
  {
    if(d >= 0.)
    {
      D2 = c + sqrt(d);
      E2 = c - sqrt(d);

      if( D2 >= 0. )
      {
        D       = sqrt(D2);
        r[1][1] = -0.25*a3 + 0.5*R + 0.5*D;
        r[1][2] = -0.25*a3 + 0.5*R - 0.5*D;
        r[2][1] = 0.;
        r[2][2] = 0.;
      }
      else
      {
        D       = sqrt(-D2);
        r[1][1] = -0.25*a3 + 0.5*R;
        r[1][2] = -0.25*a3 + 0.5*R;
        r[2][1] =  0.5*D;
        r[2][2] = -0.5*D;
      }
      if( E2 >= 0. )
      {
        E       = sqrt(E2);
        r[1][3] = -0.25*a3 - 0.5*R + 0.5*E;
        r[1][4] = -0.25*a3 - 0.5*R - 0.5*E;
        r[2][3] = 0.;
        r[2][4] = 0.;
      }
      else
      {
        E       = sqrt(-E2);
        r[1][3] = -0.25*a3 - 0.5*R;
        r[1][4] = -0.25*a3 - 0.5*R;
        r[2][3] =  0.5*E;
        r[2][4] = -0.5*E;
      }
    }
    else
    {
      ds = sqrt(-d);
      //G4complex CD2(c,ds); /* CORRECT CORRECT CORRECT */
      //G4complex CD = sqrt(CD2);
      CD2_abs_sqrt = sqrtf (sqrtf ( c*c + ds*ds ) );
      CE2_theta = atan ( ds / c);
      real_CD = CD2_abs_sqrt * cos ( CE2_theta/2.);
      imag_CD = CD2_abs_sqrt * sin ( CE2_theta/2.);

      r[1][1] = -0.25*a3 + 0.5*real_CD;
      r[1][2] = -0.25*a3 - 0.5*real_CD;
      r[2][1] =  0.5*R + 0.5*imag_CD;
      r[2][2] =  0.5*R - 0.5*imag_CD;

      //G4complex CE2(c,-ds);  
      //G4complex CE = std::sqrt(CE2);
      // float CE2_abs = CD2_abs_sqrt
      CE2_theta = atan ( - ds / c);
      real_CE = CD2_abs_sqrt * cos ( CE2_theta/2.);
      imag_CE = CD2_abs_sqrt * sin ( CE2_theta/2.);

      r[1][3] = -0.25*a3 + 0.5*real_CE;
      r[1][4] = -0.25*a3 - 0.5*real_CE;
      r[2][3] =  -0.5*R + 0.5*imag_CE;
      r[2][4] =  -0.5*R - 0.5*imag_CE;
    }  
  }
  return 4;
}

/* computing quadroots for a bunch of parameters */
__global__ void comp_quartic (float (*p_mat)[5], float (*r_mat)[3][5]) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x ;
  
  if (idx < 1000) // THIS NEEDS TO CHANGE TO COMPLETE TO COMPLETE TO COMPLETE 
    CubicRoots (*(p_mat + idx), *(r_mat + idx) );
}




/* __global__ void imp_vol_quartic (float *vol_mat, float *c, float *K_v, float *ttm_v) { */

/*   int vol_idx = threadIdx.x + blockIdx.x * blockDim.x ; /\* block index, goes over simulations *\/ */

/*   float r[3][5]; */
/*   if (vol_idx < K_size * ttm_size ) { */
/*     QuarticRoots( float p[5], r); */
/*     /\* vol_mat[vol_idx] = quartic comp_imp_dev (S0, K, ttm, sigma_0,	\ */
/* 				     A, B, C, P, alphaC, alphaP); */
/*     *\/ */
/*   } */
  
/* } */

// computes the truncated E[ N^{0,1,2,3,4} * 1(N <a) ] where N std. normal
// in succession 
/* __device__ float  _trunc_normal_above(float a, float N[5]): */
/*   return array([ normcdf (a) ,  */
/*                  - exp ( - a*a / 2.0 ) / sqrt (2 * pi),		\ */
/* 		 0.5 + 0.5 * scipy.special.erf (a / sqrt (2) ) -	\ */
/* 		 exp (-a**2/2.0) * a / sqrt (2 * pi ),  */
/* 		 - (a**2 + 2) * exp (- a**2 / 2.0) / sqrt (2 * pi), */
/* 		 - a * (a**2 + 3) * exp (- a**2 / 2.0) /	\ */
/* 		 sqrt (2 * pi) +					\ */
/* 		 1.5 * (1.0 + scipy.special.erf (a / sqrt (2) ) ) ]) */

/*   // computes the truncated E[ N^{0,1,2,3,4} * 1(N >a) ] where N std. normal */
/*   // in succession  */
/*     def _trunc_normal_below(self, a): */
/*         return - self._trunc_normal_above(a) + array([1.0, 0.0, 1.0, 0.0, 3.0]) */

/*     # computes the truncated E[ N^{0,1,2,3,4} * 1(a < N <b) ] where N std. normal */
/*     # in succession  */
/*     def _trunc_normal_interval(self, a, b): */
/*         return self._trunc_normal_above(b) - self._trunc_normal_above (a) */
