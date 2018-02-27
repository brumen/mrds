import pickle

import config # very general config
from numpy import *
import numpy.random
import scipy
import scipy.optimize
import scipy.integrate
import scipy.special
import scipy.stats 
import scipy.optimize 
import scipy.interpolate # spline package 
import time # for seeds 


# graphics packages needed
import matplotlib
matplotlib.use('TkAgg')
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from Tkinter import *


import abs_class # imports abstract classes 
import pricers # imports the black's pricers, etc.
import vols
import near_corr # finding the nearest corr matrix 
import spikes

from mrds import mrd_skew, read_curve_vols


s = spikes.spikes_model (1.2, 0.01, 0.1)
# TO IMPROVE, TO IMPROVE 
s.get_discounts (config.market_file, 'WTI_JW', 'usd_rate')

# testing the pricers for europe call
# and APO call 
s.europe_price (100., 102., 0.2, 1.)
s.europe_price_strip (100., 102., 0.2, 0.2, arange (0.5, 2., 0.01))
s.calib_lambda (7.5, 102., 100., arange (0.5, 2., 0.00001))
s.apo_price (100., 102., 0.2, 1.)

s.impl_vol (s.europe_price, 100., arange(80.,120.,1.), 0.2, 1.)
s.impl_vol (s.apo_price, 100., arange(80., 120., 1.), 0.2, 1.)
