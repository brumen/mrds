#! /usr/bin/python

#
# File defines:
#   mrd skew model for commodities (state reference)
#   a general diffusion model

import config
import numpy as np
import time # timing 
import datetime

import weather as we

def calc_1(gpu_ind):

    if gpu_ind:
        print "GPU working"
    else:
        print "CPU working"

    # historical parameters: A,B,C,omega,phi = hp
    # hp = (5.97, 6.57*10**(-5), 10.4, 2 * pi , -2.01, 0.5)
    hp = (54.161, 0.0001918, 22.183, 2*np.pi, -2.034, 114.261)

    date_o = datetime.date(1973,1,1)
    date_p = datetime.date(2012,11,24)
    date_s = datetime.date(2012, 4,1)
    date_l = [date_o, date_p, date_s]
    nm_1 = 1
    nm_2 = 3
    
    HDD_date_l = {0: (date_s, nm_1),
                  1: (date_s, nm_2)
                  }
    
    HDD_price_l = {0: 480., 
                   1: 630.
                   }

    w1 = we.weather(1000, hp, date_l, gpu_ind=gpu_ind)
    
    t1 = time.time()
    hdd1_hist = w1.HDD_histo (nm_1, HDD_date_l)
    t1 = time.time() - t1


    t2 = time.time()
    hdd2_hist = w1.HDD_histo (nm_1, HDD_date_l)
    t2 = time.time() - t2
    print "hdd 1m hist = ", hdd1_hist
    print "hdd 3m hist = ", hdd2_hist
    print "time 1m HDD= ", t1
    print "time 3m HDD= ", t2

    # now options 
    t3 = time.time()
    hddo1_hist = w1.HDDO_histo (hdd1_hist, nm_1, HDD_date_l)
    t3 = time.time() - t3

    t4 = time.time()
    hddo2_hist = w1.HDDO_histo (hdd1_hist, nm_2, HDD_date_l)
    t4 = time.time() - t4

    print "hddo 1m hist = ", hddo1_hist
    print "hddo 3m hist = ", hddo2_hist
    print "time 1m HDDO= ", t3
    print "time 3m HDDO= ", t4


    # HDD options given in a format (K, price)

    HDDO_price_l = {0: (480., 8.), 
                    1: (630., 7.)
                    }
    
    # price under the current parameters 
    print "calibrating"
    t1 = time.time()
    ca1 = w1.HD_calib_all(HDD_date_l, HDD_price_l, HDDO_price_l)
    t1 = time.time() - t1
    print "ca1 = ", ca1
    print "time = ", t1


def calc_2(gpu_ind):

    if gpu_ind:
        print "GPU working"
    else:
        print "CPU working"

    # historical parameters: A,B,C,omega,phi = hp
    # hp = (5.97, 6.57*10**(-5), 10.4, 2 * pi , -2.01, 0.5)
    hp = (54.161, 0.0001918, 22.183, 2*np.pi, -2.034, 114.261)

    date_o = datetime.date(1973,1,1)
    date_p = datetime.date(2012,11,24)
    date_s = datetime.date(2012, 4,1)
    date_l = [date_o, date_p, date_s]
    nm_1 = 1
    nm_2 = 3
    nm_3 = 6
    HDD_date_l = {0: (date_s, nm_1),
                  1: (date_s, nm_2),
                  2: (date_s, nm_3)
                  }
    
    HDD_price_l = {0: 540., 
                   1: 550.,
                   2: 1225.
                   }

    w1 = we.weather(1000, hp, date_l, gpu_ind=gpu_ind)
    
    t1 = time.time()
    hdd1_hist = w1.HDD_histo (nm_1, HDD_date_l)
    t1 = time.time() - t1


    t2 = time.time()
    hdd2_hist = w1.HDD_histo (nm_1, HDD_date_l)
    t2 = time.time() - t2

    t25 = time.time()
    hdd3_hist = w1.HDD_histo (nm_3, HDD_date_l)
    t25 = time.time() - t25


    print "hdd 1m hist = ", hdd1_hist
    print "hdd 3m hist = ", hdd2_hist
    print "hdd 6m hist = ", hdd3_hist
    print "time 1m HDD= ", t1
    print "time 3m HDD= ", t2
    print "time 6m HDD= ", t25

    # now options 
    t3 = time.time()
    hddo1_hist = w1.HDDO_histo (hdd1_hist, nm_1, HDD_date_l)
    t3 = time.time() - t3

    t4 = time.time()
    hddo2_hist = w1.HDDO_histo (hdd2_hist, nm_2, HDD_date_l)
    t4 = time.time() - t4

    t5 = time.time()
    hddo3_hist = w1.HDDO_histo (hdd3_hist, nm_3, HDD_date_l)
    t5 = time.time() - t5


    print "hddo 1m hist = ", hddo1_hist
    print "hddo 3m hist = ", hddo2_hist
    print "hddo 6m hist = ", hddo3_hist
    print "time 1m HDDO= ", t3
    print "time 3m HDDO= ", t4
    print "time 6m HDDO= ", t5


    # HDD options given in a format (K, price)

    HDDO_price_l = {0: (550., 8.), 
                    1: (540., 7.),
                    2: (1225., 10.)
                    }
    
    # price under the current parameters 
    print "calibrating"
    t6 = time.time()
    ca1 = w1.HD_calib_all(HDD_date_l, HDD_price_l, HDDO_price_l)
    t6 = time.time() - t6
    print "ca1 = ", ca1
    print "time = ", t6




#calc_1(False)
#calc_1(True)

calc_2(True)
