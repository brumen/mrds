import config
import numpy as np
import sqlite3
import datetime as dt
import pickle

import ds
# import ao_db
# import air_search
import air_option as ao
# import ao_estimate as aoe


def test_ao_1(nb_sim=50000):
    """
    tests the air option 
    """
    tickets = np.linspace(450., 500., 50)
    s_v = 0.2 * np.ones(len(tickets))
    T_l = np.linspace(1./365., 62./365., 62)
    ao_p = {'model': 'max',
            'F_max_prev': np.zeros((len(tickets), nb_sim))}

    print ao.air_option_seq(tickets, s_v, T_l, 200., 100., 1.,
                            ao_f=ao.ao_f, ao_p=ao_p,
                            nb_sim=nb_sim,
                            underlyer='ln')


def test_ao_2(nb_sim=50000):
    """
    tests the air option 
    """
    tickets = np.linspace(450., 400., 50)
    s_v = 100. * np.ones(len(tickets))
    T_l = np.linspace(1./365., 62./365., 62)
    print ao.air_option(tickets, s_v, T_l, 450., 100., 0.2, d_c=0.95, nb_sim=nb_sim,
                        model='prob')


def test_ao_25(nb_sim=50000):
    """
    tests the air option 
    """
    tickets = np.linspace(450., 400., 50)
    s_v = 100. * np.ones(len(tickets))
    T_l = np.linspace(1./365., 180./365., 180)
    print ao.air_option(tickets, s_v, T_l, 450., 100., 0.2, d_c=0.95, nb_sim=nb_sim,
                        model='max')


def test_ao_26(nb_sim=10000):
    """
    tests the air option 
    """
    tickets = np.linspace(450., 400., 50)
    s_v = 100. * np.ones(len(tickets))
    T_l = np.linspace(1./365., 180./365., 180)

    ao_p = {'model': 'max',
            'F_max_prev': np.zeros((len(tickets), nb_sim))}
    ao_f = ao.ao_f

    print ao.air_option_seq(tickets, s_v, T_l, 450., 100., 0.2,
                            ao_f=ao_f, ao_p=ao_p,
                            d_c=0.95, nb_sim=nb_sim,
                            model='max')


def test_ao_27(nb_sim=10000):
    """
    tests the air option 
    """
    tickets = np.linspace(450., 400., 50)
    s_v = 100. * np.ones(len(tickets))
    T_l = np.linspace(1./365., 180./365., 180)

    ao_p = {'model': 'prob',
            'F_max_prev': np.zeros((len(tickets), nb_sim))}
    ao_f = ao.ao_f

    print ao.air_option_seq(tickets, s_v, T_l, 450., 100., 0.2,
                            ao_f=ao_f, ao_p=ao_p,
                            d_c=0.95, nb_sim=nb_sim,
                            model='max')
    

def test_ao_3(nb_sim=2**9*1000):
    """
    tests the air option 
    """
    tickets = np.linspace(450., 400., 50)
    s_v = 100. * np.ones(len(tickets))
    T_l = np.linspace(1./365., 62./365., 62)
    print ao.air_option(tickets, s_v, T_l, 450., 100., 0.2, d_c=0.95, nb_sim=nb_sim,
                        model='prob', cuda_ind=True)


def test_ao_32(nb_sim=10000, date_today='20161001'):
    """
    tests the air option 
    """
    tickets = np.linspace(450., 400., 50)

    date_today_dt = ds.convert_str_datetime(date_today)

    s_v = 100. * np.ones(len(tickets))
    T_l = np.linspace(1./365., 180./365., 180)

    ao_p = {'model': 'max',
            'F_max_prev': np.zeros((len(tickets), nb_sim))}
    ao_f = ao.ao_f

    print ao.air_option_seq(tickets, s_v, T_l, 450., 100., 0.2,
                            ao_f=ao_f, ao_p=ao_p,
                            d_c=0.95, nb_sim=nb_sim,
                            model='max')


def test_ao_33():
    vals = ao.compute_option_val(option_end_date='20161030',
                                 outbound_date_end='2016-10-05',
                                 nb_sim=2000, s_v_test=200.)
    print "VALS:", vals


def test_compute_opt_val_nosearch_1(s_v_test=0.2):
    """
    compute the option val 
    """
    F_v = np.linspace(600., 900., 250)
    date_option_start, date_option_end = '20161101', '20161231'
    K = 900.
    simplify_compute = 'all_sim_dates' # 'take_last_only' is the other option
    s_v_test = s_v_test
    rho = 0.95
    
    res = ao.compute_option_val_nosearch(F_v, option_start_date=date_option_start,
                                         option_end_date=date_option_end,
                                         K=K,
                                         simplify_compute=simplify_compute,
                                         s_v_test=s_v_test,
                                         rho=rho)
    return res


def test_compute_vols(airline='Alaska Airlines'):
    """
    airline from the cache database 
    """
    aoe.all_vols_by_airline(airline, use_cache=True)
    

def test_simple():
    v1 = ao.compute_option_val(origin_place='SFO', dest_place='EWR', option_start_date='20161115',
                               option_end_date='20161116', option_ret_start_date='20161122',
                               option_ret_end_date='20161123', outbound_date_start='2016-11-05',
                               outbound_date_end='2016-11-06', inbound_date_start='2016-12-12',
                               inbound_date_end='2016-12-13', K=200.0, penalty=100.0, p_c=0.2, carrier='UA',
                               nb_sim=10000, rho=0.95, country='US', currency='USD', locale='en-US', adults=1,
                               model='max', cuda_ind=False, s_v_test=1.2, debug=False, errors='graceful',
                               simplify_compute='take_last_only', underlyer='ln', mt_ind=True, price_by_range=False,
                               return_flight=True)

    
def test_ao_new1(simplify_compute='take_last_only'):
    v1 = ao.compute_option_val(option_start_date='20161211',
                               option_end_date='20161212',
                               option_ret_start_date='20161201',
                               option_ret_end_date='20161202',
                               K=1600.0, penalty=100.0,
                               nb_sim=10000, rho=0.95,
                               simplify_compute=simplify_compute,
                               underlyer='ln',
                               return_flight=True)
    print v1[0]


def test_ao_new2(simplify_compute='take_last_only'):
    v1 = ao.compute_option_val(option_start_date='20161201',
                               option_end_date='20170515',
                               outbound_date_start='2017-05-25',
                               outbound_date_end='2017-05-31',
                               K=800.0, penalty=100.0,
                               nb_sim=10000, rho=0.95,
                               simplify_compute=simplify_compute,
                               underlyer='ln',
                               return_flight=False)
    print v1[0]


def test_ao_new3(simplify_compute='take_last_only', nb_sim=10000, cuda_ind=False):
    v1 = ao.compute_option_val(option_start_date='20161204',
                               option_end_date='20161231',
                               option_ret_start_date='20161204',
                               option_ret_end_date='20161231',
                               K=900.0, penalty=100.0,
                               nb_sim=10000, rho=0.95,
                               simplify_compute=simplify_compute,
                               underlyer='n',
                               return_flight=False,
                               cuda_ind=cuda_ind)
    print v1[0]


def test_ao_new4(simplify_compute='take_last_only', nb_sim=10000, cuda_ind=True):
    ress = pickle.load(open('/home/brumen/work/mrds/ao/tmp/res_1.obj'))
    v1 = ao.compute_option_val(option_start_date='20161204',
                               option_end_date='20161231',
                               option_ret_start_date='20161204',
                               option_ret_end_date='20161231',
                               K=900.0, penalty=100.0,
                               nb_sim=nb_sim, rho=0.95,
                               simplify_compute=simplify_compute,
                               underlyer='n',
                               return_flight=False,
                               cuda_ind=cuda_ind,
                               res_supplied=ress)
    print v1[0]


def test_ao_new5(simplify_compute='take_last_only', nb_sim=10000, cuda_ind=True):
    ress = pickle.load(open('/home/brumen/work/mrds/ao/tmp/res_1.obj'))
    v1 = ao.compute_option_val(option_start_date='20161215',
                               option_end_date='20161216',
                               option_ret_start_date='20161225',
                               option_ret_end_date='20161226',
                               K=900.0, penalty=100.0,
                               nb_sim=nb_sim, rho=0.95,
                               simplify_compute=simplify_compute,
                               underlyer='n',
                               return_flight=False,
                               cuda_ind=cuda_ind,
                               res_supplied=ress)
    print v1[0]


# test:
#   1. what if the inputs are wrong
#   2. what if
