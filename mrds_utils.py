# utility functions for the mrds model

import os, csv, pickle, numpy as np

from mrds   import ComSkew
from config import work_dir

import logging
logger = logging.Logger(__name__)

# calibration files
mm_calib_file_actual       = 'mobj/mm_calib.txt'
mm_calib_multi_file_actual = 'mobj/mm_calib_multiple.txt'
single_asset_mm_calib = work_dir + mm_calib_file_actual
multi_asset_mm_calib = work_dir + mm_calib_multi_file_actual


def read_mm_hash(multi_single_ind):
    """
    Reads the market model hash tables from the file

    :param  multi_single_ind: indicator to load single or multiple asset models
                              puts together a hash table of all already calibrated market models

    file has structure: COM, DATE_, NB_FWD (single asset)
                        (COM1, COM2), DATE_, (NFWD_1, NFWD_2)  for multiple assets
    """

    mm_calib_file = single_asset_mm_calib if multi_single_ind is 'single' else multi_asset_mm_calib
    logger.info('Reading prev. calibrated market models from' + mm_calib_file)

    if os.stat(mm_calib_file).st_size == 0:  # file is empty
        return {}  # empty dict

    # file is not empty section
    with open(mm_calib_file) as mm_calib_file_handle:

        mm_hash = dict()

        for com, date_, nb_fwd_str in csv.reader(mm_calib_file_handle):  # csv format

            if multi_single_ind is 'single':
                nb_fwd = int(nb_fwd_str)
                if com not in mm_hash.keys():
                    mm_hash[com] = {date_: nb_fwd}
                else:
                    if date_ not in mm_hash[com].keys():
                        mm_hash[com][date_] = nb_fwd
                    else:
                        mm_hash[com][date_] = max(mm_hash[com][date_], nb_fwd)  # take the maximum of the two calibrated models TODO: THIS IS BAD, KEEP BOTH MODELS

            else:  # multiple models
                nb_fwd_l = tuple(nb_fwd_str.split('___'))  # in the form A1___A2
                nb_fwd_l_int = tuple([int(o) for o in nb_fwd_l])
                com_l = tuple(com.split('___'))
                if com_l not in mm_hash.keys():
                    mm_hash[com_l] = dict()
                    mm_hash[com_l][date_] = [nb_fwd_l_int]
                else:  # already have that com
                    if date_ not in mm_hash[com_l].keys():
                        mm_hash[com_l][date_] = [nb_fwd_l_int]
                    else:
                        if nb_fwd_l_int not in mm_hash[com_l][date_]:
                            mm_hash[com_l][date_].append(nb_fwd_l_int)

    return mm_hash

mm_hash = read_mm_hash('single')
mm_hash_multiple = read_mm_hash('multiple')


def write_mm_hash(multi_single_ind, mm_hash):
    """
    Writes the hash tables into the file.
    TODO: DESCRIBE HERE BETTER.

    :param multi_single_ind: Indicator whether the model calibrates single or multiple asset,
                             ('single', 'multiple')
    :type multi_single_ind: str
    :param mm_hash:
    :type mm_hash:

    """

    with open(single_asset_mm_calib if multi_single_ind is 'single' else multi_asset_mm_calib, 'w') as hashFile:

        for com in mm_hash.keys():
            for date_ in mm_hash[com].keys():
                if multi_single_ind is 'single':
                    hashFile.write(com + ',' + date_ + ',' + str(mm_hash[com][date_]) + '\n')
                else:
                    # go through all fwd list
                    string_comp = ''
                    for asset in com:
                        string_comp += asset + '___'
                    string_comp = string_comp[:-3]  # remove last ___
                    string_comp += ',' + date_ + ','
                    for fwd_pair in mm_hash[com][date_]:
                        string_final = string_comp
                        for nb_fwds in fwd_pair:
                            string_final += str(nb_fwds) + '___'
                        string_final = string_final[:-3]
                        string_final += '\n'
                        hashFile.write(string_final)


def find_adj_tenors(com_nb,
                    adj_fwd_tenors_days,
                    adj_vol_tenors_days):

    if adj_vol_tenors_days is not None and adj_fwd_tenors_days is not None:
        if com_nb in adj_vol_tenors_days.keys():
            if com_nb in adj_fwd_tenors_days.keys():
                adj_fwd_tenors = adj_fwd_tenors_days[com_nb]
                adj_vol_tenors = adj_vol_tenors_days[com_nb]
            else:
                adj_fwd_tenors = adj_fwd_tenors_days[com_nb]
                adj_vol_tenors = None
        else:
            if com_nb in adj_fwd_tenors_days.keys():
                adj_fwd_tenors = adj_fwd_tenors_days[com_nb]
                adj_vol_tenors = None
            else:
                adj_fwd_tenors, adj_vol_tenors = None, None
    elif adj_vol_tenors_days is not None and adj_fwd_tenors_days is None:
        if com_nb in adj_vol_tenors_days.keys():
            adj_vol_tenors = adj_vol_tenors_days[com_nb]
            adj_fwd_tenors = None
        else:
            adj_fwd_tenors, adj_vol_tenors = None, None
    elif adj_vol_tenors_days is None and adj_fwd_tenors_days is not None:
        if com_nb in adj_fwd_tenors_days.keys():
            adj_fwd_tenors = adj_fwd_tenors_days[com_nb]
            adj_vol_tenors = None
        else:
            adj_fwd_tenors, adj_vol_tenors = None, None
    else:
        adj_fwd_tenors, adj_vol_tenors = None, None

    return adj_fwd_tenors, adj_vol_tenors


def find_mm(com, date_, fwd, mm_hash):
    """
    Finds the market model for single asset.

    :param com: commodity considered (WTI)
    :type com: str, or tuple(str) for multiple commodities
    :param date_: date considered
    :type date_:
    :param fwd: number of forward contracts that one needs to calibrate
    :type fwd: int, or tuple(int)
    :param mm_hash: hash of calibrated market models.
    :type mm_hash: dict, where the first level are commodities, the second level are dates calibrated.
    """

    if com not in mm_hash.keys():
        return False

    if date_ not in mm_hash[com].keys():
        return False

    return False if fwd > mm_hash[com][date_] else True


def update_mm_hash(multi_single_ind, mm_hash, new_mm):
    """

    """

    com, date_, nb_fwd = new_mm
    mm_hash_new = mm_hash

    if com not in mm_hash.keys():
        mm_hash_new[com] = dict()
        if multi_single_ind is 'single':
            mm_hash_new[com][date_] = nb_fwd
        else:
            mm_hash_new[com][date_] = [nb_fwd]
    else:
        if date_ not in mm_hash[com].keys():
            if multi_single_ind is 'single':
                mm_hash_new[com][date_] = nb_fwd
            else:
                mm_hash_new[com][date_] = [nb_fwd]
        else:
            if multi_single_ind is 'single':
                if nb_fwd > mm_hash_new[com][date_]:
                    mm_hash_new[com][date_] = nb_fwd
            else:
                if nb_fwd not in mm_hash_new[com][date_]:
                    mm_hash_new[com][date_].append(nb_fwd)

    return mm_hash_new


def mrds_calib( com
              , date_
              , nb_fwd
              , mm_hash      = mm_hash
              , mt           = True
              , model_ind    = 'skew' ):
    """
    Calibrates the mrds object. If the model is already calibrated, it simply loads it.

    :param com: commodity considered, like 'WTI', ...
    :type com: str

    """

    mobj_mm_beg = 'mobj/mm_'

    if find_mm(com, date_, nb_fwd, mm_hash):  # finds if the model is already calibrated.
        mm_file = work_dir + mobj_mm_beg + str(com) + '_' + str(date_) + '_' + str(mm_hash[com][date_]) + '.obj'
        mm = pickle.load(open(mm_file, 'rb'))
        mm.multi_thread_ind = mt
    else:
        mm = mrds_calib_db( com
                          , com
                          , date_
                          , nb_fwd
                          , mt        = mt
                          , model_ind = model_ind
                          , cuda_ind  = False )
        pickle.dump( mm
                   , open(work_dir + mobj_mm_beg + str(com) + '_' + str(date_) + '_' + str(nb_fwd) + '.obj', 'wb'))
        write_mm_hash('single', update_mm_hash('single', mm_hash, [com, date_, nb_fwd]))

    return mm


def mrds_calib_db( com_fwd
                 , com_vol
                 , date_
                 , nb_fwd
                 , mt        = True
                 , model_ind = 'skew'
                 , cuda_ind  = False ):

    mm = ComSkew( 1
                 , date_
                 , multi_thread_ind  = mt
                 , model_skew_ln_ind = model_ind
                 , cuda_ind          = cuda_ind)

    mm.read_curve_vol_data_db(date_, 0, com_fwd, com_vol, sub_idx_rows=np.arange(nb_fwd))
    mm.read_discount_curve_db(date_)
    mm.read_model_config_db(0)
    mm.set_other_params(0)

    if mm.vol_surface_name_list[0] == 'ATM':
        mm.model_skew_ln_ind = 'ln_ln'
    mm.black_vol_calibration(0)
    if model_ind is 'skew':
        mm.calibrate_skew_params(0)
    mm.generate_large_corr_mat()

    return mm


def mrds_calib_multiple( com_l
                       , date_
                       , nb_fwd_l
                       , mm_hash             = mm_hash_multiple
                       , model_ind           = 'skew'
                       , adj_fwd_tenors_days = None
                       , adj_vol_tenors_days = None
                       , multi_thread_ind    = True
                       , cuda_ind            = False):
    """
    Calibrates the multiple models

    """

    nb_comm = len(com_l)
    assert(nb_comm == len(nb_fwd_l))
    com_str = reduce(lambda x, y: x+'___'+y, com_l)
    nb_fwd_str = reduce(lambda x, y: str(x)+'___'+str(y), nb_fwd_l)

    mm_calibrated = find_mm(com_l[0], date_, nb_fwd_l[0], mm_hash) if nb_comm == 1 else find_mm(tuple(com_l), date_, tuple(nb_fwd_l), mm_hash)

    mm_file = work_dir + 'mobj/mm_' + str(com_str) + '_' + \
        str(date_) + '_' + str(nb_fwd_str) + '.obj'
    if mm_calibrated:
        mm = pickle.load(open(mm_file))
    else:
        mm = mrds_calib_db_multiple(com_l, com_l, date_, nb_fwd_l,
                                    model_ind=model_ind,
                                    adj_fwd_tenors_days=adj_fwd_tenors_days,
                                    adj_vol_tenors_days=adj_vol_tenors_days,
                                    mt=multi_thread_ind,
                                    cuda_ind=cuda_ind)
        pickle.dump(mm, open(mm_file, 'wb'))
        mm_hash_new = update_mm_hash('multiple', mm_hash,
                                     [tuple(com_l), date_, tuple(nb_fwd_l)])
        write_mm_hash('multiple', mm_hash_new)

    return mm


def mrds_calib_db_multiple( com_fwd_l
                          , com_vol_l
                          , date_
                          , nb_fwd_l
                          , mt                  = True
                          , model_ind           = 'skew'
                          , adj_fwd_tenors_days = None
                          , adj_vol_tenors_days = None
                          , cuda_ind            = False):
    """
    TODO: FINISH HERE.

    :param com_fwd_l: list of com to calibrate
    col_vol_l ... list of vols for counterparted coms
    nb_fwd_l ... list of nb of forwards [12, 10, 12]
    :param mt: multithreading indicator
    :type mt: bool
    :param adj_fwd_tenors_days:  dict of {com_nb: nb_days_adj}
    """

    nb_comm = len(com_fwd_l)
    assert(nb_comm == len(com_vol_l) and nb_comm == len(nb_fwd_l)), \
        "Unequal lists for fwd, vol, nb_fwd"

    mm = MrdSkew( nb_comm
                , date_
                , multi_thread_ind  = mt
                , model_skew_ln_ind = model_ind
                , cuda_ind          = cuda_ind )

    mm.read_discount_curve_db(date_)
    for com_nb, (com_used, vol_used) in enumerate(zip(com_fwd_l, com_vol_l)):
        adj_fwd_tenors, adj_vol_tenors = find_adj_tenors(com_nb,
                                                         adj_fwd_tenors_days,
                                                         adj_vol_tenors_days)
        mm.read_curve_vol_data_db(date_, com_nb, com_used, vol_used,
                                  sub_idx_rows=np.arange(nb_fwd_l[com_nb]),
                                  adj_fwd_tenors_days=adj_fwd_tenors,
                                  adj_vol_tenors_days=adj_vol_tenors)
        mm.read_model_config_db(com_nb)
        mm.set_other_params(com_nb)
        mm.black_vol_calibration(com_nb)
        if model_ind is 'skew':
            mm.calibrate_skew_params(com_nb)
    mm.generate_large_corr_mat()

    return mm


def compute_partial_deltas(mm
                           , pricer
                           , params
                           , nb_sim
                           , subset_idx
                           , delta=.01
                           , seed=None
                           , verbose=None):
    """
    computes partial deltas for all assets
    :param mm: calibrated market model
    :type mm:
    :param pricer:  has to be in the form
       pricer (mo, params) where
         mo ... market object
         params ... parameters
    :param nb_sim: nb. of simulations
    :type nb_sim: int
    :param subset_idx: for which futures to compute the deltas
    :type subset_idx: list[int]
    :param delta: bump, default is the same as for structured desk
    # seed ... seed for random numbers
    # verbose ... prints additional things
    """

    deltas = mm._empty_list_fct(mm.nb_assets)  # empty list of deltas

    # TODO: WOULD BE BETTER INTERPRETED AS A DICTIONARY CORRECT CORRECT CORRECT
    for asset_nb in range(mm.nb_assets):
        deltas[asset_nb] = np.zeros(len(subset_idx))
        for idx in range(len(subset_idx)):
            logging.debug("Computing deltas for asset", asset_nb, "and fwd. idx.", subset_idx[idx])
            logging.debug("Original price of future's", subset_idx[idx], "for asset", asset_nb, "=", \
                          mm.forward_curve_list[asset_nb][subset_idx[idx]])
            mm.forward_curve_list[asset_nb][subset_idx[idx]] *= (1.0 + delta)
            mm.simulate_curves(nb_sim, seed)
            deltas[asset_nb][idx] = pricer(mm, params)  # pricer with forward curves bumped
            mm.forward_curve_list[asset_nb][subset_idx[idx]] /= (1.0 + delta)
            mm.simulate_curves(nb_sim, seed)
            price2 = pricer(mm, params)
            deltas[asset_nb][idx] -= price2
            logging.debug("Bumped price", mm.forward_curve_list[asset_nb][subset_idx[idx]])
            logging.debug("price 1:", deltas[asset_nb][idx])
            logging.debug("price 2:", price2)
    return np.array(deltas)


def compute_partial_vegas(mm, pricer, params, nb_sim, subset_idx,
                          delta=0.001, seed=None, verbose='none'):
    vegas = mm._empty_list_fct(mm.nb_assets)  # empty list of deltas

    for asset_nb in range(mm.nb_assets):
        vegas[asset_nb] = np.zeros(len(subset_idx))
        for idx in range(len(subset_idx)):
            # WRONG WRONG WRONG - HERE RECALIBRATION IS NEEDED
            mm.__sigmaVecList[asset_nb][subset_idx[idx]] += delta
            mm.simulate_curves(nb_sim, seed)
            vegas[asset_nb][idx] = pricer(mm, params)  # pricer with vol curves bumped
            mm.atm_vol_list[asset_nb][subset_idx[idx]] -= delta
            mm.simulate_curves(nb_sim, seed)
            vegas[asset_nb][idx] -= pricer(mm, params)
            vegas[asset_nb][idx] *= 100.0  # scaling

    return vegas
