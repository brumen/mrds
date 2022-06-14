""" Data structures - forward and vol curves.
    IMPORTANT: ONLY MARKET DATE POSSIBLE: 20150401, i.e. 2015, 04, 01; Apr 1, 2015, datetime.date(2015, 4, 1)
"""

import datetime
import numpy as np

from typing import Dict, List, Tuple

from mrds.data                import curves_vols
from mrds.data.discount_rates import discount_curve_dates, discount_curve_vals, discount_curve_ois_rates


# mapping of commodity names to vol parametrization
vol_hash = { 'WTI'       : ('JWSS7', curves_vols.wti_vol_curve_dates, curves_vols.wti_vol_curve_vals, curves_vols.wti_vol_curve_dates_vals)
           , 'BRENT'     : ('JWSS7', curves_vols.brent_vol_dates    , curves_vols.brent_vol_vals    , curves_vols.brent_vol_curve_dates_vals)
           , 'ATSI-PEAK' : ('JWSS7', curves_vols.atsi_peak_vol_dates, curves_vols.atsi_peak_vol_vals, curves_vols.atsi_peak_vol_dates_vals)
           , 'ATSI_2X16'  : ('JWSS7', curves_vols.atsi_2x16_vol_dates, curves_vols.atsi_2x16_vol_vals, dict(zip(curves_vols.atsi_2x16_vol_dates, curves_vols.atsi_2x16_vol_vals)))
           , 'ATSI_7X8'         : ('JWSS7', curves_vols.atsi_2x16_vol_dates, curves_vols.atsi_7x8_vol_vals, dict(zip(curves_vols.atsi_2x16_vol_dates, curves_vols.atsi_7x8_vol_vals)))
           , 'NG_MICHCON_GD-PEAK': ('ATM', curves_vols.ng_michcon_gd_peak_vol_dates, curves_vols.ng_michcon_gd_peak_vol_vals, dict(zip(curves_vols.ng_michcon_gd_peak_vol_dates, curves_vols.ng_michcon_gd_peak_vol_vals)))
           , 'NG_MICHCON_CASHVOL': ('ATM', curves_vols.ng_michcon_cv_vol_dates, curves_vols.ng_michcon_cv_vol_vals)
           , 'PJMW-OFFPEAK_CV'   : ('ATM', curves_vols.pjm_offpeak_cv_vol_dates, curves_vols.pjm_offpeak_cv_vol_vals)
           , 'PJMW-PEAK_CV'      : ('ATM', curves_vols.pjm_peak_cv_vol_dates, curves_vols.pjm_peak_cv_vol_vals)}


def brentCurve():
    """ Ancillary routine to generate brent curve.
    """

    curve_dates = curves_vols.brent_curve_dates
    curve_vals_init = curves_vols.brent_curve_vals
    brent_spread = np.linspace(6, 7, len(curve_vals_init))  # fictitious spread
    curve_vals = [x + s for x, s in zip(curve_vals_init, brent_spread)]

    return curve_dates, curve_vals


fwd_hash = { 'WTI':                (curves_vols.wti_curve_dates, curves_vols.wti_curve_vals)
           , 'BRENT':              brentCurve()
           , 'ATSI-PEAK':          (curves_vols.atsi_peak_curve_dates, curves_vols.atsi_peak_curve_vals)
           , 'ATSI_7X8':           (curves_vols.atsi_7x8_curve_dates, curves_vols.atsi_7x8_curve_vals)
           , 'ATSI_2X16':          (curves_vols.atsi_2x16_curve_dates, curves_vols.atsi_2x16_curve_vals)
           , 'NG_MICHCON_GD-PEAK': (curves_vols.ng_michcon_gd_peak_dates, curves_vols.ng_michcon_gd_peak_curve_vals)
           , 'NG_MICHCON_CASHVOL': (curves_vols.ng_michcon_cv_curve_dates, curves_vols.ng_michcon_cv_curve_vals)
           , 'PJMW-OFFPEAK_CV':    (curves_vols.pjm_offpeak_cv_curve_dates, curves_vols.pjm_offpeak_cv_curve_vals)
           , 'PJMW-PEAK_CV':       (curves_vols.pjm_peak_cv_curve_dates, curves_vols.pjm_peak_cv_curve_vals)
           , 'DISCOUNT':           (discount_curve_dates, discount_curve_vals)
           , 'DISCOUNT_QL'       : discount_curve_ois_rates  # QuantLib ois rates.
           , }


def get_forward_curve( commodity : str
                     , mkt_date  : datetime.date ) -> Tuple[List[datetime.date], List[float]]:
    """ Gets the forward curve (ALWAYS USE DATE 20150410) for a particular date.

    :param commodity: name of the commodity curve
    :param mkt_date: market date for which the com curve is needed (not used here, but the interface should be such)
    :returns: commodity forward curve, given as a tuple of dates, and values.
    """

    return fwd_hash[commodity]


def get_vol_curve( com_name : str
                 , mkt_date : datetime.date) -> Tuple[str, Dict]:
    """ Gets the vol curve com_name for a particular market date.
        In out example, we don't use the second parameter.

    :param com_name: commodity name of the commodity you want to fetch.
    :param mkt_date: market date for the curve
    :returns: a tuple of (commodity typ e.g. 'JWSS7', and vol curve,
              where the vol curve is indexed by vol dates, and contains the vol parameters.
              in case of JWSS7: [S0, atm, skew, smile, putslope, putbend, callslope, callbend]
    """

    vol_type, _, _, vol_curve_with_dates = vol_hash[com_name]

    return vol_type, vol_curve_with_dates


def get_fwd_vol_curve_numeric_tenor( curve_name : str
                                   , mkt_date   : datetime.date
                                   , fwd_vol_ind     = 'fwd'
                                   , adj_tenors_days = None ):
    """ Gets the raw forward or vol curve name.

    :param curve_name: forward or vol curve name.
    :param mkt_date: market date.
    :param fwd_vol_ind: indicator of forward or vol curve
    :param adj_tenors_days: integer, to adjust the number of days in the forward/vol curve.
    """

    if fwd_vol_ind == 'fwd':
        fwd_vol_tenors_raw, fwd_vol_values_raw = get_forward_curve(curve_name, mkt_date)
        if adj_tenors_days is not None:
            fwd_vol_tenors_vals = [(ot - datetime.timedelta(days=adj_tenors_days), val)
                                   for ot, val in zip(fwd_vol_tenors_raw, fwd_vol_values_raw)
                                   if ot - datetime.timedelta(days=adj_tenors_days) > mkt_date]
            fwd_vol_tenors, fwd_vol_values = zip(*fwd_vol_tenors_vals)
        else:
            fwd_vol_tenors, fwd_vol_values = fwd_vol_tenors_raw, fwd_vol_values_raw

    else:
        vol_curve_type, vol_curve_data = get_vol_curve(curve_name, mkt_date)
        if adj_tenors_days is not None:
            fwd_vol_tenors_vals = [(ot - datetime.timedelta(days=adj_tenors_days), val)
                                   for ot, val in vol_curve_data
                                   if ot - datetime.timedelta(days=adj_tenors_days) > mkt_date]
            fwd_vol_tenors, fwd_vol_values = zip(*fwd_vol_tenors_vals)
        else:
            fwd_vol_tenors, fwd_vol_values = list(vol_curve_data.keys()), vol_curve_data

    diffs = [ten_ - mkt_date for ten_ in fwd_vol_tenors if ten_ > mkt_date]
    fwd_vol_tenors_numeric = np.array([elt.days for elt in diffs])/365.
    fwd_vol_tenors_code = [(fwt.month, fwt.year) for fwt in fwd_vol_tenors if fwt > mkt_date]
    fwd_vol_values_unexpired = np.array([fwd_vol_vals for fwt, fwd_vol_vals
                                         in zip(fwd_vol_tenors, fwd_vol_values)
                                         if fwt > mkt_date])

    return fwd_vol_tenors_numeric, fwd_vol_values_unexpired, fwd_vol_tenors_code, fwd_vol_tenors


def read_data_matched_tenors( mktDate : datetime.date
                            , fwd_curve_name : str
                            , vol_curve_name : str
                            , adj_fwd_tenors_days = None
                            , adj_vol_tenors_days = None):
    """ Matches the tenors of the forward and volatility curves.

    :param mktDate: market date for which the curves are obtained.
    :param fwd_curve_name: forward curve
    """

    fwd_tenors, fwd_curve, fwd_tenors_code, fwd_tenors_dt = \
        get_fwd_vol_curve_numeric_tenor(fwd_curve_name
                                        , mktDate
                                        , fwd_vol_ind     = 'fwd'
                                        , adj_tenors_days = adj_fwd_tenors_days)

    option_tenors, vol_params, option_tenors_code, option_tenors_dt_orig = \
        get_fwd_vol_curve_numeric_tenor(vol_curve_name
                                        , mktDate
                                        , fwd_vol_ind     = 'vol'
                                        , adj_tenors_days = adj_vol_tenors_days)

    # if option_tenors_dt and fwd_tenors_dt are the same, remove 1 day from option_tenors
    if fwd_tenors_dt == option_tenors_dt_orig:
        option_tenors_dt = [ot - datetime.timedelta(1) for ot in option_tenors_dt_orig]
    else:
        option_tenors_dt = option_tenors_dt_orig

    # match according to which curve is shorter
    if len(fwd_tenors_code) > len(option_tenors_code):
        match_idx = [(n_fwd, n_opt) for n_opt, item2 in enumerate(option_tenors_code)
                     for n_fwd, item1 in enumerate(fwd_tenors_code)
                     if item1 == item2]
    else:
        match_idx = [(n_fwd, n_opt) for n_opt, item2 in enumerate(fwd_tenors_code)
                     for n_fwd, item1 in enumerate(option_tenors_code)
                     if item1 == item2]

    def select_elts(arr, idx, fwd_opt_ind='fwd'):
        """ Selects the elements from array arr given the indices in idx

        """
        if fwd_opt_ind == 'fwd':
            return [arr[elt_fwd] for (elt_fwd, elt_opt) in idx]

        return [arr[elt_opt] for (elt_fwd, elt_opt) in idx]

    fwd_tenors_matched         = select_elts(fwd_tenors        , match_idx, 'fwd')
    fwd_tenors_code_matched    = select_elts(fwd_tenors_code   , match_idx, 'fwd')
    fwd_tenors_dt_matched      = select_elts(fwd_tenors_dt     , match_idx, 'fwd')
    fwd_curve_matched          = select_elts(fwd_curve         , match_idx, 'fwd')
    option_tenors_dt_matched   = select_elts(option_tenors_dt  , match_idx, 'opt')
    option_tenors_matched      = select_elts(option_tenors     , match_idx, 'opt')
    option_tenors_code_matched = select_elts(option_tenors_code, match_idx, 'opt')
    vol_surface_params_matched = select_elts(vol_params        , match_idx, 'opt')

    # sorting
    def sorting_fct(sort_order, to_be_sorted):
        return [x for (y, x) in sorted(zip(sort_order, to_be_sorted))]

    fwd_tenors_final = sorting_fct(fwd_tenors_dt_matched, fwd_tenors_matched)
    fwd_tenors_code_final = sorting_fct(fwd_tenors_dt_matched, fwd_tenors_code_matched)
    fwd_curve_final = sorting_fct(fwd_tenors_dt_matched, fwd_curve_matched)
    fwd_tenors_dt_final = np.sort(fwd_tenors_dt_matched)

    option_tenors_final = sorting_fct(option_tenors_dt_matched, option_tenors_matched)
    option_tenors_code_final = sorting_fct(option_tenors_dt_matched, option_tenors_code_matched)
    vol_surface_params_final = sorting_fct(option_tenors_dt_matched, vol_surface_params_matched)
    option_tenors_dt_final = np.sort(option_tenors_dt_matched)

    # remove duplicates from list_2 with regards to list_1
    def remove_duplicates(list_1, list_2):
        res = []
        no_duplicates = []
        for x, y in zip(list_1, list_2):
            if x not in no_duplicates:
                res.append(y)
                no_duplicates.append(x)

        return res

    return {'fwd_tenors'        : np.array(remove_duplicates(fwd_tenors_dt_final, fwd_tenors_final)),
            'fwd_curve'         : np.array(remove_duplicates(fwd_tenors_dt_final, fwd_curve_final)),
            'fwd_tenors_code'   : np.array(remove_duplicates(fwd_tenors_dt_final, fwd_tenors_code_final)),
            'fwd_tenors_dt'     : remove_duplicates(fwd_tenors_dt_final, fwd_tenors_dt_final),
            # vol part
            'option_tenors'     : np.array(remove_duplicates(option_tenors_dt_final, option_tenors_final)),
            'option_tenors_code': np.array(remove_duplicates(option_tenors_dt_final, option_tenors_code_final)),
            'option_tenors_dt'  : remove_duplicates(option_tenors_dt_final, option_tenors_dt_final),
            'vol_surface_params': np.array(remove_duplicates(option_tenors_dt_final, vol_surface_params_final))
            , }
