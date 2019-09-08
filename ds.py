# ONLY MARKET DATE POSSIBLE: 20150401, i.e. 2015, 04, 01; Apr 4, 2015
import datetime
import numpy as np
import matplotlib.pyplot as plt
import calendar as cal

import ds_data

# forward codes
fwd_mth_codes = ['f', 'g', 'h',
                 'j', 'k', 'm',
                 'n', 'q', 'u',
                 'v', 'x', 'z']

fwd_mapping_codes = {'f': 'JAN',
                     'g': 'FEB',
                     'h': 'MAR',
                     'j': 'APR',
                     'k': 'MAY',
                     'm': 'JUN',
                     'n': 'JUL',
                     'q': 'AUG',
                     'u': 'SEP',
                     'v': 'OCT',
                     'x': 'NOV',
                     'z': 'DEC' }

fwd_codes = {'f': 1,
             'g': 2,
             'h': 3,
             'j': 4,
             'k': 5,
             'm': 6,
             'n': 7,
             'q': 8,
             'u': 9,
             'v': 10,
             'x': 11,
             'z': 12 }

# mapping of commodity names to vol parametrization
vol_hash = { 'WTI'       : ('JWSS7', ds_data.wti_vol_curve_dates, ds_data.wti_vol_curve_vals, ds_data.wti_vol_curve_dates_vals)
           , 'BRENT'     : ('JWSS7', ds_data.brent_vol_dates, ds_data.brent_vol_vals)
           , 'ATSI-PEAK' : ('JWSS7', ds_data.atsi_peak_vol_dates, ds_data.atsi_peak_vol_vals)
           , 'ATSI_7X8'  : ('JWSS7', ds_data.atsi_2x16_vol_dates, ds_data.atsi_2x16_vol_vals)
           , 'ATSI_2X16'         : ('JWSS7', ds_data.atsi_2x16_vol_dates, ds_data.atsi_7x8_vol_vals)
           , 'NG_MICHCON_GD-PEAK': ('ATM', ds_data.ng_michcon_gd_peak_vol_dates, ds_data.ng_michcon_gd_peak_vol_vals)
           , 'NG_MICHCON_CASHVOL': ('ATM', ds_data.ng_michcon_cv_vol_dates, ds_data.ng_michcon_cv_vol_vals)
           , 'PJMW-OFFPEAK_CV'   : ('ATM', ds_data.pjm_offpeak_cv_vol_dates, ds_data.pjm_offpeak_cv_vol_vals)
           , 'PJMW-PEAK_CV'      : ('ATM', ds_data.pjm_peak_cv_vol_dates, ds_data.pjm_peak_cv_vol_vals) }


def brentCurve():
    """
    Ancillary routine to generate brent curve.
    """

    curve_dates = ds_data.brent_curve_dates
    curve_vals_init = ds_data.brent_curve_vals
    brent_spread = np.linspace(6, 7, len(curve_vals_init))  # fictitious spread
    curve_vals = [x + s for x, s in zip(curve_vals_init, brent_spread)]

    return curve_dates, curve_vals


fwd_hash = { 'WTI':                (ds_data.wti_curve_dates, ds_data.wti_curve_vals)
           , 'BRENT':              brentCurve()
           , 'ATSI-PEAK':          (ds_data.atsi_peak_curve_dates, ds_data.atsi_peak_curve_vals)
           , 'ATSI_7X8':           (ds_data.atsi_7x8_curve_dates, ds_data.atsi_7x8_curve_vals)
           , 'ATSI_2X16':          (ds_data.atsi_2x16_curve_dates, ds_data.atsi_2x16_curve_vals)
           , 'NG_MICHCON_GD-PEAK': (ds_data.ng_michcon_gd_peak_dates, ds_data.ng_michcon_gd_peak_curve_vals)
           , 'NG_MICHCON_CASHVOL': (ds_data.ng_michcon_cv_curve_dates, ds_data.ng_michcon_cv_curve_vals)
           , 'PJMW-OFFPEAK_CV':    (ds_data.pjm_offpeak_cv_curve_dates, ds_data.pjm_offpeak_cv_curve_vals)
           , 'PJMW-PEAK_CV':       (ds_data.pjm_peak_cv_curve_dates, ds_data.pjm_peak_cv_curve_vals)
           , 'DISCOUNT':           (ds_data.discount_curve_dates, ds_data.discount_curve_vals) }


def convert_str_datetime(date_):
    """
    Converts yyyymmdd into datetime.

    :param date_: date in above format, or a list of these formats.
    :type date_: str or list[str]
    :returns: same date in a different format
    :rtype: datetime.date or list[datetime.date]
    """

    def conv_local(d_elt):
        return datetime.date(int(d_elt[0:4]),
                       int(d_elt[4:6]),
                       int(d_elt[6:8]))

    if type(date_) is list:
        return [conv_local(d_elt) for d_elt in date_]
    else:
        return conv_local(date_)

    
def convert_str_dateslash(date_):
    """
    Converts yyyymmdd into mm/dd/yyyy

    :param date_: date in the format
    :type date_: str
    :returns: date in the different format
    :rtype: str
    """

    def conv_local(d_elt):
        return str(int(d_elt[4:6])) + '/' + str(int(d_elt[6:8])) + '/' + str(int(d_elt[0:4]))

    if type(date_) is list:
        return [conv_local(d_elt) for d_elt in date_]
    else:
        return conv_local(date_)
    

def convert_str_date(date_):
    """
    converts yyyymmdd into datetime
    """
    def conv_local(d_elt):
        return datetime.date(int(d_elt[0:4]),
                       int(d_elt[4:6]),
                       int(d_elt[6:8]))
    
    if type(date_) is list:
        return [conv_local(d_elt) for d_elt in date_]
    else:
        return conv_local(date_)


def d2s(i):
    """
    digit to string conversion, adding 0 if < 10

    """

    return "0" + str(i) if i < 10 else str(i)


def convert_datetime_str(date_):
    """
    Converts the date in datetime format into string format.

    """

    return str(date_.year) + d2s(date_.month) + d2s(date_.day)


def convert_dt_minus(date_):
    return str(date_.year) + '-' + d2s(date_.month) + '-' + d2s(date_.day)


def convert_dateslash_str(dates):
    """
    converts date in form 10/5/2016 -> 20161005
    """
    mon, day, year = dates.split('/')
    return year + d2s(int(mon)) + d2s(int(day))    


def convert_datedash_str(dates):
    """
    converts date in form 10-5-2016 -> 20161005
    """
    mon, day, year = dates.split('-')
    return year + d2s(int(mon)) + d2s(int(day))    


def convert_datedash_date(dates):
    """
    converts date in form 2016-10-5 -> datetime.date(..)
    """

    year, mon, day = dates.split('-')
    return datetime.date(int(year), int(mon), int(day))


def convert_datedash_time_dt(date_i, hour_i):
    """
    returns 
    """
    year, mon, day = date_i.split('-')
    hour, minutes, sec = hour_i.split(':')

    return datetime.datetime(int(year), int(mon), int(day), int(hour), int(minutes))


def convert_hour_time(hour):
    """
    converts date in form 12:00:02 -> datetime.time(..)

    """

    hour, minute, sec = hour.split(':')
    return datetime.time(int(hour), int(minute), int(sec))


def convert_dateslash_dash(dates):
    """
    Converts date in form 10/5/2016 -> 2016-10-05
    """

    mon, day, year = dates.split('/')
    return year + '-' + d2s(int(mon)) + '-' + d2s(int(day))    


def construct_date_range(date_b, date_e):
    """
    Constructs the date range between date_b and date_e

    :param date_b: begin date, in string format
    :param date_e: end date, in string format
    """

    date_b_dt = convert_str_date(date_b)
    date_e_dt = convert_str_date(date_e)
    year_b, month_b, day_b = date_b_dt.year, date_b_dt.month, date_b_dt.day
    year_e, month_e, day_e = date_e_dt.year, date_e_dt.month, date_e_dt.day

    T_l = []  # construction of the date list

    def process_mm(m, year, month, day_b, day_e):
        """
        process the month matrix between day_b and day_e
        """
        T_l = []
        for row in m:
            for day in row:
                if day >= day_b and day <= day_e:
                    T_l.append(datetime.date(year, month, day))
        return T_l
        
    for year in range(year_b, year_e+1):
        if year == year_b:
            if year_e == year_b:
                month_ends = month_e
            else:
                month_ends = 12
            for month in range(month_b, month_ends+1):
                mm = cal.monthcalendar(year, month)  # month matrix
                if month_b == month_e:
                    T_l.extend(process_mm(mm, year, month, day_b, day_e))
                elif month == month_b:
                    T_l.extend(process_mm(mm, year, month, day_b, 31))
                elif month == month_e:
                    T_l.extend(process_mm(mm, year, month, 1, day_e))
                else:
                    T_l.extend(process_mm(mm, year, month, 1, 31))
        elif year == year_e:
            for month in range(1, month_e+1):
                mm = cal.monthcalendar(year, month)  # month matrix
                if month == month_e:
                    T_l.extend(process_mm(mm, year, month, 1, day_e))
                else:
                    T_l.extend(process_mm(mm, year, month, 1, 31))
        else:
            for month in range(1, 13):  # all months 
                mm = cal.monthcalendar(year, month)  # month matrix
                T_l.extend(process_mm(mm, year, month, 1, 31))

    return T_l


def get_forward_curve( comName: str
                     , mktDate : datetime.date):
    """
    Gets the forward curve (ALWAYS USE DATE 20150410) for a particular date.

    :param comName: name of the commodity curve
    :param mktDate: market date for which the com curve is needed (not used here, but the interface should be such)
    """

    return fwd_hash[comName]


def get_forward_curve_slice(fwd, date_, date_b, date_e,
                            adj_tenors_days=0):
    """
    returns the slice between date_b, date_e, both are in string formats
    returns curve in [(date, value, com_coda)
    """

    fc1 = get_forward_curve(fwd, date_)
    fc_tenors = get_forward_curve_pretty2(fwd, date_)[1]
    date_b_dt = convert_str_datetime(date_b)
    date_e_dt = convert_str_datetime(date_e)
    adj_days_dt = datetime.timedelta(days=adj_tenors_days)

    if type(date_b) is list:
        return [[(fc1[0][k] - adj_days_dt, fc1[1][k], fc_tenors[k])
                for k in range(len(fc1[0]))
                if (fc1[0][k] >= date_b_dt_elt + adj_days_dt)
                 and (fc1[0][k] <= date_e_dt_elt + adj_days_dt)]
                for (date_b_dt_elt, date_e_dt_elt) in zip(date_b_dt, date_e_dt)]
    else:
        return [(fc1[0][k]-adj_days_dt, fc1[1][k], fc_tenors[k])
                for k in range(len(fc1[0]))
                if (fc1[0][k] >= date_b_dt + adj_days_dt)
                and (fc1[0][k] <= date_e_dt + adj_days_dt)]


def get_forward_curve_pretty2(fwd, date_):
    tenors, curve = get_forward_curve(fwd, date_)
    tenors_codes = [str(fwd_mth_codes[tenor.month-1]) + str(tenor.year-2000)
                    for tenor in tenors]
    return dict(zip(tenors_codes, curve)), tenors_codes


def get_forward_curve_plot(fwd, date_):
    tenors, curve = get_forward_curve(fwd, date_)
    curve_len = len(tenors)
    plt.plot(curve)
    xtics = []
    for i1 in range(curve_len):
        if np.mod(i1, 50) == 0:
            xtics.append(str(tenors[i1].year) + '-' + str(tenors[i1].month))
        else:
            xtics.append('')

    plt.xticks(range(len(xtics)), xtics, size='small', rotation='vertical')
    plt.show()


def getVolCurve( comName : str
               , mktDate : datetime.date):
    """
    Gets the vol curve comName for a particular market date. In out example, we don't use the second parameter.

    """

    volType, volDates, volCurve, volCurveWithDates = vol_hash[comName]

    return { 'vol_name'            : comName
           , 'vol_type'            : volType
           , 'vol_dates'           : volDates
           , 'vol_curve'           : volCurve
           , 'vol_curve_with_dates': volCurveWithDates }


def get_fwd_vol_curve_numeric_tenor( curveName : str
                                   , mktDate   : datetime.date
                                   , fwd_vol_ind     = 'fwd'
                                   , adj_tenors_days = None ):
    """
    Gets the raw forward or vol curve name.

    :param curveName: forward or vol curve name.
    :param mktDate: market date.
    :param fwd_vol_ind: indicator of forward or vol curve
    :param adj_tenors_days: integer, to adjust the number of days in the forward/vol curve.

    """

    if fwd_vol_ind is 'fwd':
        fwd_vol_tenors_raw, fwd_vol_values_raw = get_forward_curve(curveName, mktDate)
        if adj_tenors_days is not None:
            fwd_vol_tenors_vals = [(ot - datetime.timedelta(days=adj_tenors_days), val)
                                   for ot, val in zip(fwd_vol_tenors_raw, fwd_vol_values_raw)
                                   if ot - datetime.timedelta(days=adj_tenors_days) > mktDate ]
            fwd_vol_tenors, fwd_vol_values = zip(*fwd_vol_tenors_vals)
        else:
            fwd_vol_tenors, fwd_vol_values = fwd_vol_tenors_raw, fwd_vol_values_raw

    else:
        fwd_vol_tenors_raw, fwd_vol_values_raw = getVolCurve(curveName, mktDate)
        if adj_tenors_days is not None:
            fwd_vol_tenors_vals = [(ot - datetime.timedelta(days=adj_tenors_days), val)
                                   for ot, val in zip(fwd_vol_tenors_raw, fwd_vol_values_raw)
                                   if ot - datetime.timedelta(days=adj_tenors_days) > mktDate ]
            fwd_vol_tenors, fwd_vol_values = zip(*fwd_vol_tenors_vals)
        else:
            fwd_vol_tenors, fwd_vol_values = fwd_vol_tenors_raw, fwd_vol_values_raw

    diffs = [ten_ - mktDate for ten_ in fwd_vol_tenors if ten_ > mktDate ]
    fwd_vol_tenors_numeric = np.array([elt.days for elt in diffs])/365.
    fwd_vol_tenors_code = [(fwt.month, fwt.year) for fwt in fwd_vol_tenors if fwt > mktDate ]
    fwd_vol_values_unexpired = np.array([fwd_vol_vals for fwt, fwd_vol_vals
                                         in zip(fwd_vol_tenors, fwd_vol_values)
                                         if fwt > mktDate])

    return fwd_vol_tenors_numeric, fwd_vol_values_unexpired, fwd_vol_tenors_code, fwd_vol_tenors


def read_data_matched_tenors(mktDate : datetime.date
                             , fwdCurveName : str
                             , volCurveName : str
                             , adj_fwd_tenors_days = None
                             , adj_vol_tenors_days = None):

    """
    Matches the tenors of the forward and volatility curves.

    :param mktDate: market date for which the curves are obtained.
    :param fwdCurveName: forward curve
    """

    fwd_tenors, fwd_curve, fwd_tenors_code, fwd_tenors_dt = \
        get_fwd_vol_curve_numeric_tenor(fwdCurveName
                                        , mktDate
                                        , fwd_vol_ind     = 'fwd'
                                        , adj_tenors_days = adj_fwd_tenors_days)

    option_tenors, vol_params, option_tenors_code, option_tenors_dt_orig = \
        get_fwd_vol_curve_numeric_tenor( volCurveName
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
        if fwd_opt_ind is 'fwd':
            return [arr[elt_fwd] for (elt_fwd, elt_opt) in idx]
        else:
            return [arr[elt_opt] for (elt_fwd, elt_opt) in idx]

    fwd_tenors_matched = select_elts(fwd_tenors, match_idx, 'fwd')
    fwd_tenors_code_matched = select_elts(fwd_tenors_code, match_idx, 'fwd')
    fwd_tenors_dt_matched = select_elts(fwd_tenors_dt, match_idx, 'fwd')
    fwd_curve_matched = select_elts(fwdCurveName, match_idx, 'fwd')
    option_tenors_dt_matched = select_elts(option_tenors_dt, match_idx, 'opt')
    option_tenors_matched = select_elts(option_tenors, match_idx, 'opt')
    option_tenors_code_matched = select_elts(option_tenors_code, match_idx, 'opt')
    vol_surface_params_matched = select_elts(vol_params, match_idx, 'opt')

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

    return {'fwd_tenors'        : np.array(fwd_tenors_final),
            'fwd_curve'         : np.array(fwd_curve_final),
            'fwd_tenors_code'   : np.array(fwd_tenors_code_final),
            'fwd_tenors_dt'     : fwd_tenors_dt_final,
            'option_tenors'     : np.array(option_tenors_final),
            'option_tenors_code': np.array(option_tenors_code_final),
            'option_tenors_dt'  : option_tenors_dt_final,
            'vol_surface_params': np.array(vol_surface_params_final)}
