#
import calendar
import datetime

# COMMENTED OUT TO BE REMOVED!!!

# def convert_str_datetime(date_) -> datetime.date:
#     """ Converts yyyymmdd into datetime.
#
#     :param date_: date in above format, or a list of these formats.
#     :type date_: str or list[str]
#     :returns: same date in a different format
#     :rtype: datetime.date or list[datetime.date]
#     """
#
#     def conv_local(d_elt):
#         return datetime.date(int(d_elt[0:4])
#                              , int(d_elt[4:6])
#                              , int(d_elt[6:8]))
#
#     if isinstance(date_, list):
#         return [conv_local(d_elt) for d_elt in date_]
#
#     return conv_local(date_)


# def convert_str_dateslash(date_):
#     """ Converts yyyymmdd into mm/dd/yyyy
#
#     :param date_: date in the format
#     :type date_: str
#     :returns: date in the different format
#     :rtype: str
#     """
#
#     def conv_local(d_elt):
#         return str(int(d_elt[4:6])) + '/' + str(int(d_elt[6:8])) + '/' + str(int(d_elt[0:4]))
#
#     if isinstance(date_, list):
#         return [conv_local(d_elt) for d_elt in date_]
#
#     return conv_local(date_)


# def convert_str_date(date_: str) -> datetime.date:
#     """ converts yyyymmdd into datetime.date.
#
#     :param date_: datetime in str format.
#     :returns: date in datetime.date format.
#     """
#
#     def conv_local(d_elt):
#         return datetime.date(int(d_elt[0:4])
#                              , int(d_elt[4:6])
#                              , int(d_elt[6:8]))
#
#     if isinstance(date_, list):
#         return [conv_local(d_elt) for d_elt in date_]
#
#     return conv_local(date_)


# def d2s(i):
#     """
#     digit to string conversion, adding 0 if < 10
#
#     """
#
#     return "0" + str(i) if i < 10 else str(i)


# def convert_datetime_str(date_):
#     """ Converts the date in datetime format into string format.
#
#     """
#
#     return str(date_.year) + d2s(date_.month) + d2s(date_.day)


# def convert_dt_minus(date_):
#     return str(date_.year) + '-' + d2s(date_.month) + '-' + d2s(date_.day)


# def convert_dateslash_str(dates):
#     """
#     converts date in form 10/5/2016 -> 20161005
#     """
#     mon, day, year = dates.split('/')
#     return year + d2s(int(mon)) + d2s(int(day))


# def convert_datedash_str(dates):
#     """
#     converts date in form 10-5-2016 -> 20161005
#     """
#     mon, day, year = dates.split('-')
#     return year + d2s(int(mon)) + d2s(int(day))


# def convert_datedash_date(dates):
#     """
#     converts date in form 2016-10-5 -> datetime.date(..)
#     """
#
#     year, mon, day = dates.split('-')
#     return datetime.date(int(year), int(mon), int(day))


# def convert_datedash_time_dt(date_i, hour_i):
#     """
#     returns
#     """
#     year, mon, day = date_i.split('-')
#     hour, minutes, sec = hour_i.split(':')
#
#     return datetime.datetime(int(year), int(mon), int(day), int(hour), int(minutes))


# def convert_hour_time(hour):
#     """
#     converts date in form 12:00:02 -> datetime.time(..)
#
#     """
#
#     hour, minute, sec = hour.split(':')
#     return datetime.time(int(hour), int(minute), int(sec))


# def convert_dateslash_dash(dates):
#     """
#     Converts date in form 10/5/2016 -> 2016-10-05
#     """
#
#     mon, day, year = dates.split('/')
#     return year + '-' + d2s(int(mon)) + '-' + d2s(int(day))


def construct_date_range( date_b_dt : datetime.date
                        , date_e_dt : datetime.date ):
    """ Constructs the date range between date_b and date_e

    :param date_b: begin date
    :param date_e: end date
    """

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

    for year in range(year_b, year_e + 1):
        if year == year_b:
            if year_e == year_b:
                month_ends = month_e
            else:
                month_ends = 12
            for month in range(month_b, month_ends + 1):
                mm = calendar.monthcalendar(year, month)  # month matrix
                if month_b == month_e:
                    T_l.extend(process_mm(mm, year, month, day_b, day_e))
                elif month == month_b:
                    T_l.extend(process_mm(mm, year, month, day_b, 31))
                elif month == month_e:
                    T_l.extend(process_mm(mm, year, month, 1, day_e))
                else:
                    T_l.extend(process_mm(mm, year, month, 1, 31))
        elif year == year_e:
            for month in range(1, month_e + 1):
                mm = calendar.monthcalendar(year, month)  # month matrix
                if month == month_e:
                    T_l.extend(process_mm(mm, year, month, 1, day_e))
                else:
                    T_l.extend(process_mm(mm, year, month, 1, 31))
        else:
            for month in range(1, 13):  # all months
                mm = calendar.monthcalendar(year, month)  # month matrix
                T_l.extend(process_mm(mm, year, month, 1, 31))

    return T_l
