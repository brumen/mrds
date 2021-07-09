#
import calendar
import datetime

from typing import List


def process_mm(m : int, year : int, month : int, day_b : int, day_e : int) -> List[datetime.date]:
    """ process the month matrix between day_b and day_e

    :param m: TODO: WHAT IS THIS
    :param year: year
    :param month: month
    :param day_b: beginning day
    :param day_e: end day
    :returns:
    """

    T_l = []
    for row in m:
        for day in row:
            if day_b <= day <= day_e:
                T_l.append(datetime.date(year, month, day))
    return T_l


def construct_date_range( date_b : datetime.date
                        , date_e : datetime.date ) -> List[datetime.date]:
    """ Constructs the date range between date_b and date_e

    :param date_b: begin date
    :param date_e: end date
    :returns: dates between beginning and the end.
    """

    year_b, month_b, day_b = date_b.year, date_b.month, date_b.day
    year_e, month_e, day_e = date_e.year, date_e.month, date_e.day

    T_l = []  # construction of the date list
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
