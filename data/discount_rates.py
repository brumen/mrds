# sample data for discount curve

import datetime
import QuantLib as ql

# discount curves basic
discount_curve_dates = []
for year in range(2015, 2027):  # appx 10 years
    for month in range(1, 13):
        discount_curve_dates.append(datetime.date(year, month, 1))

discount_curve_vals = [0.003] * len(discount_curve_dates)


# discount curve quantlib
discount_curve_ois_rates = [ (0.002, (15, ql.Months))
                           , (0.008, (18, ql.Months))
                           , (0.021, (21, ql.Months))
                           , (0.036, (2,  ql.Years))
                           , (0.127, (3,  ql.Years))
                           , (0.274, (4,  ql.Years))
                           , (0.456, (5,  ql.Years))
                           , (0.647, (6,  ql.Years))
                           , (0.827, (7,  ql.Years))
                           , (0.996, (8,  ql.Years))
                           , (1.147, (9,  ql.Years))
                           , (1.280, (10, ql.Years))
                           , (1.404, (11, ql.Years))
                           , (1.516, (12, ql.Years))
                           , (1.764, (15, ql.Years))
                           , (1.939, (20, ql.Years))
                           , (2.003, (25, ql.Years))
                           , (2.038, (30, ql.Years))]
