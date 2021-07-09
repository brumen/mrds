# volatility

import datetime

from mrds.ds import get_forward_curve

wti2_dates, _ = get_forward_curve('WTI', datetime.date(2015, 4, 10))
wti2_vol = {wti2_date: (0.1, 0.2, 0.3, 0.4, 0.5)  # c0, c1, c2, theta, alpha
            for wti2_date in wti2_dates}
