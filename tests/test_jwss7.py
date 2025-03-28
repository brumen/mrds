import datetime
import matplotlib.pyplot as plt
import numpy as np

from mrds.forward_curve import FwdCurve
from mrds.vols.jwss7 import JWSS7Volatility, JWSS7VolatilityDisplay


S0 = 100.
K = 110.
ttm = 2.

sigma_0 = 0.2
A = 1.
B = 1.
C = 1.
P = 1.
alpha_C = 1.
alpha_P = 1.

# TODO: REALISTIC VALUES.
sigma_0 = 0.41
skew = 2.9
smile = 15.2
call_slope = -6.4
put_slope = -13.
call_bend = -12
put_bend = -5.8

jwss7_params = (sigma_0, skew, smile, put_slope, put_bend, call_slope, call_bend)

v1 = JWSS7Volatility._vol_from_jwss7(S0, K, ttm, jwss7_params)
print(v1)


# draw a line for the params.

plt.style.use('_mpl-gallery')

# make data
#K_v = np.linspace(80., 120., 41)
#log_strike = np.array([JWSS7Volatility.normalized_strike(S0, K, sigma_0, ttm) for K in K_v])
#y = np.array([JWSS7Volatility._vol_from_jwss7(S0, K, ttm, jwss7_params)
#              for S0 in K_v])

# plot
# fig, ax = plt.subplots()

#plt.plot(log_strike, y, linewidth=2.0)

#ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#       ylim=(0, 8), yticks=np.arange(1, 8))

# plt.show()


# JWSSS7 example
MKT_DATE = datetime.date(2025, 3, 27)
fwd_curve = FwdCurve(
    MKT_DATE,
    fwd_name='AAPL',
    fwd_tenors=[datetime.date(2025, 5, 1), datetime.date(2025, 6, 1), datetime.date(2025, 7, 1)],
    fwd_values=[100., 110., 120.],
)
vol_params = {
    datetime.date(2025, 5, 1): (sigma_0, skew, smile, call_slope, put_slope, call_bend, put_bend)
}

jwss7_vol = JWSS7Volatility(
    'AAPL',
    MKT_DATE,
    fwd_params=fwd_curve,
    vol_params=vol_params,
)


jwss7_vol = JWSS7VolatilityDisplay(
    'AAPL',
    MKT_DATE,
    fwd_params=fwd_curve,
    vol_params=vol_params,
)

jwss7_vol.create_plot()
