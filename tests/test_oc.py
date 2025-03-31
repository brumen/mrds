""" Tests the option chain from yahoo finance.
"""

import datetime
import yfinance as yf

from itertools import chain
from mrds.vols.jwss7 import JWSS7Volatility, CallPut


def calibrate_date(
        mkt_date: datetime.date,
        ticker: str,
        expiry: datetime.date,
        r: float = 0.05,  # TODO:
):
    """

    """

    expiry_s = datetime.date.strftime(expiry, '%Y-%m-%d')
    ticker_obj = yf.Ticker(ticker)
    bid = ticker_obj.info['bid']
    ask = ticker_obj.info['ask']
    mid = (bid + ask)/2
    ticker_chain = ticker_obj.option_chain(expiry_s)

    ticker_calls_df = ticker_chain.calls
    ticker_puts_df = ticker_chain.puts

    nb_calls = len(ticker_calls_df)
    nb_puts = len(ticker_puts_df)
    print('NB CALLS', nb_calls)
    print('NB PUTS', nb_puts)
    ticker_calls_prices_strikes = list(zip(
        ticker_calls_df['lastPrice'],
        ticker_calls_df['strike'],
        [CallPut.Call]*nb_calls,
    ))
    ticker_put_prices_strikes = list(zip(
        ticker_puts_df['lastPrice'],
        ticker_calls_df['strike'],
        [CallPut.Put]*nb_puts,
    ))

    prices_strikes_cp = ticker_calls_prices_strikes + ticker_put_prices_strikes

    # print("TOOTAL", len(list(prices_strikes_cp)))

    # calibrate the options
    jw_calib = JWSS7Volatility.calibrate_params(
        mkt_date,
        mid,
        prices_strikes_cp,
        expiry,
        r,
    )

    return jw_calib

    ticker_calls_prices_strikes2 = list(zip(
        ticker_calls_df['lastPrice'],
        ticker_calls_df['strike'],
        ["call"]*nb_calls,
    ))
    ticker_put_prices_strikes2 = list(zip(
        ticker_puts_df['lastPrice'],
        ticker_calls_df['strike'],
        ["put"]*nb_puts,
    ))

    prices_strikes_cp2 = ticker_calls_prices_strikes2 + ticker_put_prices_strikes2

    jw_calib2 = JWSS7Volatility.calibrate_params_jw7(
        mkt_date,
        mid,
        prices_strikes_cp2,
        expiry,
        r,
    )

    return jw_calib2


def _calibrate_all():
    msft_ticker = yf.Ticker('MSFT')
    msft_options = msft_ticker.options
    msft_expiries = [
        datetime.datetime.strptime(option_expiry, "%Y-%m-%d").date()
        for option_expiry in msft_options
    ]

    for expiry in msft_expiries:
        print("CALIBRATING EXPIRY", expiry)
        res1 = calibrate_date(datetime.date(2025, 3, 28), 'MSFT', expiry)
        print(f'Result {expiry}: {res1}')


# _calibrate_all()
