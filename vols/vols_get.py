# Gets the appropriate volatility for the commodity considered.

import datetime
import logging

from ds          import vol_hash
from vols.vols   import Volatility

logger = logging.getLogger(__name__)


def get_vol_object(com_name : str, mkt_date : datetime.date) -> Volatility:
    """ Gets the vol object for the commodity in question for the market date mkt_date

    :param com_name: commodity name for the vol object
    :param mkt_date: market date
    :param

    """

    logger.info('Getting volatility for commodity {0}'.format(com_name))

    vol_type, _, _, _ = vol_hash[com_name]

    if vol_type == 'JWSS7':
        from vols.jwss7 import JWSS7Volatility
        vol_class = JWSS7Volatility
    elif vol_type == 'ATM':
        from vols.vols import ATMFVolatility
        vol_class = ATMFVolatility
    elif vol_type == 'C0C1C2':
        from vols.c0c1c2 import C0C1C2Volatility
        vol_class = C0C1C2Volatility
    else:
        raise RuntimeError('Volatility class for {0} not yet implemented'.format(com_name))

    return vol_class.from_db(com_name, mkt_date)
