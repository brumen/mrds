# Gets the appropriate volatility for the commodity considered.

import datetime
import logging

from mrds.ds        import vol_hash
from mrds.vols.vols import Volatility

logger = logging.getLogger(__name__)


def get_vol_object(com_name : str, mkt_date : datetime.date) -> Volatility:
    """ Gets the vol object for the commodity in question for the market date mkt_date

    :param com_name: commodity name for the vol object
    :param mkt_date: market date
    :returns: volatility object for commodity & market date.
    """

    logger.info('Getting volatility for commodity {0}'.format(com_name))

    vol_type, _, _, _ = vol_hash[com_name]

    if vol_type == 'JWSS7':
        from mrds.vols.jwss7 import JWSS7Volatility
        vol_class = JWSS7Volatility
    elif vol_type == 'ATM':
        from mrds.vols.vols import ATMFVolatility
        vol_class = ATMFVolatility
    elif vol_type == 'C0C1C2':
        from mrds.vols.quadratic import QuadraticVol
        vol_class = QuadraticVol
    else:
        raise RuntimeError('Volatility class for {0} not yet implemented'.format(com_name))

    return vol_class.from_db(com_name, mkt_date)
