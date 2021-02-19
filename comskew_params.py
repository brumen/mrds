# flight class for ORM, trade access

import pickle

from sqlalchemy                 import ( Column
                                       , String
                                       , Date
                                       , BigInteger
                                       , Enum
                                       , LargeBinary
                                       , )
from sqlalchemy.ext.declarative import declarative_base

from ao.flight   import create_session
from mrds.config import brumen_pass

DB = f'postgres://brumen:{brumen_pass}@localhost:5434/com_skew'

DEFAULT_SESSION = create_session(db=DB)

ComSkewParamsBaseORM = declarative_base()  # common base class


class ComSkewLnParams(ComSkewParamsBaseORM):
    """ Commodity skew lognormal parameters table.
    """

    __tablename__ = 'params'

    id          = Column(BigInteger, primary_key=True)  # serial, primary key
    market_date = Column(Date)
    commodity   = Column(String)
    value       = Column(LargeBinary)
    param       = Column(Enum('sigma', 'kappa', 'rho', ))

    @property
    def val(self):
        return pickle.loads(self.value)

    @val.setter
    def val(self, new_val):
        self.value = pickle.dumps(new_val)


class ComSkewCParams(ComSkewParamsBaseORM):
    """ Commodity skew C parameters table.
    """

    __tablename__ = 'params_C'

    id          = Column(BigInteger, primary_key=True)  # serial, primary key
    market_date = Column(Date)
    commodity   = Column(String)
    value       = Column(LargeBinary)
    # param       = Column(Enum('sigma', 'kappa', 'rho', ))
    fwd_date    = Column(Date)

    @property
    def val(self):
        return pickle.loads(self.value)

    @val.setter
    def val(self, new_val):
        self.value = pickle.dumps(new_val)


# Example of usage
# p1 = ComSkewParams(market_date=datetime.date(2015, 4, 1), commodity='WTI', param='sigma', value=json.dumps({'a': 1}))
# DEFAULT_SESSION.add(p1)
# DEFAULT_SESSION.commit()

# typical query which implements AOTrade:
# select fid.flight_id_long, fo.price
# from flight_ids fid, option_positions op, trades_flights tf, flights_ord fo
# where op.origin = 'SFO' and op.dest = 'EWR'
#      and op.position_id = tf.trade_id
#      and tf.flight_id = fid.flight_id
#      and fo.flight_id = tf.flight_id
