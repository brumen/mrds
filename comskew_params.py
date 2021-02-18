# flight class for ORM, trade access

import pickle

from sqlalchemy                 import ( Column
                                       , Integer
                                       , String
                                       , DateTime
                                       , Date
                                       , ForeignKey
                                       , BigInteger
                                       , Table
                                       , Float
                                       , SmallInteger
                                       , Enum
                                       , JSON
                                       , LargeBinary
                                       , )
from sqlalchemy.orm             import relation
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



# class Flight(AOORM):
#     """ Description of the flight.
#
#     flight_id: identifier of the flight in the db.
#     flight_id_long: flight identifier from skyscanner.
#     dep_date: departure date of the flight
#     orig: IATA code of the origin airport ('EWR')
#     dest: IATA code of the destination airport ('SFO')
#     carrier: airline carrier, e.g. 'UA'
#     """
#
#     __tablename__ = 'flight_ids'
#
#     flight_id      = Column(Integer, primary_key=True)
#     flight_id_long = Column(String)
#     orig           = Column(String)
#     dest           = Column(String)
#     dep_date       = Column(DateTime)
#     arr_date       = Column(DateTime)
#     carrier        = Column(String)
#
#
# # in between table that links Flight and AOTrade.
# t_trades_flights = Table( 'trades_flights'
#                         , AOORM.metadata
#                         , Column('entry_id' , BigInteger, primary_key=True)
#                         , Column('trade_id' , Integer   , ForeignKey('option_positions.position_id'))
#                         , Column('flight_id', Integer   , ForeignKey('flight_ids.flight_id')        )
#                         , )
#
#
# class AOTrade(AOORM):
#     """ AirOptions trade.
#     """
#
#     __tablename__ = 'option_positions'
#
#     position_id           = Column(Integer, primary_key=True)
#     origin                = Column(String)
#     dest                  = Column(String)
#     option_start_date     = Column(DateTime)
#     option_end_date       = Column(DateTime)
#     option_ret_start_date = Column(DateTime)
#     option_ret_end_date   = Column(DateTime)
#     outbound_date_start   = Column(DateTime)
#     outbound_date_end     = Column(DateTime)
#     inbound_date_start    = Column(DateTime)
#     inbound_date_end      = Column(DateTime)
#     strike                = Column(Float)
#     carrier               = Column(String)
#     nb_adults             = Column(Integer)
#     cabinclass            = Column(String)
#
#     flights = relation('Flight', secondary=t_trades_flights)
#
#
# class AORegIds(AOORM):
#     """ Region Id table reference.
#     """
#
#     __tablename__ = 'reg_ids'
#
#     reg_id      = Column(Integer, primary_key=True)
#     month       = Column(SmallInteger)
#     tod         = Column(Enum('morning', 'afternoon', 'evening', 'night'))
#     weekday_ind = Column(Enum('weekday', 'weekend'))
#
#
# class AOParam(AOORM):
#     """ Parameters for ao flights, volatility, drift, etc.
#     """
#
#     __tablename__ = 'params'
#
#     param_id  = Column(BigInteger, primary_key=True)
#     as_of     = Column(DateTime)
#     orig      = Column(String(3))
#     dest      = Column(String(3))
#     carrier   = Column(String(2))
#     drift     = Column(Float)
#     vol       = Column(Float)
#     avg_price = Column(Float)
#     reg_id    = Column(BigInteger, ForeignKey('reg_ids.reg_id'))
#
#
#
#
# def select_random_flights( nb_flights : int = 10
#                          , db : str = 'mysql://brumen@localhost/ao'
#                          , session = DEFAULT_SESSION ):
#     """ Selects random flights from the database
#
#     """
#
#     rand_flights = set(np.random.randint(1, 1000, nb_flights))  # remove duplicates
#
#     return session.query(Flight).filter(Flight.flight_id.in_(rand_flights)).all()  # all flights
#
#
# def insert_random_flights(nb_positions : int = 10
#                          , nb_flights  : Optional[int] = 10
#                          , db          : str = 'mysql://brumen@localhost/ao'
#                          , session = DEFAULT_SESSION ):
#     """ Inserts number of positions in the database.
#
#     :param nb_positions: number of positions to be inserted in the database.
#     :param nb_flights: each position has this number of flights considered.
#     :param db: database where positions are inserted.
#     """
#
#     start_pos_id = session.query(AOTrade).count() + 1
#
#     trades = [ AOTrade( flights     = select_random_flights(nb_flights=nb_flights, db = db)
#                       , strike      = 200.
#                       , nb_adults   = 1
#                       , cabinclass  = 'Economy'
#                       , position_id = pos_id )
#                for pos_id in range(start_pos_id, start_pos_id + nb_positions) ]
#
#     for trade in trades:
#         session.add(trade)
#     session.commit()


# examples:
# res1 = sess.query(Flight)
# tr1 = sess.query(AOTrade).filter_by(position_id=2).all()[0]
# tr2 = tr1.flights
# print(tr2)

# typical query which implements AOTrade:
# select fid.flight_id_long, fo.price
# from flight_ids fid, option_positions op, trades_flights tf, flights_ord fo
# where op.origin = 'SFO' and op.dest = 'EWR'
#      and op.position_id = tf.trade_id
#      and tf.flight_id = fid.flight_id
#      and fo.flight_id = tf.flight_id
