# Com Skew w/ ORM

import datetime

from typing import Tuple, List
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, BigInteger, Table, Float, SmallInteger, Enum, \
    create_engine
from sqlalchemy.orm import relation, sessionmaker

from mrds.mrds import ComSkew

ComORM = declarative_base()  # default sqlalchemy class


class ComSkewORM(ComSkew, ComORM):
    """ Commodity skew model with ORM baked in.
    """

    __tablename__ = 'com_skew'

    object_id = Column(String)
    commodity = Column(String)

    def object_identifier(self) -> Tuple[datetime.date, List[str]]:
        """ Object identifier for the purposes of pickling. returns a tuple which can be used for storing objects.

        :returns: tuple which identifies the curve - market date of the curve, followed by a list of curves stored in the object.
        """

        return self.mkt_date, [fwd_curve.fwd_name for fwd_curve in self.fwd_curves]

