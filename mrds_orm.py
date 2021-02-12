# Com Skew w/ ORM


from sqlalchemy.ext.declarative import declarative_base
from mrds.mrds import ComSkew

ComORM = declarative_base()  # default sqlalchemy class


class ComSkewORM(ComSkew, ComORM):
    """ Commodity skew model with ORM baked in.
    """

    # sqlalchemy part
    from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, BigInteger, Table, Float, SmallInteger, Enum, \
        create_engine
    from sqlalchemy.orm import relation, sessionmaker

    __tablename__ = 'com_skew'

    object_id = Column(String)
    commodity = Column(String)
