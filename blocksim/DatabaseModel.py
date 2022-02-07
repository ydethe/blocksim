from datetime import datetime

import numpy as np
from sqlalchemy import (
    Column,
    BigInteger,
    Integer,
    Float,
    Boolean,
    String,
    DateTime,
    ForeignKey,
    create_engine,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from psycopg2.extensions import register_adapter, AsIs

register_adapter(np.int64, AsIs)

# https://www.pythoncentral.io/introductory-tutorial-python-sqlalchemy/
# https://docs.sqlalchemy.org/en/14/core/type_basics.html#sqlalchemy.types.ARRAY
Base = declarative_base()


class Simulation(Base):
    __tablename__ = "simulation"
    # Here we define columns for the table person
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    program = Column(String(250), nullable=False)
    arguments = Column(String(1024))
    node = Column(String(1024))
    user = Column(String(1024))
    cwd = Column(String(2048))

    def __repr__(self):
        if self.arguments == "":
            cmd = self.program
        else:
            cmd = "%s %s" % (self.program, self.arguments)
        dtime = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        s = "[%i](%s) by %s@%s in '%s', run %s" % (
            self.id,
            cmd,
            self.user,
            self.node,
            self.cwd,
            dtime,
        )
        return s


class IntegerSerie(Base):
    __tablename__ = "integer_serie"
    # Here we define columns for the table address.
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True)
    name = Column(String(250), nullable=False)
    unit = Column(String(250))
    data = Column(ARRAY(BigInteger))
    simulation_id = Column(Integer, ForeignKey("simulation.id"))
    simulation = relationship(Simulation)


class FloatSerie(Base):
    __tablename__ = "float_serie"
    # Here we define columns for the table address.
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True)
    name = Column(String(250), nullable=False)
    unit = Column(String(250))
    data = Column(ARRAY(Float))
    simulation_id = Column(Integer, ForeignKey("simulation.id"))
    simulation = relationship(Simulation)


class ComplexSerie(Base):
    __tablename__ = "complex_serie"
    # Here we define columns for the table address.
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True)
    name = Column(String(250), nullable=False)
    unit = Column(String(250))
    data = Column(ARRAY(Float, dimensions=2))
    simulation_id = Column(Integer, ForeignKey("simulation.id"))
    simulation = relationship(Simulation)


class BoolSerie(Base):
    __tablename__ = "bool_serie"
    # Here we define columns for the table address.
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True)
    name = Column(String(250), nullable=False)
    unit = Column(String(250))
    data = Column(ARRAY(Boolean))
    simulation_id = Column(Integer, ForeignKey("simulation.id"))
    simulation = relationship(Simulation)
