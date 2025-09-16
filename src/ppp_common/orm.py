from __future__ import annotations
from dataclasses import dataclass
import os
from sqlalchemy import String, Integer, BigInteger, Numeric, text, create_engine, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

# Use DATABASE_URL from .env (inside containers points to host 'db')
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres")

class Base(DeclarativeBase):
    pass

class Geography(Base):
    __tablename__ = "geography"
    __table_args__ = {"schema": "core"}
    geo_code: Mapped[str] = mapped_column(String, primary_key=True)
    geo_name: Mapped[str] = mapped_column(String)
    geo_type: Mapped[str] = mapped_column(String)

    populations: Mapped[list["PopulationObservation"]] = relationship(back_populates="geo")

class PopulationObservation(Base):
    __tablename__ = "population_observations"
    __table_args__ = {"schema": "core"}
    geo_code: Mapped[str] = mapped_column(ForeignKey("core.geography.geo_code"), primary_key=True)
    year: Mapped[int] = mapped_column(Integer, primary_key=True)
    population: Mapped[int] = mapped_column(BigInteger)
    geo: Mapped[Geography] = relationship(back_populates="populations")

class IndicatorValue(Base):
    __tablename__ = "indicator_values"
    __table_args__ = {"schema": "core"}
    geo_code: Mapped[str] = mapped_column(ForeignKey("core.geography.geo_code"), primary_key=True)
    year: Mapped[int] = mapped_column(Integer, primary_key=True)
    indicator_code: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[float] = mapped_column(Numeric)
    source: Mapped[str] = mapped_column(String)
    unit: Mapped[str] = mapped_column(String)
    batch_id: Mapped[str] = mapped_column(String)

# Engine & Session
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def ping_db() -> bool:
    with engine.connect() as conn:
        return conn.execute(text("SELECT 1")).scalar() == 1