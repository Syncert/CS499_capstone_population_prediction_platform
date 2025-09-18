from __future__ import annotations
from sqlalchemy import create_engine, text, String, Integer, BigInteger, Numeric, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from .settings import build_database_url

# Optional: load a local .env when running on host (no-op in Docker)
try:
    from dotenv import load_dotenv  # part of your base deps
    load_dotenv(override=False)
except Exception:
    pass

# --- SQLAlchemy base/models --------------------------------------------------

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
    etl_batch_id: Mapped[str] = mapped_column(String)  # name matches your schema


DATABASE_URL = build_database_url()

# Engine & Session (pool_pre_ping avoids stale sockets)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def ping_db() -> bool:
    with engine.connect() as conn:
        return conn.execute(text("SELECT 1")).scalar() == 1