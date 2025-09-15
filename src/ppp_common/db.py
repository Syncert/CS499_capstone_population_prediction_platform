from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from .settings import load_settings

_engine: Engine | None = None

def get_engine() -> Engine:
    global _engine
    if _engine is None:
        settings = load_settings()
        _engine = create_engine(settings.database_url, pool_pre_ping=True)
    return _engine
