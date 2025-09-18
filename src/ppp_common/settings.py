# src/ppp_common/settings.py
from __future__ import annotations
import os
from urllib.parse import quote_plus
from pydantic import BaseModel

class Settings(BaseModel):
    database_url: str

def build_database_url() -> str:
    """
    Precedence:
      1) DATABASE_URL (use as-is)
      2) Compose from POSTGRES_* parts + DB_HOST (default 'localhost')
         – in containers, docker-compose sets DB_HOST=db.
    """
    if url := os.getenv("DATABASE_URL"):
        return url

    user = os.getenv("POSTGRES_USER", "postgres")
    pwd  = quote_plus(os.getenv("POSTGRES_PASSWORD", "postgres"))
    db   = os.getenv("POSTGRES_DB", "postgres")
    host = os.getenv("DB_HOST", "localhost")       # host default
    port = os.getenv("POSTGRES_PORT", "5432")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"

def load_settings() -> Settings:
    # Load .env on host only; don't override existing env (Compose)
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except Exception:
        pass
    return Settings(database_url=build_database_url())