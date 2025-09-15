from pydantic import BaseModel
import os

class Settings(BaseModel):
    database_url: str

def load_settings() -> Settings:
    # optional: load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    return Settings(database_url=os.getenv("DATABASE_URL", ""))
