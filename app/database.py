# app/database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models import Base

DATABASE_URL = settings.database_url

engine = create_engine(
    DATABASE_URL,
    echo=False,
)

SessionLocal = sessionmaker(
    bind=engine,
)


def init_db():
    Base.metadata.create_all(engine)
