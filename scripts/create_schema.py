import os
from sqlalchemy import create_engine
from app.db import Base
from app import models  # registers tables
e = create_engine(os.environ["DATABASE_URL"], future=True)
Base.metadata.create_all(e)
print("schema ensured")
