from sqlalchemy import Column, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class YourModel(Base):
    __tablename__ = 'your_model'

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    embedding = Column(Float, nullable=False)  # Adjust type as needed for pgvector

    def __repr__(self):
        return f"<YourModel(id={self.id}, name={self.name})>"