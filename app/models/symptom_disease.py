from sqlalchemy import Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class SymptomDisease(Base):
    __tablename__ = "symptom_disease"

    id = Column(Integer, primary_key=True, index=True)
    symptom = Column(String, unique=True, index=True)
    disease = Column(String)
    department = Column(String)