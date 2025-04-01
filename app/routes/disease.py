from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.services.database import get_db
from app.models.symptom_disease import SymptomDisease

router = APIRouter()

@router.post("/disease_info")
def get_disease_info(symptoms: list[str], db: Session = Depends(get_db)):
    results = []
    for symptom in symptoms:
        record = db.query(SymptomDisease).filter(SymptomDisease.symptom == symptom).first()
        if record:
            results.append({"증상": symptom, "질병": record.disease, "분과": record.department})
        else:
            results.append({"증상": symptom, "질병": "알 수 없음", "분과": "일반의"})
    return results