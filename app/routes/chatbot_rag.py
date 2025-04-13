from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sqlalchemy.orm import Session
from app.services.database import get_db
from app.models.symptom_disease import SymptomDisease
from sqlalchemy import or_
import re
import torch

router = APIRouter()

MODEL_NAME = "beomi/KoAlpaca-Polyglot-5.8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to("mps")
model.eval()

class ChatRequest(BaseModel):
    prompt: str

@router.post("/hello-chat")
def generate_response(request: ChatRequest):
    chat_prompt = f"""다음은 AI와 사용자 간의 대화입니다.

    예시)
    사용자: 안녕?
    AI: 안녕하세요! 오늘 기분은 어떠신가요?

    사용자: 헬로우~
    AI:"""

    inputs = tokenizer(chat_prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to("mps") for k, v in inputs.items() if k != "token_type_ids"}

    output = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.9,
        top_k=30,
        top_p=0.9,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    response = decoded.split("AI:")[-1].strip()
    response = response.split("사용자:")[0].strip()
    response = response.replace("|", "").strip() 

    return {"response": response}

# 자연어 생성에 강함. 추론, 이해는 아직 부족 한 듯
@router.post("/predict/disease")
def generate_response(request: ChatRequest):
    chat_prompt = f"""너는 증상을 입력하면 해당 증상으로 의심되는 질병과 적절한 진료과를 추천해주는 의료 AI야.  
    가능한 경우 1개 이상의 질병을 추천하고, 그에 맞는 진료과도 함께 알려줘.  
    간단하고 이해하기 쉬운 설명을 포함해줘.  
    모르는 경우는 "정확한 진단은 어렵습니다. 가까운 병원을 방문해보세요."라고 안내해줘.

    예시:
    사용자: 머리가 아파요
    AI: 머리가 아픈 증상은 편두통, 긴장성 두통 등이 의심됩니다.  
    추천 진료과: 신경과

    사용자: 배가 아파요
    AI: 복통은 위염, 과민성대장증후군, 장염 등이 원인일 수 있습니다.  
    추천 진료과: 소화기내과

    사용자: {request}
    AI:"""

    inputs = tokenizer(chat_prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to("mps") for k, v in inputs.items() if k != "token_type_ids"}

    output = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.9,
        top_k=30,
        top_p=0.9,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    response = decoded.split("AI:")[-1].strip()
    response = response.split("사용자:")[0].strip()
    response = response.replace("|", "").strip() 

    return {"response": response}

@router.post("/v1/predict/disease-rag")
def recommend_disease(req: ChatRequest, db: Session = Depends(get_db)) -> dict:
    # 1) 증상 키워드 분리
    input_symptoms = [kw.strip() for kw in re.split(r"[ ,\.!?~\n]", req.prompt) if kw.strip()]
    print(input_symptoms);

    # 2) 조건 생성
    conditions = [SymptomDisease.symptom.ilike(f"%{symptom}%") for symptom in input_symptoms]
    print(conditions);

    # 3) OR 조건으로 질병 후보군 검색
    related_rows = db.query(SymptomDisease).filter(or_(*conditions)).all()
    print(related_rows);

    if not related_rows:
        return {"message": "해당 증상에 대한 질병 정보를 찾을 수 없습니다."}

    candidate_diseases = list({row.disease for row in related_rows})
    disease_str = ", ".join(candidate_diseases)


    # 2) 프롬프트 생성
    prompt = f"""
    너는 의학 전문가 AI야. 사용자 증상을 보고, 아래 후보 질병 중에서 가장 적절한 질병 하나만 골라줘.

    [사용자 증상]
    {req.prompt}

    [질병 후보]
    {disease_str}

    [답변 형식 - 아래처럼 정확히 따라야 해]
    가장 적절한 질병은: (질병명)

    예시:
    가장 적절한 질병은: 감기

    정답:
    가장 적절한 질병은:"""

    # 3) 토크나이즈 및 생성
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")
    output = model.generate(
        input_ids,
        max_new_tokens=30,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 4) 결과 파싱
    if "가장 적절한 질병은:" in result:
        answer = result.split("가장 적절한 질병은:")[-1].strip()
    else:
        answer = result.strip()

    return {
        "symptom": req.prompt,
        "candidate_diseases": candidate_diseases,
        "recommended_disease": answer,
    }