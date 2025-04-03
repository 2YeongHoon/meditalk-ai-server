from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# FastAPI 앱 생성
router = APIRouter()

# 모델 및 토크나이저 로드
MODEL_NAME = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to("cpu")

# 요청 모델 정의
class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 개선된 프롬프트: 오직 실제 답변만 나오도록 요청
        input_text = (
            f"Patient: {request.message}\n"
            "Based on the above symptom, determine the appropriate medical department and the most likely diagnosis.\n"
            "Answer in EXACTLY two lines with no extra text, instructions, or examples, following this format:\n"
            "Department: [Your answer]\n"
            "Diagnosis: [Your answer]\n"
            "Please replace any placeholder with the actual answer and do not include any angle brackets."
        )

        # 입력 토큰화
        inputs = tokenizer(input_text, return_tensors="pt").to("cpu")

        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=50,
                temperature=0.3,
                top_k=20,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id  
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True, errors="replace")
        # 프롬프트 부분 제거
        response = response.replace(input_text, "").strip()
        
        # 후처리: "Department:"와 "Diagnosis:"가 포함된 줄만 추출
        lines = response.splitlines()
        refined_lines = []
        for line in lines:
            line_strip = line.strip()
            if line_strip.startswith("Department:") or line_strip.startswith("Diagnosis:"):
                # placeholder 문구가 포함되어 있으면 제거
                line_strip = line_strip.replace("<department>", "").replace("<diagnosis>", "").strip()
                refined_lines.append(line_strip)
            if len(refined_lines) == 2:
                break
        refined_response = "\n".join(refined_lines)
        
        return {"response": refined_response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))