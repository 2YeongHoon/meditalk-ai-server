from fastapi import APIRouter
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

router = APIRouter()

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")  # "cuda"로 GPU 사용 가능

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat(request: ChatRequest):
    user_message = request.message
    bot_response = f"당신이 말한 '{user_message}'에 대해 더 알아볼게요!"
    return {"user_message": user_message, "bot_response": bot_response}

@router.post("/chat-ai")
async def predict(request: ChatRequest):
    # 모델 입력 준비
    inputs = tokenizer(request.message, return_tensors="pt").to("cpu")
    
    # 모델 예측 (생성)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=100)

    # 모델 응답 디코딩
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return {"response": response}