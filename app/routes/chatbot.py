from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# 사용자의 질문을 받을 모델 정의
class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat(request: ChatRequest):
    user_message = request.message
    bot_response = f"당신이 말한 '{user_message}'에 대해 더 알아볼게요!"
    
    return {"user_message": user_message, "bot_response": bot_response}