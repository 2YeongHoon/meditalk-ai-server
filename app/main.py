from fastapi import FastAPI
from app.routes import chatbot

app = FastAPI()

# 라우트 등록
app.include_router(chatbot.router)

@app.get("/")
def read_root():
    return {"message": "Hello, Meditalk AI!"}