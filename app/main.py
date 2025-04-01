from fastapi import FastAPI
from app.routes import chatbot
from app.routes import disease

app = FastAPI()

app.include_router(chatbot.router)
app.include_router(disease.router)