from fastapi import FastAPI
from app.routes import chatbot_fine_tuning
from app.routes import disease
from app.routes import chatbot_rag

app = FastAPI()

app.include_router(chatbot_fine_tuning.router)
app.include_router(disease.router)
app.include_router(chatbot_rag.router)