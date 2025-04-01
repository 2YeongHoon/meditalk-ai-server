from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# MySQL 연결 정보 (환경변수에서 가져오기)
DATABASE_URL = os.getenv("DATABASE_URL")

# 데이터베이스 엔진 생성
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# DB 세션 의존성 주입 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()