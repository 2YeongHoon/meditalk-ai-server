환자의 증상 입력으로 병명 / 분과 추천 챗봇

## 고도화 순서
```
STEP1. 테이블 기반 단순 매핑
post /mapped_chat

STEP2. RAG 기반
post /predict/disease

STEP2. 파인튜닝 (로컬 자원의 한계로 작업 진행중)
post /add-data - 학습 데이터 추가
post /train - 학습
post /fine-tuning-chat 
```

## 구동방법
```
가상환경 활성화
source venv/bin/activate

서버 실행
uvicorn app.main:app --reload 
```

## Swagger
```
http://localhost:8000/docs
```
