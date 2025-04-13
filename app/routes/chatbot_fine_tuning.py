from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

router = APIRouter()

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")

training_data = []

class ChatRequest(BaseModel):
    message: str

class TrainData(BaseModel):
    input: str
    label: str

@router.post("/add-data")
async def add_data(data: TrainData):
    training_data.append({"input": data.input, "label": data.label})
    return {"message": "데이터 추가 완료", "total": len(training_data)}

@router.post("/train")
async def train():
    if not training_data:
        raise HTTPException(status_code=400, detail="학습할 데이터가 없습니다.")

    dataset = Dataset.from_dict({
        "input": [data["input"] for data in training_data],
        "label": [tokenizer(data["label"], truncation=True, padding="max_length", max_length=512)["input_ids"]
                  for data in training_data]
    })

    if len(training_data) > 1:
        dataset = dataset.train_test_split(test_size=0.2)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    def tokenize_function(examples):
        inputs = tokenizer(
            examples["input"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

        labels = examples["label"]

        labels = [
            [(token if token != tokenizer.pad_token_id else -100) for token in label]
            for label in labels
        ]

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }

    tokenized_datasets = train_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=eval_dataset if eval_dataset else None,
        tokenizer=tokenizer
    )

    trainer.train()

    model.save_pretrained("trained_model")
    tokenizer.save_pretrained("trained_model")

    return {"message": "학습 완료"}

@router.post("/fine-tuning-chat")
async def predict(request: ChatRequest):
    inputs = tokenizer(request.message, return_tensors="pt").to("cpu")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": response}