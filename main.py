from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import gdown
import os

app = FastAPI()

# File IDs from Google Drive
MODEL_FILE_ID = "1EdBjbZEiLwCw-RD5dX-wtKCd5Hmh9LnJ"
MODEL_FILENAME = "model.safetensors"
MODEL_DIR = "kelantan-model"

class TranslationInput(BaseModel):
    input: str

@app.on_event("startup")
async def load_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)

    if not os.path.exists(model_path):
        print("⬇️ Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        gdown.download(url, model_path, quiet=False)

    print("✅ Loading model...")
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

@app.post("/translate")
async def translate(input_data: TranslationInput):
    input_ids = tokenizer(input_data.input, return_tensors="pt", padding=True).input_ids
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=100)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return { "output": decoded }
