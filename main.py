import os
import re
import string
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, TFBertForSequenceClassification

app = FastAPI()

# Text cleaning function (from your model code)
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Text encoding function (from your model code)
def encode_texts(tokenizer, texts, max_length=256):
    encoding = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='tf'
    )
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'token_type_ids': encoding.get('token_type_ids', tf.zeros_like(encoding['input_ids']))
    }

# Load model and tokenizer with detailed debugging
def load_model_and_tokenizer():
    model_path = "models/final_bertCopy_model"
    tokenizer_path = "models/final_bertCopy_tokenizer"
    abs_model_path = os.path.abspath(model_path)
    abs_tokenizer_path = os.path.abspath(tokenizer_path)
    print(f"Checking model path: {abs_model_path}")
    print(f"Checking tokenizer path: {abs_tokenizer_path}")
    if not os.path.exists(abs_model_path):
        raise FileNotFoundError(f"Model directory not found at {abs_model_path}")
    if not os.path.exists(abs_tokenizer_path):
        raise FileNotFoundError(f"Tokenizer directory not found at {abs_tokenizer_path}")
    # List directory contents for debugging
    print(f"Model directory contents: {os.listdir(abs_model_path)}")
    print(f"Tokenizer directory contents: {os.listdir(abs_tokenizer_path)}")
    try:
        model = TFBertForSequenceClassification.from_pretrained(abs_model_path)
        tokenizer = BertTokenizer.from_pretrained(abs_tokenizer_path)
        print("Model and tokenizer loaded successfully")
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model/tokenizer: {str(e)}")

# Initialize model and tokenizer
try:
    model, tokenizer = load_model_and_tokenizer()
except Exception as e:
    print(f"Error loading model/tokenizer: {e}")
    model, tokenizer = None, None

# Input data model
class TextInput(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "BERT Text Classification API is running"}

@app.post("/predict")
async def predict(input_data: TextInput):
    if model is None or tokenizer is None:
        return {"error": "Model or tokenizer not loaded"}
    
    try:
        # Clean and encode input text
        cleaned_text = clean_text(input_data.text)
        inputs = encode_texts(tokenizer, [cleaned_text])
        
        # Perform inference
        logits = model.predict(inputs).logits
        predicted_class = np.argmax(logits, axis=1)[0]
        prediction = "AI-generated" if predicted_class == 1 else "Human-written"
        
        # Include confidence scores
        probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
        confidence = float(probabilities[predicted_class])
        
        return {
            "text": input_data.text,
            "prediction": prediction,
            "confidence": confidence,
            "logits": logits.tolist()[0]
        }
    except Exception as e:
        return {"error": str(e)}