from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
from typing import List, Dict, Any

app = FastAPI(title="Sign Language Tutorial API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample data for testing
SAMPLE_SIGNS = [
    {"id": 1, "name": "hello", "description": "Greeting sign", "confidence": 0.95},
    {"id": 2, "name": "thank you", "description": "Gratitude sign", "confidence": 0.88},
    {"id": 3, "name": "goodbye", "description": "Farewell sign", "confidence": 0.92},
    {"id": 4, "name": "yes", "description": "Affirmative sign", "confidence": 0.87},
    {"id": 5, "name": "no", "description": "Negative sign", "confidence": 0.89},
]


@app.get("/")
async def root():
    return {"message": "Sign Language Tutorial API is running!", "status": "healthy"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "sign-language-api"}


@app.get("/api/signs")
async def get_signs():
    return {"signs": SAMPLE_SIGNS, "count": len(SAMPLE_SIGNS)}


@app.post("/api/detect")
async def detect_sign(data: Dict[str, Any]):
    # Simulate sign detection
    import random

    detected_sign = random.choice(SAMPLE_SIGNS)
    return {
        "detected_sign": detected_sign["name"],
        "confidence": detected_sign["confidence"],
        "message": f"Detected: {detected_sign['name']}",
        "timestamp": "2025-08-17T21:00:00Z",
    }


@app.get("/api/quiz")
async def get_quiz():
    # Sample quiz data
    quiz_questions = [
        {
            "id": 1,
            "question": "What does this sign mean?",
            "options": ["Hello", "Goodbye", "Thank you", "Yes"],
            "correct_answer": 0,
            "sign_name": "hello",
        },
        {
            "id": 2,
            "question": "Which sign shows gratitude?",
            "options": ["Hello", "Goodbye", "Thank you", "No"],
            "correct_answer": 2,
            "sign_name": "thank you",
        },
    ]
    return {"questions": quiz_questions, "total": len(quiz_questions)}


@app.get("/api/facts")
async def get_facts():
    facts = [
        "Sign language is a complete, natural language with its own grammar and syntax.",
        "There are over 300 different sign languages used around the world.",
        "American Sign Language (ASL) is the primary language of many North Americans who are deaf.",
        "Sign language uses facial expressions and body language to convey meaning.",
        "Learning sign language can improve communication skills and cognitive abilities.",
    ]
    return {"facts": facts, "count": len(facts)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
