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


# Frontend-compatible endpoints (without /api prefix)
@app.get("/facts")
async def get_facts_frontend():
    """Frontend-compatible facts endpoint"""
    return await get_facts()


@app.get("/quiz")
async def get_quiz_frontend(category: str = "all", difficulty: str = "beginner"):
    """Frontend-compatible quiz endpoint with query parameters"""
    # Sample quiz data based on category and difficulty
    quiz_questions = [
        {
            "id": 1,
            "question": "What does this sign mean?",
            "options": ["Hello", "Goodbye", "Thank you", "Yes"],
            "correct_answer": 0,
            "sign_name": "hello",
            "category": "greetings",
            "difficulty": "beginner"
        },
        {
            "id": 2,
            "question": "Which sign shows gratitude?",
            "options": ["Hello", "Goodbye", "Thank you", "No"],
            "correct_answer": 2,
            "sign_name": "thank you",
            "category": "greetings",
            "difficulty": "beginner"
        },
        {
            "id": 3,
            "question": "What is the sign for 'goodbye'?",
            "options": ["Hello", "Goodbye", "Thank you", "Yes"],
            "correct_answer": 1,
            "sign_name": "goodbye",
            "category": "greetings",
            "difficulty": "beginner"
        },
        {
            "id": 4,
            "question": "Which sign means 'yes'?",
            "options": ["Hello", "Goodbye", "Yes", "No"],
            "correct_answer": 2,
            "sign_name": "yes",
            "category": "responses",
            "difficulty": "beginner"
        },
        {
            "id": 5,
            "question": "What is the sign for 'no'?",
            "options": ["Hello", "Goodbye", "Yes", "No"],
            "correct_answer": 3,
            "sign_name": "no",
            "category": "responses",
            "difficulty": "beginner"
        }
    ]
    
    # Filter by category and difficulty if specified
    if category != "all":
        quiz_questions = [q for q in quiz_questions if q["category"] == category]
    if difficulty != "all":
        quiz_questions = [q for q in quiz_questions if q["difficulty"] == difficulty]
    
    return {"questions": quiz_questions, "total": len(quiz_questions)}


# Chat-related endpoints
@app.get("/chat-suggestions")
async def get_chat_suggestions():
    """Get chat suggestions for users"""
    suggestions = [
        "How do I sign 'hello'?",
        "What are the basic signs I should learn first?",
        "Can you explain the difference between ASL and BSL?",
        "How do I practice sign language effectively?",
        "What are some common mistakes beginners make?",
        "How do I sign numbers in BSL?",
        "What's the sign for 'thank you'?",
        "How do I introduce myself in sign language?"
    ]
    return {"suggestions": suggestions, "count": len(suggestions)}


@app.get("/common-questions")
async def get_common_questions():
    """Get frequently asked questions about sign language"""
    questions = [
        {
            "question": "What is the difference between ASL and BSL?",
            "answer": "ASL (American Sign Language) and BSL (British Sign Language) are different sign languages with their own grammar, vocabulary, and cultural context. They are not mutually intelligible."
        },
        {
            "question": "How long does it take to learn sign language?",
            "answer": "Learning sign language varies by individual, but basic conversational skills can be achieved in 6-12 months with regular practice. Full fluency typically takes 2-5 years."
        },
        {
            "question": "Is sign language universal?",
            "answer": "No, sign language is not universal. There are over 300 different sign languages worldwide, each with its own unique grammar and vocabulary."
        },
        {
            "question": "Can hearing people learn sign language?",
            "answer": "Absolutely! Many hearing people learn sign language to communicate with deaf family members, friends, or for professional reasons."
        },
        {
            "question": "What are the benefits of learning sign language?",
            "answer": "Learning sign language improves communication skills, cognitive abilities, cultural awareness, and can open up new career opportunities in education, healthcare, and interpretation."
        }
    ]
    return {"questions": questions, "count": len(questions)}


@app.post("/chat")
async def chat_with_ai(data: Dict[str, Any]):
    """Chat endpoint for AI-powered sign language assistance"""
    try:
        user_message = data.get("message", "")
        
        # For now, return a simple response
        # TODO: Integrate with Google Gemini API
        response = f"I understand you're asking about sign language: '{user_message}'. This is a placeholder response. Please provide your Google Gemini API key to enable AI-powered responses."
        
        return {
            "response": response,
            "timestamp": "2025-08-17T21:00:00Z",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
