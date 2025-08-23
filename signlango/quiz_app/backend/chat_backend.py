from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

# LangChain and Google Gemini imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Sign Language Tutorial API with AI Chat", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in environment variables")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    max_output_tokens=2048,
) if GOOGLE_API_KEY else None

# BSL-specific system prompt
BSL_SYSTEM_PROMPT = """You are a British Sign Language (BSL) teaching assistant and information provider. Your role is to:

1. **Teach BSL**: Provide clear, accurate information about British Sign Language signs, grammar, and usage
2. **Educational Context**: Focus on BSL specifically (not ASL or other sign languages unless asked to compare)
3. **Conversation Memory**: Remember the user's learning progress and previous questions
4. **Practical Guidance**: Offer practical tips for learning and using BSL
5. **Cultural Sensitivity**: Be aware of Deaf culture and community perspectives
6. **Encouraging**: Support the user's learning journey with positive reinforcement

Key BSL Information:
- BSL is the primary sign language used in the UK
- BSL has its own grammar structure different from English
- BSL uses facial expressions, body language, and hand movements
- BSL has regional variations across the UK
- BSL was officially recognized as a language in the UK in 2003

Always provide helpful, accurate, and culturally appropriate responses about BSL. If you're unsure about something, acknowledge the limitation and suggest reliable resources."""

# Conversation memory storage (in production, use a database)
conversation_memories = {}

# Chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", BSL_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Create the conversation chain
conversation_chain = LLMChain(
    llm=llm,
    prompt=chat_prompt,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    verbose=False
) if llm else None

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
    return {"message": "Sign Language Tutorial API with AI Chat is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "sign-language-api", "ai_enabled": llm is not None}

@app.get("/api/signs")
async def get_signs():
    return {"signs": SAMPLE_SIGNS, "count": len(SAMPLE_SIGNS)}

@app.post("/api/detect")
async def detect_sign(data: Dict[str, Any]):
    import random
    detected_sign = random.choice(SAMPLE_SIGNS)
    return {
        "detected_sign": detected_sign["name"],
        "confidence": detected_sign["confidence"],
        "message": f"Detected: {detected_sign['name']}",
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/api/quiz")
async def get_quiz():
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
        {
            "id": 1,
            "title": "Complete Language System",
            "content": "Sign language is a complete, natural language with its own grammar and syntax, not just a visual representation of spoken language.",
            "image": "https://images.unsplash.com/photo-1581833971358-2c8b550f87b3?w=400&h=300&fit=crop",
            "icon": "🗣️",
            "category": "Language"
        },
        {
            "id": 2,
            "title": "Global Diversity",
            "content": "There are over 300 different sign languages used around the world, each with its own unique grammar and vocabulary.",
            "image": "https://images.unsplash.com/photo-1522202176988-66273c2fd55f?w=400&h=300&fit=crop",
            "icon": "🌍",
            "category": "Global"
        },
        {
            "id": 3,
            "title": "BSL in the UK",
            "content": "British Sign Language (BSL) is the primary language of many Deaf people in the UK and was officially recognized in 2003.",
            "image": "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=400&h=300&fit=crop",
            "icon": "🇬🇧",
            "category": "BSL"
        },
        {
            "id": 4,
            "title": "Facial Expressions",
            "content": "Sign language uses facial expressions and body language to convey meaning, making it a rich and expressive form of communication.",
            "image": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=300&fit=crop",
            "icon": "😊",
            "category": "Communication"
        },
        {
            "id": 5,
            "title": "Cognitive Benefits",
            "content": "Learning sign language can improve communication skills, cognitive abilities, and spatial awareness.",
            "image": "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=400&h=300&fit=crop",
            "icon": "🧠",
            "category": "Education"
        },
        {
            "id": 6,
            "title": "Deaf Culture",
            "content": "Deaf culture is rich and diverse, with its own traditions, art, literature, and community values.",
            "image": "https://images.unsplash.com/photo-1522202176988-66273c2fd55f?w=400&h=300&fit=crop",
            "icon": "🎭",
            "category": "Culture"
        }
    ]
    return {"facts": facts, "count": len(facts)}

# Frontend-compatible endpoints
@app.get("/facts")
async def get_facts_frontend():
    return await get_facts()

@app.get("/quiz")
async def get_quiz_frontend(category: str = "all", difficulty: str = "beginner"):
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
    
    if category != "all":
        quiz_questions = [q for q in quiz_questions if q["category"] == category]
    if difficulty != "all":
        quiz_questions = [q for q in quiz_questions if q["difficulty"] == difficulty]
    
    return {"questions": quiz_questions, "total": len(quiz_questions)}

# Chat-related endpoints
@app.get("/chat-suggestions")
async def get_chat_suggestions():
    suggestions = [
        "How do I sign 'hello' in BSL?",
        "What are the basic BSL signs I should learn first?",
        "Can you explain the difference between BSL and ASL?",
        "How do I practice BSL effectively?",
        "What are some common mistakes beginners make in BSL?",
        "How do I sign numbers in BSL?",
        "What's the BSL sign for 'thank you'?",
        "How do I introduce myself in BSL?",
        "What is BSL grammar like?",
        "How do I sign family members in BSL?"
    ]
    return {"suggestions": suggestions, "count": len(suggestions)}

@app.get("/common-questions")
async def get_common_questions():
    questions = [
        {
            "question": "What is the difference between BSL and ASL?",
            "answer": "BSL (British Sign Language) and ASL (American Sign Language) are different sign languages with their own grammar, vocabulary, and cultural context. They are not mutually intelligible. BSL uses a two-handed alphabet while ASL uses one-handed signs."
        },
        {
            "question": "How long does it take to learn BSL?",
            "answer": "Learning BSL varies by individual, but basic conversational skills can be achieved in 6-12 months with regular practice. Full fluency typically takes 2-5 years. Consistent practice and interaction with the Deaf community are key."
        },
        {
            "question": "Is BSL universal?",
            "answer": "No, BSL is specific to the UK. There are over 300 different sign languages worldwide, each with its own unique grammar and vocabulary. BSL is different from ASL, Auslan (Australian), and other sign languages."
        },
        {
            "question": "Can hearing people learn BSL?",
            "answer": "Absolutely! Many hearing people learn BSL to communicate with Deaf family members, friends, or for professional reasons. Learning BSL shows respect for Deaf culture and improves accessibility."
        },
        {
            "question": "What are the benefits of learning BSL?",
            "answer": "Learning BSL improves communication skills, cognitive abilities, cultural awareness, and can open up new career opportunities in education, healthcare, and interpretation. It also helps break down communication barriers."
        }
    ]
    return {"questions": questions, "count": len(questions)}

@app.post("/chat")
async def chat_with_ai(data: Dict[str, Any]):
    """Enhanced chat endpoint with conversation history and BSL context"""
    try:
        user_message = data.get("message", "")
        session_id = data.get("session_id", "default")
        
        if not user_message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Check if AI is available
        if not llm or not conversation_chain:
            return {
                "response": "I understand you're asking about BSL: '" + user_message + "'. The AI chat feature is currently unavailable. Please check your Google Gemini API key configuration.",
                "timestamp": datetime.now().isoformat(),
                "status": "ai_unavailable",
                "session_id": session_id
            }
        
        # Get or create conversation memory for this session
        if session_id not in conversation_memories:
            conversation_memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True
            )
        
        # Create a new chain with the session-specific memory
        session_chain = LLMChain(
            llm=llm,
            prompt=chat_prompt,
            memory=conversation_memories[session_id],
            verbose=False
        )
        
        # Generate response
        response = session_chain.run(input=user_message)
        
        return {
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "session_id": session_id,
            "ai_enabled": True
        }
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return {
            "response": f"I'm sorry, I encountered an error while processing your BSL question. Please try again or rephrase your question.",
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e),
            "session_id": session_id
        }

@app.post("/chat/reset")
async def reset_chat_session(data: Dict[str, Any]):
    """Reset conversation history for a session"""
    try:
        session_id = data.get("session_id", "default")
        
        if session_id in conversation_memories:
            del conversation_memories[session_id]
        
        return {
            "message": "Conversation history reset successfully",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting chat: {str(e)}")

@app.get("/chat/sessions")
async def get_active_sessions():
    """Get list of active chat sessions"""
    return {
        "active_sessions": list(conversation_memories.keys()),
        "count": len(conversation_memories),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 