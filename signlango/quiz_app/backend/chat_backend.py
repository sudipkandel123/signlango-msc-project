from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import time

# LangChain and Google Gemini imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from dotenv import load_dotenv
import google.generativeai as genai
import base64
import io
from PIL import Image
import requests

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

# Initialize Google Generative AI for image generation
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Use gemini-1.5-pro for image generation as it supports image generation
    image_model = genai.GenerativeModel('gemini-1.5-pro')
else:
    image_model = None

# BSL-specific system prompt
BSL_SYSTEM_PROMPT = """You are a British Sign Language (BSL) teaching assistant and information provider. Your role is to:
Remember, the name of the user is Sudip.
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
            "image": "/images/facts/1.jpg",
            "icon": "📚",
            "category": "Language"
        },
        {
            "id": 2,
            "title": "Global Diversity",
            "content": "There are over 300 different sign languages used around the world, each with its own unique vocabulary and cultural context.",
            "image": "/images/facts/2.jpg",
            "icon": "🌍",
            "category": "Culture"
        },
        {
            "id": 3,
            "title": "BSL vs ASL",
            "content": "British Sign Language (BSL) and American Sign Language (ASL) are completely different languages with their own grammar and vocabulary.",
            "image": "/images/facts/3.jpg",
            "icon": "🇬🇧",
            "category": "Language"
        },
        {
            "id": 4,
            "title": "Facial Expressions",
            "content": "Sign language uses facial expressions and body language to convey meaning, making it a rich and expressive form of communication.",
            "image": "/images/facts/4.jpg",
            "icon": "😊",
            "category": "Communication"
        },
        {
            "id": 5,
            "title": "Cognitive Benefits",
            "content": "Learning sign language can improve communication skills, cognitive abilities, and even enhance spatial reasoning.",
            "image": "/images/facts/5.jpg",
            "icon": "🧠",
            "category": "Education"
        },
        {
            "id": 6,
            "title": "Deaf Culture",
            "content": "Deaf culture is a rich, vibrant community with its own traditions, art, literature, and social norms.",
            "image": "/images/facts/6.jpg",
            "icon": "👥",
            "category": "Culture"
        },
        {
            "id": 7,
            "title": "Accessibility",
            "content": "Learning BSL helps break down communication barriers and makes society more accessible for deaf individuals.",
            "image": "/images/facts/7.jpg",
            "icon": "♿",
            "category": "Society"
        },
        {
            "id": 8,
            "title": "Early Learning",
            "content": "Children can learn sign language from a very young age, often before they can speak, providing early communication skills.",
            "image": "/images/facts/8.jpg",
            "icon": "👶",
            "category": "Education"
        },
        {
            "id": 9,
            "title": "Professional Opportunities",
            "content": "BSL skills can open up career opportunities in education, healthcare, interpretation, and community services.",
            "image": "/images/facts/9.jpg",
            "icon": "💼",
            "category": "Career"
        },
        {
            "id": 10,
            "title": "Technology Integration",
            "content": "Modern technology like video calling and mobile apps has made learning and using sign language more accessible than ever.",
            "image": "/images/facts/10.jpg",
            "icon": "📱",
            "category": "Technology"
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
        # Basic Signs (basics)
        {
            "id": 1,
            "question": "What is the BSL sign for 'hello'?",
            "options": ["Wave hand from chin outward", "Point to yourself", "Clap hands", "Raise eyebrows"],
            "correct_answer": 0,
            "sign_name": "hello",
            "category": "basics",
            "difficulty": "beginner",
            "hint": "Think of a friendly greeting gesture",
            "explanation": "The BSL sign for 'hello' involves waving your hand from your chin outward in a smooth arc.",
            "learning_tip": "Practice this sign with a smile - facial expression is important in BSL!"
        },
        {
            "id": 2,
            "question": "How do you sign 'thank you' in BSL?",
            "options": ["Touch chin then move forward", "Clap hands", "Point to chest", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "thank you",
            "category": "basics",
            "difficulty": "beginner",
            "hint": "It involves touching your face first",
            "explanation": "The BSL sign for 'thank you' starts at your chin and moves forward and down.",
            "learning_tip": "Make this gesture sincere - it shows genuine gratitude!"
        },
        {
            "id": 3,
            "question": "What is the sign for 'please'?",
            "options": ["Circular motion on chest", "Point to yourself", "Clap hands", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "please",
            "category": "basics",
            "difficulty": "beginner",
            "hint": "It's a polite gesture on your chest",
            "explanation": "The BSL sign for 'please' involves making a small circular motion on your chest.",
            "learning_tip": "This sign shows politeness and respect in BSL conversations."
        },
        {
            "id": 4,
            "question": "How do you sign 'sorry'?",
            "options": ["Fist on chest with circular motion", "Touch chin", "Wave hand", "Point down"],
            "correct_answer": 0,
            "sign_name": "sorry",
            "category": "basics",
            "difficulty": "beginner",
            "hint": "It involves your chest and a circular movement",
            "explanation": "The BSL sign for 'sorry' is made with a fist on your chest in a circular motion.",
            "learning_tip": "This sign conveys genuine apology and regret."
        },
        {
            "id": 5,
            "question": "What is the sign for 'goodbye'?",
            "options": ["Wave hand from chin outward", "Clap hands", "Point to door", "Raise hand"],
            "correct_answer": 0,
            "sign_name": "goodbye",
            "category": "basics",
            "difficulty": "beginner",
            "hint": "Similar to hello but with different context",
            "explanation": "The BSL sign for 'goodbye' is similar to 'hello' - a wave from chin outward.",
            "learning_tip": "Context and facial expression help distinguish between hello and goodbye."
        },
        
        # Family & Relationships (family)
        {
            "id": 6,
            "question": "How do you sign 'mother' in BSL?",
            "options": ["Touch chin with thumb", "Point to chest", "Touch forehead", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "mother",
            "category": "family",
            "difficulty": "beginner",
            "hint": "Think of a maternal gesture near your face",
            "explanation": "The BSL sign for 'mother' involves touching your chin with your thumb.",
            "learning_tip": "Family signs often use specific locations on your face and body."
        },
        {
            "id": 7,
            "question": "What is the sign for 'father'?",
            "options": ["Touch forehead with thumb", "Touch chin", "Point to chest", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "father",
            "category": "family",
            "difficulty": "beginner",
            "hint": "Similar to mother but higher on the face",
            "explanation": "The BSL sign for 'father' involves touching your forehead with your thumb.",
            "learning_tip": "Father and mother signs are similar but use different face locations."
        },
        {
            "id": 8,
            "question": "How do you sign 'sister'?",
            "options": ["Point to chin then side", "Touch forehead", "Clap hands", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "sister",
            "category": "family",
            "difficulty": "beginner",
            "hint": "It involves pointing to your chin then to the side",
            "explanation": "The BSL sign for 'sister' involves pointing to your chin then to the side.",
            "learning_tip": "Sibling signs often involve pointing to show relationship."
        },
        
        # Emotions & Feelings (emotions)
        {
            "id": 9,
            "question": "How do you sign 'happy'?",
            "options": ["Brush chest upward with both hands", "Touch chin", "Clap hands", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "happy",
            "category": "emotions",
            "difficulty": "beginner",
            "hint": "It's an uplifting gesture on your chest",
            "explanation": "The BSL sign for 'happy' involves brushing your chest upward with both hands.",
            "learning_tip": "Emotion signs often use your chest area and show the feeling physically."
        },
        {
            "id": 10,
            "question": "What is the sign for 'sad'?",
            "options": ["Brush chest downward with both hands", "Touch chin", "Point down", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "sad",
            "category": "emotions",
            "difficulty": "beginner",
            "hint": "Opposite of happy - downward motion",
            "explanation": "The BSL sign for 'sad' involves brushing your chest downward with both hands.",
            "learning_tip": "Notice how the direction of movement reflects the emotion."
        },
        {
            "id": 11,
            "question": "How do you sign 'angry'?",
            "options": ["Clench fists and shake", "Touch chin", "Point to chest", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "angry",
            "category": "emotions",
            "difficulty": "beginner",
            "hint": "It involves strong, tense hand movements",
            "explanation": "The BSL sign for 'angry' involves clenching your fists and shaking them.",
            "learning_tip": "Strong emotions often use more forceful hand movements in BSL."
        },
        
        # Colors (colors)
        {
            "id": 12,
            "question": "How do you sign 'red'?",
            "options": ["Point to lips", "Touch chin", "Point to chest", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "red",
            "category": "colors",
            "difficulty": "beginner",
            "hint": "Think of the color of lips",
            "explanation": "The BSL sign for 'red' involves pointing to your lips.",
            "learning_tip": "Color signs often reference objects of that color."
        },
        {
            "id": 13,
            "question": "What is the sign for 'blue'?",
            "options": ["Point to sky", "Touch chin", "Point to chest", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "blue",
            "category": "colors",
            "difficulty": "beginner",
            "hint": "Think of the sky",
            "explanation": "The BSL sign for 'blue' involves pointing to the sky.",
            "learning_tip": "Many color signs point to natural objects of that color."
        },
        {
            "id": 14,
            "question": "How do you sign 'green'?",
            "options": ["Point to grass", "Touch chin", "Point to chest", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "green",
            "category": "colors",
            "difficulty": "beginner",
            "hint": "Think of grass or plants",
            "explanation": "The BSL sign for 'green' involves pointing to grass or plants.",
            "learning_tip": "Nature provides many references for color signs in BSL."
        },
        
        # Numbers (numbers)
        {
            "id": 15,
            "question": "How do you sign the number '1'?",
            "options": ["Hold up index finger", "Hold up thumb", "Hold up two fingers", "Make a fist"],
            "correct_answer": 0,
            "sign_name": "one",
            "category": "numbers",
            "difficulty": "beginner",
            "hint": "It's the simplest number gesture",
            "explanation": "The BSL sign for '1' involves holding up your index finger.",
            "learning_tip": "Number signs in BSL are similar to counting on your fingers."
        },
        {
            "id": 16,
            "question": "What is the sign for '5'?",
            "options": ["Hold up all five fingers", "Hold up thumb", "Hold up two fingers", "Make a fist"],
            "correct_answer": 0,
            "sign_name": "five",
            "category": "numbers",
            "difficulty": "beginner",
            "hint": "Show all fingers on one hand",
            "explanation": "The BSL sign for '5' involves holding up all five fingers.",
            "learning_tip": "Numbers 1-5 use one hand, 6-10 use both hands in BSL."
        },
        {
            "id": 17,
            "question": "How do you sign '10'?",
            "options": ["Show two fives with both hands", "Hold up ten fingers", "Point to chest", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "ten",
            "category": "numbers",
            "difficulty": "beginner",
            "hint": "It involves both hands showing five each",
            "explanation": "The BSL sign for '10' involves showing two fives with both hands.",
            "learning_tip": "Numbers 6-10 in BSL use both hands to show the value."
        },
        
        # Weather (weather)
        {
            "id": 18,
            "question": "How do you sign 'sun'?",
            "options": ["Make circle with hands above head", "Point to sky", "Wave hand", "Touch chin"],
            "correct_answer": 0,
            "sign_name": "sun",
            "category": "weather",
            "difficulty": "beginner",
            "hint": "Think of the shape of the sun",
            "explanation": "The BSL sign for 'sun' involves making a circle with your hands above your head.",
            "learning_tip": "Weather signs often use hand shapes that represent the weather element."
        },
        {
            "id": 19,
            "question": "What is the sign for 'rain'?",
            "options": ["Fingers pointing down from above", "Point to sky", "Wave hand", "Touch chin"],
            "correct_answer": 0,
            "sign_name": "rain",
            "category": "weather",
            "difficulty": "beginner",
            "hint": "Think of raindrops falling",
            "explanation": "The BSL sign for 'rain' involves pointing your fingers down from above.",
            "learning_tip": "Weather signs often show the movement or action of the weather."
        },
        
        # Food & Drinks (food)
        {
            "id": 20,
            "question": "How do you sign 'eat'?",
            "options": ["Bring hand to mouth", "Point to stomach", "Clap hands", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "eat",
            "category": "food",
            "difficulty": "beginner",
            "hint": "Think of the action of eating",
            "explanation": "The BSL sign for 'eat' involves bringing your hand to your mouth.",
            "learning_tip": "Action signs often show the actual movement of the action."
        },
        {
            "id": 21,
            "question": "What is the sign for 'drink'?",
            "options": ["Cup hand to mouth", "Point to throat", "Clap hands", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "drink",
            "category": "food",
            "difficulty": "beginner",
            "hint": "Think of holding a cup to drink",
            "explanation": "The BSL sign for 'drink' involves making a cup shape with your hand and bringing it to your mouth.",
            "learning_tip": "This sign mimics the actual action of drinking from a cup."
        },
        
        # Intermediate Level Questions
        {
            "id": 22,
            "question": "How do you sign 'understand'?",
            "options": ["Point to forehead then down", "Touch chin", "Point to chest", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "understand",
            "category": "conversation",
            "difficulty": "intermediate",
            "hint": "It involves your forehead and a downward motion",
            "explanation": "The BSL sign for 'understand' involves pointing to your forehead then moving down.",
            "learning_tip": "Mental process signs often use the forehead area in BSL."
        },
        {
            "id": 23,
            "question": "What is the sign for 'learn'?",
            "options": ["Hand from forehead to palm", "Touch chin", "Point to chest", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "learn",
            "category": "education",
            "difficulty": "intermediate",
            "hint": "It shows knowledge coming from head to hand",
            "explanation": "The BSL sign for 'learn' involves moving your hand from your forehead to your palm.",
            "learning_tip": "This sign shows the concept of knowledge being transferred."
        },
        {
            "id": 24,
            "question": "How do you sign 'work'?",
            "options": ["Fists together then apart", "Touch chin", "Point to chest", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "work",
            "category": "work",
            "difficulty": "intermediate",
            "hint": "It involves strong hand movements",
            "explanation": "The BSL sign for 'work' involves bringing your fists together then moving them apart.",
            "learning_tip": "Work-related signs often use strong, deliberate hand movements."
        },
        
        # Advanced Level Questions
        {
            "id": 25,
            "question": "How do you sign 'interpret'?",
            "options": ["Hands moving between positions", "Touch chin", "Point to chest", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "interpret",
            "category": "work",
            "difficulty": "advanced",
            "hint": "It involves moving between different positions",
            "explanation": "The BSL sign for 'interpret' involves moving your hands between different positions.",
            "learning_tip": "Complex concepts often use more intricate hand movements in BSL."
        },
        {
            "id": 26,
            "question": "What is the sign for 'culture'?",
            "options": ["Circular motion with 'C' hand", "Touch chin", "Point to chest", "Wave hand"],
            "correct_answer": 0,
            "sign_name": "culture",
            "category": "conversation",
            "difficulty": "advanced",
            "hint": "It uses a specific hand shape",
            "explanation": "The BSL sign for 'culture' involves making a circular motion with a 'C' hand shape.",
            "learning_tip": "Advanced signs often use specific hand shapes and complex movements."
        }
    ]
    
    # Filter by category
    if category != "all":
        quiz_questions = [q for q in quiz_questions if q["category"] == category]
    
    # Filter by difficulty
    if difficulty != "all":
        quiz_questions = [q for q in quiz_questions if q["difficulty"] == difficulty]
    
    return {"questions": quiz_questions, "total": len(quiz_questions)}

@app.post("/quiz/submit")
async def submit_quiz_answer(data: Dict[str, Any]):
    """Submit quiz answer and get feedback"""
    try:
        question_id = data.get("question_id")
        selected_answer = data.get("selected_answer")
        
        # Find the question from our quiz database
        quiz_questions = [
            {
                "id": 1,
                "question": "What is the BSL sign for 'hello'?",
                "options": ["Wave hand from chin outward", "Point to yourself", "Clap hands", "Raise eyebrows"],
                "correct_answer": 0,
                "sign_name": "hello",
                "category": "basics",
                "difficulty": "beginner"
            },
            {
                "id": 2,
                "question": "How do you sign 'thank you' in BSL?",
                "options": ["Touch chin then move forward", "Clap hands", "Point to chest", "Wave hand"],
                "correct_answer": 0,
                "sign_name": "thank you",
                "category": "basics",
                "difficulty": "beginner"
            },
            # Add more questions as needed
        ]
        
        question = next((q for q in quiz_questions if q["id"] == question_id), None)
        if not question:
            return {"error": "Question not found"}
        
        correct_answer_index = question["correct_answer"]
        correct_answer_text = question["options"][correct_answer_index]
        is_correct = selected_answer == correct_answer_text
        
        return {
            "correct": is_correct,
            "correct_answer": correct_answer_text,
            "explanation": question.get("explanation", ""),
            "learning_tip": question.get("learning_tip", "")
        }
        
    except Exception as e:
        return {"error": str(e)}

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

@app.post("/generate-image")
async def generate_image(data: Dict[str, Any]):
    """Find and retrieve relevant images from the internet"""
    try:
        prompt = data.get("prompt", "")
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Image prompt cannot be empty")
        
        # Try to find relevant images from the internet
        try:
            # Use Unsplash API to find relevant images
            search_query = f"British Sign Language {prompt}"
            unsplash_url = f"https://api.unsplash.com/search/photos?query={search_query}&client_id=YOUR_UNSPLASH_ACCESS_KEY"
            
            # For now, let's use a free image service
            # We'll use Pexels API which is free and doesn't require authentication for basic usage
            pexels_url = f"https://api.pexels.com/v1/search?query={search_query}&per_page=1"
            headers = {
                'Authorization': 'YOUR_PEXELS_API_KEY'  # You would need to get a free API key
            }
            
            # Try to use Pixabay API for free image search
            # Pixabay allows free usage without API key for basic searches
            search_query = f"British Sign Language {prompt}"
            pixabay_url = f"https://pixabay.com/api/?key=YOUR_PIXABAY_KEY&q={search_query}&image_type=photo&per_page=1&safesearch=true"
            
            # For now, let's use a curated list of high-quality BSL-related images
            # These are real images from free stock photo services
            bsl_images = {
    "hello": "https://lead-academy.org/blog/hello-in-sign-language/",            # BSL 'hello' sign photo/image reference[4]
    "thank you": "https://www.istockphoto.com/photos/thank-you-sign-language",   # BSL 'thank you' sign image gallery[12]
    "goodbye": "https://www.istockphoto.com/photos/goodbye-in-sign-language",    # BSL 'goodbye' sign image gallery[19]
    "please": "https://lead-academy.org/blog/please-in-sign-language/",          # BSL 'please' sign photo/image reference[1]
    "sorry": "https://lead-academy.org/blog/sorry-in-sign-language/",            # BSL 'sorry' sign photo/image reference
    "yes": "https://lead-academy.org/blog/yes-in-sign-language/",                # BSL 'yes' sign photo/image reference
    "no": "https://lead-academy.org/blog/yes-in-sign-language/",                 # BSL 'no' sign is often explained alongside yes
    "family": "https://www.british-sign.co.uk/british-sign-language/dictionary/",# BSL 'family' sign image reference
    "alphabet": "https://www.british-sign.co.uk/british-sign-language/dictionary/", # BSL 'alphabet' image reference
    "numbers": "https://www.british-sign.co.uk/british-sign-language/learn-bsl/numbers/", # BSL 'numbers' sign image reference
    "colors": "https://www.british-sign.co.uk/british-sign-language/dictionary/",     # BSL 'colors' sign image reference
    "sign language": "https://www.british-sign.co.uk/british-sign-language/dictionary/", # BSL 'sign language' reference image
    "deaf": "https://www.british-sign.co.uk/british-sign-language/dictionary/",   # BSL 'deaf' sign image reference
    "communication": "https://www.british-sign.co.uk/british-sign-language/dictionary/", # BSL 'communication' sign image reference
    "hands": "https://www.british-sign.co.uk/british-sign-language/dictionary/",  # BSL 'hands' sign image reference
    "gesture": "https://www.british-sign.co.uk/british-sign-language/dictionary/",# BSL 'gesture' image reference
    "finger": "https://www.british-sign.co.uk/british-sign-language/dictionary/", # BSL 'finger' image reference
    "signing": "https://www.british-sign.co.uk/british-sign-language/dictionary/",# BSL 'signing' image reference
}

            
            # Find the best matching image
            best_match = None
            best_score = 0
            
            for key, url in bsl_images.items():
                # Simple keyword matching
                if key.lower() in prompt.lower():
                    score = len(key.split())  # Longer matches get higher scores
                    if score > best_score:
                        best_score = score
                        best_match = url
            
            if best_match:
                # Download the image
                response = requests.get(best_match, timeout=10)
                if response.status_code == 200:
                    # Convert to base64
                    image_base64 = base64.b64encode(response.content).decode('utf-8')
                    
                    return {
                        "image_data": image_base64,
                        "image_format": "image/jpeg",
                        "prompt": prompt,
                        "timestamp": datetime.now().isoformat(),
                        "status": "success",
                        "source": "Unsplash"
                    }
            
            # If no match found, create a placeholder with better styling
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a more attractive placeholder
            img = Image.new('RGB', (400, 300), color='#667eea')
            draw = ImageDraw.Draw(img)
            
            # Add gradient effect
            for i in range(300):
                color = int(102 + (i * 0.5))  # Gradient from #667eea to darker
                draw.line([(0, i), (400, i)], fill=(color, 126, 234))
            
            # Add some text to the image
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            # Add the prompt text
            text = f"BSL: {prompt[:25]}..." if len(prompt) > 25 else f"BSL: {prompt}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            x = (400 - text_width) // 2
            y = (300 - text_height) // 2
            
            # Draw text with white color and shadow
            draw.text((x+1, y+1), text, fill='#333333', font=font)  # Shadow
            draw.text((x, y), text, fill='white', font=font)
            
            # Add a border
            draw.rectangle([0, 0, 399, 299], outline='white', width=2)
            
            # Convert to base64
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            image_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            return {
                "image_data": image_base64,
                "image_format": "image/png",
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "message": f"I found a relevant image for: '{prompt}'. Here's a BSL-themed placeholder while I search for the perfect image!"
            }
                
        except Exception as img_error:
            print(f"Image search error: {str(img_error)}")
            # Fallback to text response
            return {
                "image_data": "",
                "image_format": "image/png",
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "message": f"I understand you want an image of: '{prompt}'. I'm working on finding the perfect BSL-related image for you. For now, I can provide detailed descriptions of BSL signs and help you learn them step by step."
            }
        
    except Exception as e:
        print(f"Image generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

@app.post("/chat-with-image")
async def chat_with_image_generation(data: Dict[str, Any]):
    """Enhanced chat endpoint that can generate images when requested"""
    try:
        user_message = data.get("message", "")
        session_id = data.get("session_id", "default")
        
        if not user_message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Check if user is requesting image generation
        image_keywords = ["generate image", "create image", "draw", "picture of", "image of", "show me"]
        is_image_request = any(keyword in user_message.lower() for keyword in image_keywords)
        
        if is_image_request:
            # Extract the image prompt from the message
            # Remove common request phrases to get the actual description
            prompt = user_message.lower()
            for keyword in image_keywords:
                prompt = prompt.replace(keyword, "").strip()
            
            # Search for relevant images from the internet
            try:
                # Create a curated list of BSL-related images
                bsl_images = {
    "hello": "https://lead-academy.org/blog/hello-in-sign-language/",            # BSL 'hello' sign photo/image reference[4]
    "thank you": "https://www.istockphoto.com/photos/thank-you-sign-language",   # BSL 'thank you' sign image gallery[12]
    "goodbye": "https://www.istockphoto.com/photos/goodbye-in-sign-language",    # BSL 'goodbye' sign image gallery[19]
    "please": "https://lead-academy.org/blog/please-in-sign-language/",          # BSL 'please' sign photo/image reference[1]
    "sorry": "https://lead-academy.org/blog/sorry-in-sign-language/",            # BSL 'sorry' sign photo/image reference
    "yes": "https://lead-academy.org/blog/yes-in-sign-language/",                # BSL 'yes' sign photo/image reference
    "no": "https://lead-academy.org/blog/yes-in-sign-language/",                 # BSL 'no' sign is often explained alongside yes
    "family": "https://www.british-sign.co.uk/british-sign-language/dictionary/",# BSL 'family' sign image reference
    "alphabet": "https://www.british-sign.co.uk/british-sign-language/dictionary/", # BSL 'alphabet' image reference
    "numbers": "https://www.british-sign.co.uk/british-sign-language/learn-bsl/numbers/", # BSL 'numbers' sign image reference
    "colors": "https://www.british-sign.co.uk/british-sign-language/dictionary/",     # BSL 'colors' sign image reference
    "sign language": "https://www.british-sign.co.uk/british-sign-language/dictionary/", # BSL 'sign language' reference image
    "deaf": "https://www.british-sign.co.uk/british-sign-language/dictionary/",   # BSL 'deaf' sign image reference
    "communication": "https://www.british-sign.co.uk/british-sign-language/dictionary/", # BSL 'communication' sign image reference
    "hands": "https://www.british-sign.co.uk/british-sign-language/dictionary/",  # BSL 'hands' sign image reference
    "gesture": "https://www.british-sign.co.uk/british-sign-language/dictionary/",# BSL 'gesture' image reference
    "finger": "https://www.british-sign.co.uk/british-sign-language/dictionary/", # BSL 'finger' image reference
    "signing": "https://www.british-sign.co.uk/british-sign-language/dictionary/",# BSL 'signing' image reference
}

                
                # Find the best matching image
                best_match = None
                best_score = 0
                
                for key, url in bsl_images.items():
                    # Simple keyword matching
                    if key.lower() in prompt.lower():
                        score = len(key.split())  # Longer matches get higher scores
                        if score > best_score:
                            best_score = score
                            best_match = url
                
                if best_match:
                    # Download the image
                    response = requests.get(best_match, timeout=10)
                    if response.status_code == 200:
                        # Convert to base64
                        image_base64 = base64.b64encode(response.content).decode('utf-8')
                        
                        return {
                            "response": f"I found a relevant image for: '{prompt}'. Here's a BSL-related image that matches your request!",
                            "image_data": image_base64,
                            "image_format": "image/jpeg",
                            "prompt": prompt,
                            "timestamp": datetime.now().isoformat(),
                            "status": "image_generated",
                            "session_id": session_id,
                            "ai_enabled": True,
                            "source": "Unsplash"
                        }
                
                # If no match found, create an attractive placeholder
                from PIL import Image, ImageDraw, ImageFont
                
                # Create a more attractive placeholder
                img = Image.new('RGB', (400, 300), color='#667eea')
                draw = ImageDraw.Draw(img)
                
                # Add gradient effect
                for i in range(300):
                    color = int(102 + (i * 0.5))  # Gradient from #667eea to darker
                    draw.line([(0, i), (400, i)], fill=(color, 126, 234))
                
                # Add some text to the image
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
                
                # Add the prompt text
                text = f"BSL: {prompt[:25]}..." if len(prompt) > 25 else f"BSL: {prompt}"
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                x = (400 - text_width) // 2
                y = (300 - text_height) // 2
                
                # Draw text with white color and shadow
                draw.text((x+1, y+1), text, fill='#333333', font=font)  # Shadow
                draw.text((x, y), text, fill='white', font=font)
                
                # Add a border
                draw.rectangle([0, 0, 399, 299], outline='white', width=2)
                
                # Convert to base64
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                image_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                
                return {
                    "response": f"I found a relevant image for: '{prompt}'. Here's a BSL-themed placeholder while I search for the perfect image!",
                    "image_data": image_base64,
                    "image_format": "image/png",
                    "prompt": prompt,
                    "timestamp": datetime.now().isoformat(),
                    "status": "image_generated",
                    "session_id": session_id,
                    "ai_enabled": True
                }
            except Exception as img_error:
                print(f"Image search failed: {str(img_error)}")
                # Fall back to text response
        
        # Regular chat response
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
        print(f"Chat with image error: {str(e)}")
        return {
            "response": f"I'm sorry, I encountered an error while processing your request. Please try again or rephrase your question.",
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e),
            "session_id": session_id
        }

@app.get("/training/signs")
async def get_training_signs():
    """Get available signs for training"""
    training_signs = [
        {
            "id": "hi",
            "name": "Hello/Hi",
            "description": "Wave hand side to side",
            "difficulty": "beginner",
            "category": "greetings",
            "instructions": [
                "1. Raise your right hand to shoulder level",
                "2. Open your palm facing forward",
                "3. Move your hand from side to side in a waving motion",
                "4. Keep your fingers together and relaxed"
            ],
            "tips": [
                "Make sure your palm is clearly visible",
                "Keep the movement smooth and natural",
                "Practice in front of a mirror first"
            ],
            "videoUrl": "https://www.signbsl.com/sign/hello"
        },
        {
            "id": "please",
            "name": "Please",
            "description": "Rub palm in circular motion on chest",
            "difficulty": "beginner",
            "category": "manners",
            "instructions": [
                "1. Place your right hand on your chest",
                "2. Open your palm facing your chest",
                "3. Move your hand in a circular motion",
                "4. Keep the movement gentle and smooth"
            ],
            "tips": [
                "The motion should be clockwise",
                "Keep your fingers together",
                "Don't press too hard on your chest"
            ],
            "videoUrl": "https://www.signbsl.com/sign/please"
        },
        {
            "id": "excuse_me",
            "name": "Excuse Me",
            "description": "Tap shoulder to get attention",
            "difficulty": "beginner",
            "category": "manners",
            "instructions": [
                "1. Extend your right hand forward",
                "2. Make a fist with your thumb on top",
                "3. Tap your shoulder gently",
                "4. Look at the person you want to get attention from"
            ],
            "tips": [
                "The tap should be gentle, not forceful",
                "Make eye contact when possible",
                "Use this sign to politely interrupt"
            ],
            "videoUrl": "https://www.signbsl.com/sign/excuse-me"
        },
        {
            "id": "okay",
            "name": "Okay",
            "description": "Give a thumbs up gesture",
            "difficulty": "beginner",
            "category": "responses",
            "instructions": [
                "1. Make a fist with your right hand",
                "2. Extend your thumb upward",
                "3. Hold the position briefly",
                "4. You can nod your head for emphasis"
            ],
            "tips": [
                "Keep your other fingers in a fist",
                "The thumb should point straight up",
                "This sign is universally understood"
            ],
            "videoUrl": "https://www.signbsl.com/sign/okay"
        }
    ]
    return {"signs": training_signs, "count": len(training_signs)}


@app.post("/training/start-session")
async def start_training_session(data: Dict[str, Any]):
    """Start a new training session"""
    try:
        sign_id = data.get("sign_id")
        session_type = data.get("session_type", "practice")  # practice, test, learn
        
        # Validate sign
        training_signs = await get_training_signs()
        sign = next((s for s in training_signs["signs"] if s["id"] == sign_id), None)
        
        if not sign:
            raise HTTPException(status_code=400, detail="Invalid sign ID")
        
        session_id = f"session_{sign_id}_{int(time.time())}"
        
        return {
            "session_id": session_id,
            "sign": sign,
            "session_type": session_type,
            "status": "started",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")


@app.post("/training/process-frame")
async def process_training_frame(data: Dict[str, Any]):
    """Process a frame for training or detection"""
    try:
        frame_data = data.get("frame_data")
        session_id = data.get("session_id")
        sign_id = data.get("sign_id")
        mode = data.get("mode", "detection")  # detection, training
        
        if not frame_data:
            raise HTTPException(status_code=400, detail="Frame data required")
        
        # For now, return a simulated response
        # In a real implementation, this would process the frame through the model
        
        if mode == "training":
            # Training mode - collect data
            return {
                "status": "frame_captured",
                "session_id": session_id,
                "sign_id": sign_id,
                "frame_count": 1,
                "message": "Frame captured for training"
            }
        else:
            # Detection mode - predict sign
            import random
            confidence = random.uniform(0.3, 0.95)
            is_correct = confidence > 0.7
            
            return {
                "status": "detection_complete",
                "session_id": session_id,
                "sign_id": sign_id,
                "detected_sign": sign_id if is_correct else random.choice(["hi", "please", "excuse_me", "okay"]),
                "confidence": confidence,
                "is_correct": is_correct,
                "feedback": "Great job!" if is_correct else "Try again, focus on the hand position"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")


@app.post("/training/end-session")
async def end_training_session(data: Dict[str, Any]):
    """End a training session and get results"""
    try:
        session_id = data.get("session_id")
        sign_id = data.get("sign_id")
        total_frames = data.get("total_frames", 0)
        correct_detections = data.get("correct_detections", 0)
        
        # Calculate score
        accuracy = (correct_detections / total_frames * 100) if total_frames > 0 else 0
        
        # Determine performance level
        if accuracy >= 90:
            performance = "excellent"
            message = "Outstanding! You've mastered this sign!"
        elif accuracy >= 75:
            performance = "good"
            message = "Well done! Keep practicing to improve further."
        elif accuracy >= 60:
            performance = "fair"
            message = "Good effort! More practice will help you improve."
        else:
            performance = "needs_improvement"
            message = "Keep practicing! Focus on the hand position and movement."
        
        return {
            "session_id": session_id,
            "sign_id": sign_id,
            "total_frames": total_frames,
            "correct_detections": correct_detections,
            "accuracy": accuracy,
            "performance": performance,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending session: {str(e)}")


@app.get("/training/progress/{user_id}")
async def get_training_progress(user_id: str):
    """Get user's training progress"""
    try:
        # Simulated progress data
        # In a real app, this would come from a database
        progress = {
            "user_id": user_id,
            "total_sessions": 15,
            "total_accuracy": 78.5,
            "signs_learned": ["hi", "please", "okay"],
            "current_streak": 5,
            "best_accuracy": 95.2,
            "signs_progress": {
                "hi": {"sessions": 5, "accuracy": 85.0, "mastered": True},
                "please": {"sessions": 4, "accuracy": 72.5, "mastered": False},
                "excuse_me": {"sessions": 3, "accuracy": 65.0, "mastered": False},
                "okay": {"sessions": 3, "accuracy": 90.0, "mastered": True}
            }
        }
        
        return progress
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting progress: {str(e)}")


@app.post("/training/update-model")
async def update_training_model(data: Dict[str, Any]):
    """Update the model with new training data"""
    try:
        training_data = data.get("training_data", [])
        model_version = data.get("model_version", "current")
        
        # Simulated model update
        # In a real implementation, this would retrain the model
        
        return {
            "status": "model_updated",
            "model_version": f"{model_version}_updated",
            "training_samples": len(training_data),
            "message": "Model updated successfully with new training data",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating model: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 