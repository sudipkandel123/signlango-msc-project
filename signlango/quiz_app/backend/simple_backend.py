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


# Frontend-compatible endpoints (without /api prefix)
@app.get("/facts")
async def get_facts_frontend():
    """Frontend-compatible facts endpoint"""
    return await get_facts()


@app.get("/quiz")
async def get_quiz_frontend(category: str = "all", difficulty: str = "beginner"):
    """Frontend-compatible quiz endpoint with query parameters"""
    # Comprehensive BSL quiz data based on category and difficulty
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
async def submit_quiz_answer(data: dict):
    """Submit quiz answer and get feedback"""
    try:
        question_id = data.get("question_id")
        selected_answer = data.get("selected_answer")
        
        # Find the question from our quiz database
        quiz_questions = [
            # Basic Signs (basics)
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
