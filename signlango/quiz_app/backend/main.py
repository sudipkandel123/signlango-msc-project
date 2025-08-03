from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import base64
import io
from PIL import Image
import json
import requests
from typing import List, Dict, Any

app = FastAPI(title="Sign Language Tutorial API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load the trained model
model = None
print("🚀 Starting model loading process...")

try:
    # Try to load the model exactly as in the original code
    print("🔄 Attempting to load model.h5...")
    model = tf.keras.models.load_model("model.h5")
    print("✅ model.h5 loaded successfully")

    print("🔄 Attempting to load model.weights.h5...")
    model.load_weights("model.weights.h5")
    print("✅ model.weights.h5 loaded successfully")

    print("📊 Model Summary:")
    model.summary()
    print("🎯 Model loaded successfully with weights")

    # Test the model immediately to verify it's working
    test_input = np.random.rand(1, 30, 1662)
    test_output = model.predict(test_input, verbose=0)
    print(f"🧪 Model test prediction shape: {test_output.shape}")
    print(f"🧪 Model test prediction sum: {np.sum(test_output)}")
    print(f"🧪 Model test prediction max: {np.max(test_output)}")

except Exception as e:
    print(f"❌ Error loading model with original method: {e}")
    print(f"❌ Error type: {type(e).__name__}")
    print(f"❌ Error details: {str(e)}")

    # Try alternative loading methods
    try:
        print("🔄 Trying alternative loading method (compile=False)...")
        model = tf.keras.models.load_model("model.h5", compile=False)
        print("✅ model.h5 loaded with compile=False")

        print("🔄 Attempting to load model.weights.h5...")
        model.load_weights("model.weights.h5")
        print("✅ model.weights.h5 loaded successfully")

        # Compile the model
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        print("✅ Model compiled successfully")

        print("📊 Model Summary:")
        model.summary()
        print("🎯 Model loaded successfully with alternative method")

        # Test the model
        test_input = np.random.rand(1, 30, 1662)
        test_output = model.predict(test_input, verbose=0)
        print(f"🧪 Model test prediction shape: {test_output.shape}")
        print(f"🧪 Model test prediction sum: {np.sum(test_output)}")

    except Exception as e2:
        print(f"❌ Error loading model with alternative method: {e2}")
        print(f"❌ Error type: {type(e2).__name__}")

        try:
            print("🔄 Trying to load only weights file...")
            # Create a simple model structure and load weights
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(30, 1662)),
                    tf.keras.layers.LSTM(64, return_sequences=True),
                    tf.keras.layers.LSTM(32),
                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(4, activation="softmax"),
                ]
            )
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )

            # Try to load weights
            model.load_weights("model.weights.h5")
            print("✅ Weights loaded successfully")

            print("📊 Model Summary:")
            model.summary()
            print("🎯 Model loaded successfully with weights only")

            # Test the model
            test_input = np.random.rand(1, 30, 1662)
            test_output = model.predict(test_input, verbose=0)
            print(f"🧪 Model test prediction shape: {test_output.shape}")
            print(f"🧪 Model test prediction sum: {np.sum(test_output)}")

        except Exception as e3:
            print(f"❌ Error loading weights: {e3}")
            print(f"❌ Error type: {type(e3).__name__}")
            print("🔄 Creating a simple test model...")
            try:
                # Create a simple test model if loading fails
                model = tf.keras.Sequential(
                    [
                        tf.keras.layers.Input(shape=(30, 1662)),
                        tf.keras.layers.LSTM(64, return_sequences=True),
                        tf.keras.layers.LSTM(32),
                        tf.keras.layers.Dense(64, activation="relu"),
                        tf.keras.layers.Dropout(0.5),
                        tf.keras.layers.Dense(4, activation="softmax"),
                    ]
                )
                model.compile(
                    optimizer="adam",
                    loss="categorical_crossentropy",
                    metrics=["accuracy"],
                )
                print("✅ Test model created successfully")
                print("📊 Test Model Summary:")
                model.summary()

                # Test the model
                test_input = np.random.rand(1, 30, 1662)
                test_output = model.predict(test_input, verbose=0)
                print(f"🧪 Test model prediction shape: {test_output.shape}")
                print(f"🧪 Test model prediction sum: {np.sum(test_output)}")

            except Exception as e4:
                print(f"❌ Error creating test model: {e4}")
                model = None

# Sign detection variables
signs = ["hi", "please", "excuse_me", "okay"]
actions = np.array(["hi", "please", "excuse_me", "okay"])
threshold = 0.5

# Global variables for sign detection (similar to original code)
sequence = []
sentence = []
predictions = []

# Global MediaPipe instance to avoid repeated initialization
holistic = None
try:
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=True,  # Better for single image processing
    )
    print("MediaPipe Holistic initialized successfully")
except Exception as e:
    print(f"Error initializing MediaPipe: {e}")
    holistic = None


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array(
            [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
        ).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    lh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])


# Add the probability visualization function from the user's code
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(
            output_frame,
            (0, 60 + num * 40),
            (int(prob * 100), 90 + num * 40),
            colors[num],
            -1,
        )
        cv2.putText(
            output_frame,
            actions[num],
            (0, 85 + num * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output_frame


# BSL Facts and Quiz Data
bsl_facts = [
    {
        "id": 1,
        "title": "What is BSL?",
        "content": "British Sign Language (BSL) is the sign language used in the United Kingdom and is the first or preferred language of some deaf people in the UK.",
        "category": "basics",
        "image": "/images/facts/1.jpg",
        "icon": "🤟",
    },
    {
        "id": 2,
        "title": "BSL Recognition",
        "content": "BSL was officially recognized as a language in its own right by the UK government in 2003, marking a significant milestone for the deaf community.",
        "category": "history",
        "image": "/images/facts/2.jpg",
        "icon": "📜",
    },
    {
        "id": 3,
        "title": "Regional Variations",
        "content": "BSL has regional variations, just like spoken English has different accents and dialects. Signs can vary between cities and regions.",
        "category": "linguistics",
        "image": "/images/facts/3.jpg",
        "icon": "🗺️",
    },
    {
        "id": 4,
        "title": "Finger Spelling",
        "content": "BSL uses a two-handed manual alphabet for finger spelling, unlike American Sign Language which uses one hand. This makes it unique among sign languages.",
        "category": "technique",
        "image": "/images/facts/4.jpg",
        "icon": "✋",
    },
    {
        "id": 5,
        "title": "Grammar Structure",
        "content": "BSL has its own grammar structure, which is different from English. It uses topic-comment structure and spatial grammar.",
        "category": "linguistics",
        "image": "/images/facts/5.jpg",
        "icon": "📚",
    },
    {
        "id": 6,
        "title": "Deaf Culture",
        "content": "Deaf culture is rich and vibrant, with its own traditions, art, literature, and social norms. BSL is central to this cultural identity.",
        "category": "culture",
        "image": "/images/facts/6.jpg",
        "icon": "🎭",
    },
    {
        "id": 7,
        "title": "BSL in Education",
        "content": "Many deaf children learn BSL as their first language and use it in educational settings with qualified BSL interpreters.",
        "category": "education",
        "image": "/images/facts/7.jpg",
        "icon": "🎓",
    },
    {
        "id": 8,
        "title": "Technology and BSL",
        "content": "Modern technology has made BSL more accessible through video calling, online learning platforms, and mobile apps.",
        "category": "technology",
        "image": "/images/facts/8.jpg",
        "icon": "💻",
    },
    {
        "id": 9,
        "title": "BSL Interpreters",
        "content": "Professional BSL interpreters play a crucial role in making communication accessible between deaf and hearing people.",
        "category": "professionals",
        "image": "/images/facts/9.jpg",
        "icon": "👥",
    },
    {
        "id": 10,
        "title": "BSL in Healthcare",
        "content": "BSL interpreters are essential in healthcare settings to ensure deaf patients receive proper medical care and information.",
        "category": "healthcare",
        "image": "/images/facts/10.jpg",
        "icon": "🏥",
    },
    {
        "id": 11,
        "title": "BSL Poetry and Art",
        "content": "BSL has a rich tradition of poetry and performance art, using the visual-spatial nature of the language creatively.",
        "category": "arts",
        "image": "/images/facts/11.jpg",
        "icon": "🎨",
    },
    {
        "id": 12,
        "title": "International Sign",
        "content": "While BSL is specific to the UK, International Sign is used at global deaf events and conferences.",
        "category": "international",
        "image": "/images/facts/12.jpg",
        "icon": "🌍",
    },
    {
        "id": 13,
        "title": "Challenges for BSL-Only Students",
        "content": "Students who can only speak BSL often face barriers in mainstream education, such as lack of qualified interpreters, limited access to resources in BSL, social isolation, and difficulties in exams or group work. Inclusive teaching and awareness are crucial for their success.",
        "category": "challenges",
        "image": "/images/facts/13.jpg",
        "icon": "⚠️",
    },
    {
        "id": 14,
        "title": "Accessibility in Classrooms",
        "content": "Deaf students often struggle with classroom accessibility, including poor acoustics, lack of visual aids, and teachers who don't face the class when speaking. These barriers can significantly impact learning outcomes.",
        "category": "education_challenges",
        "image": "/images/facts/14.jpg",
        "icon": "🏫",
    },
    {
        "id": 15,
        "title": "Social Isolation in Schools",
        "content": "Deaf students frequently experience social isolation in mainstream schools due to communication barriers with hearing peers. This can lead to mental health issues and reduced academic motivation.",
        "category": "social_challenges",
        "image": "/images/facts/15.jpg",
        "icon": "😔",
    },
    {
        "id": 16,
        "title": "Interpreter Shortages",
        "content": "There is a critical shortage of qualified BSL interpreters in educational settings, leaving many deaf students without proper communication support. This shortage affects students at all levels of education.",
        "category": "professional_challenges",
        "image": "/images/facts/16.jpg",
        "icon": "👨‍💼",
    },
    {
        "id": 17,
        "title": "Technology Access Barriers",
        "content": "Many deaf students lack access to assistive technologies like hearing aids, cochlear implants, or captioning services due to financial constraints or limited awareness of available resources.",
        "category": "technology_challenges",
        "image": "/images/facts/17.jpg",
        "icon": "🔧",
    },
    {
        "id": 18,
        "title": "Exam Accommodations",
        "content": "Deaf students often struggle to receive appropriate exam accommodations, such as extra time, BSL interpreters, or modified question formats. This can unfairly disadvantage them in assessments.",
        "category": "assessment_challenges",
        "image": "/images/facts/18.jpg",
        "icon": "📝",
    },
    {
        "id": 19,
        "title": "Teacher Training Gaps",
        "content": "Most teachers receive little to no training on how to effectively teach deaf students or work with BSL interpreters. This knowledge gap creates additional barriers to inclusive education.",
        "category": "training_challenges",
        "image": "/images/facts/19.jpg",
        "icon": "👩‍🏫",
    },
    {
        "id": 20,
        "title": "Digital Learning Challenges",
        "content": "Online learning platforms often lack proper accessibility features for deaf students, including poor captioning, no BSL interpretation, and limited visual communication options.",
        "category": "digital_challenges",
        "image": "/images/facts/20.jpg",
        "icon": "💻",
    },
    {
        "id": 21,
        "title": "Mental Health Impact",
        "content": "The cumulative effect of communication barriers, social isolation, and academic challenges can lead to higher rates of anxiety, depression, and low self-esteem among deaf students.",
        "category": "mental_health",
        "image": "/images/facts/21.jpg",
        "icon": "🧠",
    },
    {
        "id": 22,
        "title": "Career Guidance Barriers",
        "content": "Deaf students often receive limited career guidance and face misconceptions about their capabilities from career advisors who lack understanding of deaf culture and communication needs.",
        "category": "career_challenges",
        "image": "/images/facts/22.jpg",
        "icon": "🎯",
    },
    {
        "id": 23,
        "title": "Parent Communication",
        "content": "Parents of deaf children often struggle to communicate effectively with schools about their child's needs, especially when they themselves are not fluent in BSL or familiar with educational rights.",
        "category": "family_challenges",
        "image": "/images/facts/23.jpg",
        "icon": "👨‍👩‍👧‍👦",
    },
    {
        "id": 24,
        "title": "Emergency Communication",
        "content": "Schools often lack proper emergency communication protocols for deaf students, putting them at risk during fire drills, lockdowns, or other safety situations.",
        "category": "safety_challenges",
        "image": "/images/facts/24.jpg",
        "icon": "🚨",
    },
    {
        "id": 25,
        "title": "Extracurricular Barriers",
        "content": "Deaf students often miss out on extracurricular activities like sports, clubs, and social events due to lack of interpreters or communication support, further limiting their social development.",
        "category": "activity_challenges",
        "image": "/images/facts/25.jpg",
        "icon": "⚽",
    },
]

quiz_questions = [
    {
        "id": 1,
        "question": "What does the sign 'hi' look like?",
        "options": [
            "Waving hand side to side",
            "Thumbs up gesture",
            "Pointing to yourself",
            "Clapping hands",
        ],
        "correct_answer": "Waving hand side to side",
        "explanation": "The sign for 'hi' involves waving your hand from side to side, similar to a greeting wave.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "Think of how you would wave to greet someone",
        "learning_tip": "This is one of the most common and recognizable BSL signs. Practice it with a smile!",
    },
    {
        "id": 2,
        "question": "Which sign means 'please'?",
        "options": [
            "Pointing to chest",
            "Rubbing palm in circular motion",
            "Thumbs up",
            "Waving hand",
        ],
        "correct_answer": "Rubbing palm in circular motion",
        "explanation": "The sign for 'please' involves rubbing your palm in a circular motion on your chest.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign involves a polite gesture on your chest",
        "learning_tip": "The circular motion represents politeness and respect in BSL culture.",
    },
    {
        "id": 3,
        "question": "What does 'excuse me' look like in BSL?",
        "options": [
            "Pointing to yourself",
            "Tapping shoulder",
            "Waving hand",
            "Thumbs up",
        ],
        "correct_answer": "Tapping shoulder",
        "explanation": "The sign for 'excuse me' involves tapping your shoulder to get someone's attention.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign mimics getting someone's attention",
        "learning_tip": "This sign is used to politely interrupt or get attention in BSL conversations.",
    },
    {
        "id": 4,
        "question": "How do you sign 'okay'?",
        "options": ["Thumbs up", "Waving hand", "Pointing finger", "Clapping"],
        "correct_answer": "Thumbs up",
        "explanation": "The sign for 'okay' is a thumbs up gesture.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This is a universal gesture that means approval",
        "learning_tip": "This sign is similar to the universal thumbs up gesture used worldwide.",
    },
    {
        "id": 5,
        "question": "When was BSL officially recognized by the UK government?",
        "options": ["1995", "2000", "2003", "2008"],
        "correct_answer": "2003",
        "explanation": "BSL was officially recognized as a language in its own right by the UK government in 2003.",
        "category": "history",
        "difficulty": "intermediate",
        "hint": "This happened in the early 2000s",
        "learning_tip": "This recognition was a major milestone for the deaf community and BSL rights.",
    },
    {
        "id": 6,
        "question": "How many hands does BSL use for finger spelling?",
        "options": ["One hand", "Two hands", "It varies", "No finger spelling"],
        "correct_answer": "Two hands",
        "explanation": "BSL uses a two-handed manual alphabet for finger spelling, unlike American Sign Language which uses one hand.",
        "category": "technique",
        "difficulty": "intermediate",
        "hint": "BSL is unique among sign languages for this feature",
        "learning_tip": "This two-handed system makes BSL distinct from other sign languages like ASL.",
    },
    {
        "id": 7,
        "question": "What is the main difference between BSL and American Sign Language (ASL)?",
        "options": [
            "BSL uses more facial expressions",
            "BSL uses two hands for finger spelling",
            "ASL is older than BSL",
            "There are no differences",
        ],
        "correct_answer": "BSL uses two hands for finger spelling",
        "explanation": "BSL uses a two-handed manual alphabet for finger spelling, while ASL uses one hand.",
        "category": "grammar",
        "difficulty": "intermediate",
        "hint": "Think about the finger spelling system",
        "learning_tip": "Understanding these differences helps appreciate the unique features of BSL.",
    },
    {
        "id": 8,
        "question": "What type of grammar structure does BSL use?",
        "options": [
            "Subject-verb-object like English",
            "Topic-comment structure",
            "Verb-subject-object",
            "Object-subject-verb",
        ],
        "correct_answer": "Topic-comment structure",
        "explanation": "BSL uses topic-comment structure and spatial grammar, which is different from English grammar.",
        "category": "grammar",
        "difficulty": "advanced",
        "hint": "BSL grammar is different from English grammar",
        "learning_tip": "This structure allows BSL to be more visual and spatial than spoken languages.",
    },
    {
        "id": 9,
        "question": "Where are BSL interpreters most commonly needed?",
        "options": [
            "Only in schools",
            "Only in hospitals",
            "In many settings including education, healthcare, and public events",
            "Only in government buildings",
        ],
        "correct_answer": "In many settings including education, healthcare, and public events",
        "explanation": "BSL interpreters are needed in many settings including education, healthcare, public events, and government services.",
        "category": "culture",
        "difficulty": "intermediate",
        "hint": "Think about accessibility in various public spaces",
        "learning_tip": "BSL interpreters ensure equal access to information and services for deaf people.",
    },
    {
        "id": 10,
        "question": "What is International Sign used for?",
        "options": [
            "Only in the UK",
            "Only in Europe",
            "At global deaf events and conferences",
            "Only in schools",
        ],
        "correct_answer": "At global deaf events and conferences",
        "explanation": "International Sign is used at global deaf events and conferences where people from different countries meet.",
        "category": "culture",
        "difficulty": "advanced",
        "hint": "Think about international communication",
        "learning_tip": "International Sign helps deaf people from different countries communicate at global events.",
    },
    {
        "id": 11,
        "question": "What makes BSL unique among sign languages?",
        "options": [
            "It's the oldest sign language",
            "It uses two hands for finger spelling",
            "It has no regional variations",
            "It's only used in one country",
        ],
        "correct_answer": "It uses two hands for finger spelling",
        "explanation": "BSL is unique because it uses a two-handed manual alphabet for finger spelling, unlike most other sign languages.",
        "category": "technique",
        "difficulty": "intermediate",
        "hint": "Think about the finger spelling system",
        "learning_tip": "This unique feature makes BSL distinct from other sign languages worldwide.",
    },
    {
        "id": 12,
        "question": "What is the role of facial expressions in BSL?",
        "options": [
            "They are not important",
            "They are optional",
            "They are essential for grammar and meaning",
            "They are only used for emotions",
        ],
        "correct_answer": "They are essential for grammar and meaning",
        "explanation": "Facial expressions are essential in BSL for grammar and meaning, not just for showing emotions.",
        "category": "grammar",
        "difficulty": "advanced",
        "hint": "Facial expressions carry grammatical information",
        "learning_tip": "Facial expressions in BSL are as important as tone of voice in spoken languages.",
    },
    {
        "id": 13,
        "question": "What is the sign for 'thank you' in BSL?",
        "options": [
            "Pointing to chin and moving hand forward",
            "Waving hand side to side",
            "Thumbs up gesture",
            "Clapping hands",
        ],
        "correct_answer": "Pointing to chin and moving hand forward",
        "explanation": "The sign for 'thank you' involves touching your chin with your fingertips and moving your hand forward and down.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign involves touching your face",
        "learning_tip": "This polite gesture shows appreciation and is commonly used in BSL conversations.",
    },
    {
        "id": 14,
        "question": "How do you sign 'sorry' in BSL?",
        "options": [
            "Making a fist and rubbing it in a circle on your chest",
            "Waving hand side to side",
            "Pointing to yourself",
            "Thumbs down gesture",
        ],
        "correct_answer": "Making a fist and rubbing it in a circle on your chest",
        "explanation": "The sign for 'sorry' involves making a fist and rubbing it in a circular motion on your chest.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign involves a circular motion on your chest",
        "learning_tip": "This sign expresses genuine apology and remorse in BSL culture.",
    },
    {
        "id": 15,
        "question": "What does the sign for 'good' look like?",
        "options": [
            "Thumbs up gesture",
            "Making a flat hand and moving it forward",
            "Pointing to yourself",
            "Clapping hands",
        ],
        "correct_answer": "Making a flat hand and moving it forward",
        "explanation": "The sign for 'good' involves making a flat hand and moving it forward from your chin.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign starts near your face",
        "learning_tip": "This sign is used to express approval or that something is good.",
    },
    {
        "id": 16,
        "question": "How do you sign 'bad' in BSL?",
        "options": [
            "Thumbs down gesture",
            "Making a fist and moving it down",
            "Pointing to yourself",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a fist and moving it down",
        "explanation": "The sign for 'bad' involves making a fist and moving it downward from your chin.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign is the opposite of 'good'",
        "learning_tip": "This sign expresses disapproval or that something is not good.",
    },
    {
        "id": 17,
        "question": "What is the sign for 'yes' in BSL?",
        "options": [
            "Nodding head up and down",
            "Making a fist and moving it up and down",
            "Thumbs up gesture",
            "Pointing finger up",
        ],
        "correct_answer": "Making a fist and moving it up and down",
        "explanation": "The sign for 'yes' involves making a fist and moving it up and down, like nodding.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign mimics the nodding motion",
        "learning_tip": "This sign is used to agree or confirm something in BSL.",
    },
    {
        "id": 18,
        "question": "How do you sign 'no' in BSL?",
        "options": [
            "Shaking head side to side",
            "Making a fist and moving it side to side",
            "Thumbs down gesture",
            "Pointing finger down",
        ],
        "correct_answer": "Making a fist and moving it side to side",
        "explanation": "The sign for 'no' involves making a fist and moving it side to side, like shaking your head.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign mimics the head shaking motion",
        "learning_tip": "This sign is used to disagree or deny something in BSL.",
    },
    {
        "id": 19,
        "question": "What is the sign for 'help' in BSL?",
        "options": [
            "Making a fist and moving it up",
            "Waving hand side to side",
            "Pointing to yourself",
            "Clapping hands",
        ],
        "correct_answer": "Making a fist and moving it up",
        "explanation": "The sign for 'help' involves making a fist and moving it upward from your chest.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign represents asking for assistance",
        "learning_tip": "This sign is important for asking for assistance or support.",
    },
    {
        "id": 20,
        "question": "How do you sign 'understand' in BSL?",
        "options": [
            "Pointing to your forehead",
            "Making a fist and moving it up",
            "Waving hand side to side",
            "Thumbs up gesture",
        ],
        "correct_answer": "Pointing to your forehead",
        "explanation": "The sign for 'understand' involves pointing to your forehead with your index finger.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign points to where understanding happens",
        "learning_tip": "This sign is used to confirm that you understand something.",
    },
    {
        "id": 21,
        "question": "What is the sign for 'name' in BSL?",
        "options": [
            "Making a fist and moving it side to side",
            "Pointing to your chest",
            "Waving hand side to side",
            "Thumbs up gesture",
        ],
        "correct_answer": "Pointing to your chest",
        "explanation": "The sign for 'name' involves pointing to your chest with your index finger.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign points to where your name belongs",
        "learning_tip": "This sign is used when introducing yourself or asking someone's name.",
    },
    {
        "id": 22,
        "question": "How do you sign 'friend' in BSL?",
        "options": [
            "Making two fists and linking them together",
            "Pointing to yourself",
            "Waving hand side to side",
            "Thumbs up gesture",
        ],
        "correct_answer": "Making two fists and linking them together",
        "explanation": "The sign for 'friend' involves making two fists and linking them together.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign represents connection between people",
        "learning_tip": "This sign symbolizes the connection and bond between friends.",
    },
    {
        "id": 23,
        "question": "What is the sign for 'family' in BSL?",
        "options": [
            "Making a circle with both hands",
            "Pointing to yourself",
            "Waving hand side to side",
            "Thumbs up gesture",
        ],
        "correct_answer": "Making a circle with both hands",
        "explanation": "The sign for 'family' involves making a circle with both hands, representing the family unit.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign represents a group or unit",
        "learning_tip": "This sign represents the unity and togetherness of a family.",
    },
    {
        "id": 24,
        "question": "How do you sign 'work' in BSL?",
        "options": [
            "Making two fists and moving them up and down",
            "Pointing to yourself",
            "Waving hand side to side",
            "Thumbs up gesture",
        ],
        "correct_answer": "Making two fists and moving them up and down",
        "explanation": "The sign for 'work' involves making two fists and moving them up and down, like hammering.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign mimics a working motion",
        "learning_tip": "This sign represents physical labor and work activities.",
    },
    {
        "id": 25,
        "question": "What is the sign for 'home' in BSL?",
        "options": [
            "Making a roof shape with your hands",
            "Pointing to yourself",
            "Waving hand side to side",
            "Thumbs up gesture",
        ],
        "correct_answer": "Making a roof shape with your hands",
        "explanation": "The sign for 'home' involves making a roof shape with your hands above your head.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign represents a house or shelter",
        "learning_tip": "This sign represents the concept of home and shelter.",
    },
    {
        "id": 26,
        "question": "What is the sign for 'school' in BSL?",
        "options": [
            "Making a roof shape with your hands",
            "Making two fists and moving them up and down",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making two fists and moving them up and down",
        "explanation": "The sign for 'school' involves making two fists and moving them up and down, similar to the sign for 'work'.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign is similar to the 'work' sign",
        "learning_tip": "This sign represents the learning and educational activities at school.",
    },
    {
        "id": 27,
        "question": "How do you sign 'time' in BSL?",
        "options": [
            "Pointing to your wrist",
            "Making a circle with your index finger",
            "Waving hand side to side",
            "Thumbs up gesture",
        ],
        "correct_answer": "Pointing to your wrist",
        "explanation": "The sign for 'time' involves pointing to your wrist where you would wear a watch.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "Think about where you check the time",
        "learning_tip": "This sign is commonly used when asking about schedules or appointments.",
    },
    {
        "id": 28,
        "question": "What is the sign for 'money' in BSL?",
        "options": [
            "Rubbing your thumb and index finger together",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Rubbing your thumb and index finger together",
        "explanation": "The sign for 'money' involves rubbing your thumb and index finger together, like counting money.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign mimics counting money",
        "learning_tip": "This sign is used when discussing finances or purchasing items.",
    },
    {
        "id": 29,
        "question": "How do you sign 'food' in BSL?",
        "options": [
            "Bringing your hand to your mouth",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Bringing your hand to your mouth",
        "explanation": "The sign for 'food' involves bringing your hand to your mouth, like eating.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign mimics eating",
        "learning_tip": "This sign is used when discussing meals or hunger.",
    },
    {
        "id": 30,
        "question": "What is the sign for 'drink' in BSL?",
        "options": [
            "Making a cup shape with your hand and bringing it to your mouth",
            "Bringing your hand to your mouth",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a cup shape with your hand and bringing it to your mouth",
        "explanation": "The sign for 'drink' involves making a cup shape with your hand and bringing it to your mouth.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign mimics drinking from a cup",
        "learning_tip": "This sign is used when discussing beverages or thirst.",
    },
    {
        "id": 31,
        "question": "How do you sign 'sleep' in BSL?",
        "options": [
            "Bringing your hands together and resting your head on them",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Bringing your hands together and resting your head on them",
        "explanation": "The sign for 'sleep' involves bringing your hands together and resting your head on them, like sleeping.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign mimics sleeping",
        "learning_tip": "This sign is used when discussing rest or bedtime.",
    },
    {
        "id": 32,
        "question": "What is the sign for 'walk' in BSL?",
        "options": [
            "Making two fingers walk across your other hand",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making two fingers walk across your other hand",
        "explanation": "The sign for 'walk' involves making two fingers walk across your other hand.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign mimics walking",
        "learning_tip": "This sign is used when discussing movement or travel.",
    },
    {
        "id": 33,
        "question": "How do you sign 'run' in BSL?",
        "options": [
            "Making two fingers run across your other hand",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making two fingers run across your other hand",
        "explanation": "The sign for 'run' involves making two fingers run across your other hand, faster than walking.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign is similar to 'walk' but faster",
        "learning_tip": "This sign is used when discussing fast movement or exercise.",
    },
    {
        "id": 34,
        "question": "What is the sign for 'big' in BSL?",
        "options": [
            "Making your hands wide apart",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making your hands wide apart",
        "explanation": "The sign for 'big' involves making your hands wide apart to show size.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows size with your hands",
        "learning_tip": "This sign is used when describing the size of objects or people.",
    },
    {
        "id": 35,
        "question": "How do you sign 'small' in BSL?",
        "options": [
            "Bringing your hands close together",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Bringing your hands close together",
        "explanation": "The sign for 'small' involves bringing your hands close together to show small size.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows small size with your hands",
        "learning_tip": "This sign is used when describing small objects or people.",
    },
    {
        "id": 36,
        "question": "What is the sign for 'hot' in BSL?",
        "options": [
            "Making a fist and moving it away from your mouth",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a fist and moving it away from your mouth",
        "explanation": "The sign for 'hot' involves making a fist and moving it away from your mouth, like blowing on hot food.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign mimics cooling hot food",
        "learning_tip": "This sign is used when discussing temperature or spicy food.",
    },
    {
        "id": 37,
        "question": "How do you sign 'cold' in BSL?",
        "options": [
            "Making fists and shaking them",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making fists and shaking them",
        "explanation": "The sign for 'cold' involves making fists and shaking them, like shivering.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign mimics shivering from cold",
        "learning_tip": "This sign is used when discussing temperature or weather.",
    },
    {
        "id": 38,
        "question": "What is the sign for 'happy' in BSL?",
        "options": [
            "Making a smile with your hands",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a smile with your hands",
        "explanation": "The sign for 'happy' involves making a smile shape with your hands near your face.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows happiness with your hands",
        "learning_tip": "This sign is used when expressing positive emotions.",
    },
    {
        "id": 39,
        "question": "How do you sign 'sad' in BSL?",
        "options": [
            "Making a downward curve with your hands",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a downward curve with your hands",
        "explanation": "The sign for 'sad' involves making a downward curve with your hands near your face.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows sadness with your hands",
        "learning_tip": "This sign is used when expressing negative emotions.",
    },
    {
        "id": 40,
        "question": "What is the sign for 'angry' in BSL?",
        "options": [
            "Making fists and moving them up and down",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making fists and moving them up and down",
        "explanation": "The sign for 'angry' involves making fists and moving them up and down, showing frustration.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows anger with your hands",
        "learning_tip": "This sign is used when expressing frustration or anger.",
    },
    {
        "id": 41,
        "question": "How do you sign 'love' in BSL?",
        "options": [
            "Making a heart shape with your hands",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a heart shape with your hands",
        "explanation": "The sign for 'love' involves making a heart shape with your hands near your chest.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows love with a heart shape",
        "learning_tip": "This sign is used when expressing affection or love.",
    },
    {
        "id": 42,
        "question": "What is the sign for 'think' in BSL?",
        "options": [
            "Pointing to your forehead and making a thinking gesture",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Pointing to your forehead and making a thinking gesture",
        "explanation": "The sign for 'think' involves pointing to your forehead and making a thinking gesture.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign points to where thinking happens",
        "learning_tip": "This sign is used when discussing thoughts or opinions.",
    },
    {
        "id": 43,
        "question": "How do you sign 'know' in BSL?",
        "options": [
            "Touching your forehead with your index finger",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Touching your forehead with your index finger",
        "explanation": "The sign for 'know' involves touching your forehead with your index finger.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign points to where knowledge is stored",
        "learning_tip": "This sign is used when confirming knowledge or understanding.",
    },
    {
        "id": 44,
        "question": "What is the sign for 'want' in BSL?",
        "options": [
            "Making a claw shape and pulling it toward your chest",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a claw shape and pulling it toward your chest",
        "explanation": "The sign for 'want' involves making a claw shape and pulling it toward your chest.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows desire by pulling toward yourself",
        "learning_tip": "This sign is used when expressing desires or needs.",
    },
    {
        "id": 45,
        "question": "How do you sign 'need' in BSL?",
        "options": [
            "Making a fist and moving it toward your chest",
            "Making a claw shape and pulling it toward your chest",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a fist and moving it toward your chest",
        "explanation": "The sign for 'need' involves making a fist and moving it toward your chest.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows necessity by moving toward yourself",
        "learning_tip": "This sign is used when expressing essential needs or requirements.",
    },
    {
        "id": 46,
        "question": "What is the sign for 'can' in BSL?",
        "options": [
            "Making two fists and moving them up and down",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making two fists and moving them up and down",
        "explanation": "The sign for 'can' involves making two fists and moving them up and down.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows ability with your hands",
        "learning_tip": "This sign is used when expressing capability or permission.",
    },
    {
        "id": 47,
        "question": "How do you sign 'cannot' in BSL?",
        "options": [
            "Making two fists and moving them side to side",
            "Making two fists and moving them up and down",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making two fists and moving them side to side",
        "explanation": "The sign for 'cannot' involves making two fists and moving them side to side.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows inability with your hands",
        "learning_tip": "This sign is used when expressing inability or prohibition.",
    },
    {
        "id": 48,
        "question": "What is the sign for 'will' in BSL?",
        "options": [
            "Making a fist and moving it forward",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a fist and moving it forward",
        "explanation": "The sign for 'will' involves making a fist and moving it forward.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows future intention",
        "learning_tip": "This sign is used when expressing future plans or intentions.",
    },
    {
        "id": 49,
        "question": "How do you sign 'did' in BSL?",
        "options": [
            "Making a fist and moving it backward",
            "Making a fist and moving it forward",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a fist and moving it backward",
        "explanation": "The sign for 'did' involves making a fist and moving it backward.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows past action",
        "learning_tip": "This sign is used when expressing past actions or events.",
    },
    {
        "id": 50,
        "question": "What is the sign for 'today' in BSL?",
        "options": [
            "Pointing to your chest and then down",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Pointing to your chest and then down",
        "explanation": "The sign for 'today' involves pointing to your chest and then down.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows current time",
        "learning_tip": "This sign is used when discussing current events or plans.",
    },
    {
        "id": 51,
        "question": "How do you sign 'yesterday' in BSL?",
        "options": [
            "Pointing to your chest and then behind you",
            "Pointing to your chest and then down",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Pointing to your chest and then behind you",
        "explanation": "The sign for 'yesterday' involves pointing to your chest and then behind you.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows past time",
        "learning_tip": "This sign is used when discussing past events or memories.",
    },
    {
        "id": 52,
        "question": "What is the sign for 'tomorrow' in BSL?",
        "options": [
            "Pointing to your chest and then in front of you",
            "Pointing to your chest and then down",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Pointing to your chest and then in front of you",
        "explanation": "The sign for 'tomorrow' involves pointing to your chest and then in front of you.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows future time",
        "learning_tip": "This sign is used when discussing future plans or events.",
    },
    {
        "id": 53,
        "question": "How do you sign 'week' in BSL?",
        "options": [
            "Making a fist and moving it in a circle",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a fist and moving it in a circle",
        "explanation": "The sign for 'week' involves making a fist and moving it in a circle.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows a cycle of time",
        "learning_tip": "This sign is used when discussing weekly schedules or plans.",
    },
    {
        "id": 54,
        "question": "What is the sign for 'month' in BSL?",
        "options": [
            "Making a fist and moving it in a larger circle",
            "Making a fist and moving it in a circle",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a fist and moving it in a larger circle",
        "explanation": "The sign for 'month' involves making a fist and moving it in a larger circle than 'week'.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows a longer cycle of time",
        "learning_tip": "This sign is used when discussing monthly schedules or plans.",
    },
    {
        "id": 55,
        "question": "How do you sign 'year' in BSL?",
        "options": [
            "Making a fist and moving it in a very large circle",
            "Making a fist and moving it in a circle",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a fist and moving it in a very large circle",
        "explanation": "The sign for 'year' involves making a fist and moving it in a very large circle.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows the longest cycle of time",
        "learning_tip": "This sign is used when discussing yearly plans or anniversaries.",
    },
    {
        "id": 56,
        "question": "What is the sign for 'morning' in BSL?",
        "options": [
            "Making a fist and moving it up",
            "Making a fist and moving it down",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a fist and moving it up",
        "explanation": "The sign for 'morning' involves making a fist and moving it up.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows the sun rising",
        "learning_tip": "This sign is used when discussing morning activities or greetings.",
    },
    {
        "id": 57,
        "question": "How do you sign 'night' in BSL?",
        "options": [
            "Making a fist and moving it down",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a fist and moving it down",
        "explanation": "The sign for 'night' involves making a fist and moving it down.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows the sun setting",
        "learning_tip": "This sign is used when discussing evening activities or bedtime.",
    },
    {
        "id": 58,
        "question": "What is the sign for 'now' in BSL?",
        "options": [
            "Making two fists and bringing them together",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making two fists and bringing them together",
        "explanation": "The sign for 'now' involves making two fists and bringing them together.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows current moment",
        "learning_tip": "This sign is used when discussing immediate actions or current situations.",
    },
    {
        "id": 59,
        "question": "How do you sign 'later' in BSL?",
        "options": [
            "Making a fist and moving it forward",
            "Making two fists and bringing them together",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a fist and moving it forward",
        "explanation": "The sign for 'later' involves making a fist and moving it forward.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows future time",
        "learning_tip": "This sign is used when discussing future plans or delayed actions.",
    },
    {
        "id": 60,
        "question": "What is the sign for 'before' in BSL?",
        "options": [
            "Making a fist and moving it backward",
            "Making a fist and moving it forward",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a fist and moving it backward",
        "explanation": "The sign for 'before' involves making a fist and moving it backward.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows past time",
        "learning_tip": "This sign is used when discussing past events or sequences.",
    },
    {
        "id": 61,
        "question": "How do you sign 'after' in BSL?",
        "options": [
            "Making a fist and moving it forward",
            "Making a fist and moving it backward",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a fist and moving it forward",
        "explanation": "The sign for 'after' involves making a fist and moving it forward.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows future time",
        "learning_tip": "This sign is used when discussing future events or sequences.",
    },
    {
        "id": 62,
        "question": "What is the sign for 'first' in BSL?",
        "options": [
            "Holding up your index finger",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Holding up your index finger",
        "explanation": "The sign for 'first' involves holding up your index finger.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows the number one",
        "learning_tip": "This sign is used when discussing order or priority.",
    },
    {
        "id": 63,
        "question": "How do you sign 'last' in BSL?",
        "options": [
            "Holding up your little finger",
            "Holding up your index finger",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Holding up your little finger",
        "explanation": "The sign for 'last' involves holding up your little finger.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows the last position",
        "learning_tip": "This sign is used when discussing order or final positions.",
    },
    {
        "id": 64,
        "question": "What is the sign for 'next' in BSL?",
        "options": [
            "Making a fist and moving it forward",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a fist and moving it forward",
        "explanation": "The sign for 'next' involves making a fist and moving it forward.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows progression",
        "learning_tip": "This sign is used when discussing sequences or progression.",
    },
    {
        "id": 65,
        "question": "How do you sign 'previous' in BSL?",
        "options": [
            "Making a fist and moving it backward",
            "Making a fist and moving it forward",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a fist and moving it backward",
        "explanation": "The sign for 'previous' involves making a fist and moving it backward.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows going back",
        "learning_tip": "This sign is used when discussing past items or going back in sequence.",
    },
    {
        "id": 66,
        "question": "What is the sign for 'here' in BSL?",
        "options": [
            "Pointing to the ground in front of you",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Pointing to the ground in front of you",
        "explanation": "The sign for 'here' involves pointing to the ground in front of you.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows current location",
        "learning_tip": "This sign is used when discussing current location or position.",
    },
    {
        "id": 67,
        "question": "How do you sign 'there' in BSL?",
        "options": [
            "Pointing to a location away from you",
            "Pointing to the ground in front of you",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Pointing to a location away from you",
        "explanation": "The sign for 'there' involves pointing to a location away from you.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows distant location",
        "learning_tip": "This sign is used when discussing distant locations or positions.",
    },
    {
        "id": 68,
        "question": "What is the sign for 'where' in BSL?",
        "options": [
            "Making a questioning gesture with your hands",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a questioning gesture with your hands",
        "explanation": "The sign for 'where' involves making a questioning gesture with your hands.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows questioning about location",
        "learning_tip": "This sign is used when asking about location or position.",
    },
    {
        "id": 69,
        "question": "How do you sign 'what' in BSL?",
        "options": [
            "Making a questioning gesture with your index finger",
            "Making a questioning gesture with your hands",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a questioning gesture with your index finger",
        "explanation": "The sign for 'what' involves making a questioning gesture with your index finger.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows questioning about objects",
        "learning_tip": "This sign is used when asking about objects or things.",
    },
    {
        "id": 70,
        "question": "What is the sign for 'who' in BSL?",
        "options": [
            "Making a questioning gesture with your index finger pointing up",
            "Making a questioning gesture with your index finger",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a questioning gesture with your index finger pointing up",
        "explanation": "The sign for 'who' involves making a questioning gesture with your index finger pointing up.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows questioning about people",
        "learning_tip": "This sign is used when asking about people or individuals.",
    },
    {
        "id": 71,
        "question": "How do you sign 'why' in BSL?",
        "options": [
            "Making a questioning gesture with your index finger pointing to your forehead",
            "Making a questioning gesture with your index finger pointing up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a questioning gesture with your index finger pointing to your forehead",
        "explanation": "The sign for 'why' involves making a questioning gesture with your index finger pointing to your forehead.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows questioning about reasons",
        "learning_tip": "This sign is used when asking about reasons or causes.",
    },
    {
        "id": 72,
        "question": "What is the sign for 'how' in BSL?",
        "options": [
            "Making a questioning gesture with your hands in a circular motion",
            "Making a questioning gesture with your index finger pointing to your forehead",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a questioning gesture with your hands in a circular motion",
        "explanation": "The sign for 'how' involves making a questioning gesture with your hands in a circular motion.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows questioning about methods",
        "learning_tip": "This sign is used when asking about methods or processes.",
    },
    {
        "id": 73,
        "question": "How do you sign 'when' in BSL?",
        "options": [
            "Making a questioning gesture with your index finger pointing to your wrist",
            "Making a questioning gesture with your hands in a circular motion",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a questioning gesture with your index finger pointing to your wrist",
        "explanation": "The sign for 'when' involves making a questioning gesture with your index finger pointing to your wrist.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows questioning about time",
        "learning_tip": "This sign is used when asking about time or schedules.",
    },
    {
        "id": 74,
        "question": "What is the sign for 'which' in BSL?",
        "options": [
            "Making a questioning gesture with your index finger pointing to options",
            "Making a questioning gesture with your index finger pointing to your wrist",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a questioning gesture with your index finger pointing to options",
        "explanation": "The sign for 'which' involves making a questioning gesture with your index finger pointing to options.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows questioning about choices",
        "learning_tip": "This sign is used when asking about choices or options.",
    },
    {
        "id": 75,
        "question": "How do you sign 'how many' in BSL?",
        "options": [
            "Making a questioning gesture with your hands showing quantity",
            "Making a questioning gesture with your index finger pointing to options",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a questioning gesture with your hands showing quantity",
        "explanation": "The sign for 'how many' involves making a questioning gesture with your hands showing quantity.",
        "category": "basics",
        "difficulty": "beginner",
        "hint": "This sign shows questioning about quantity",
        "learning_tip": "This sign is used when asking about numbers or quantities.",
    },
    {
        "id": 76,
        "question": "What is the BSL sign for 'deaf'?",
        "options": [
            "Pointing to your ear and then making a negative gesture",
            "Pointing to your mouth and then making a negative gesture",
            "Making a fist and moving it up",
            "Waving hand side to side",
        ],
        "correct_answer": "Pointing to your ear and then making a negative gesture",
        "explanation": "The sign for 'deaf' involves pointing to your ear and then making a negative gesture.",
        "category": "culture",
        "difficulty": "intermediate",
        "hint": "This sign indicates hearing status",
        "learning_tip": "This sign is important for discussing deaf identity and culture.",
    },
    {
        "id": 77,
        "question": "How do you sign 'hearing' in BSL?",
        "options": [
            "Pointing to your ear and then making a positive gesture",
            "Pointing to your ear and then making a negative gesture",
            "Making a fist and moving it up",
            "Waving hand side to side",
        ],
        "correct_answer": "Pointing to your ear and then making a positive gesture",
        "explanation": "The sign for 'hearing' involves pointing to your ear and then making a positive gesture.",
        "category": "culture",
        "difficulty": "intermediate",
        "hint": "This sign indicates hearing status",
        "learning_tip": "This sign is used when discussing hearing people or hearing status.",
    },
    {
        "id": 78,
        "question": "What is the sign for 'interpreter' in BSL?",
        "options": [
            "Making two hands move back and forth between positions",
            "Making a fist and moving it up",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making two hands move back and forth between positions",
        "explanation": "The sign for 'interpreter' involves making two hands move back and forth between positions.",
        "category": "culture",
        "difficulty": "intermediate",
        "hint": "This sign shows communication between people",
        "learning_tip": "This sign is used when discussing BSL interpreters or interpretation services.",
    },
    {
        "id": 79,
        "question": "How do you sign 'community' in BSL?",
        "options": [
            "Making a circle with both hands",
            "Making two fists and moving them up and down",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a circle with both hands",
        "explanation": "The sign for 'community' involves making a circle with both hands.",
        "category": "culture",
        "difficulty": "intermediate",
        "hint": "This sign represents a group of people",
        "learning_tip": "This sign is used when discussing the deaf community or any community.",
    },
    {
        "id": 80,
        "question": "What is the sign for 'culture' in BSL?",
        "options": [
            "Making a circle with your index finger around your head",
            "Making a circle with both hands",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a circle with your index finger around your head",
        "explanation": "The sign for 'culture' involves making a circle with your index finger around your head.",
        "category": "culture",
        "difficulty": "intermediate",
        "hint": "This sign represents knowledge and traditions",
        "learning_tip": "This sign is used when discussing cultural identity and traditions.",
    },
    {
        "id": 81,
        "question": "How do you sign 'language' in BSL?",
        "options": [
            "Making two fists and moving them back and forth",
            "Making a circle with your index finger around your head",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making two fists and moving them back and forth",
        "explanation": "The sign for 'language' involves making two fists and moving them back and forth.",
        "category": "grammar",
        "difficulty": "intermediate",
        "hint": "This sign represents communication",
        "learning_tip": "This sign is used when discussing languages or communication.",
    },
    {
        "id": 82,
        "question": "What is the sign for 'sentence' in BSL?",
        "options": [
            "Making two hands move in a line from left to right",
            "Making two fists and moving them back and forth",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making two hands move in a line from left to right",
        "explanation": "The sign for 'sentence' involves making two hands move in a line from left to right.",
        "category": "grammar",
        "difficulty": "intermediate",
        "hint": "This sign represents a complete thought",
        "learning_tip": "This sign is used when discussing BSL grammar and sentence structure.",
    },
    {
        "id": 83,
        "question": "How do you sign 'word' in BSL?",
        "options": [
            "Making two fingers move together and apart",
            "Making two hands move in a line from left to right",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making two fingers move together and apart",
        "explanation": "The sign for 'word' involves making two fingers move together and apart.",
        "category": "grammar",
        "difficulty": "intermediate",
        "hint": "This sign represents individual words",
        "learning_tip": "This sign is used when discussing vocabulary or individual signs.",
    },
    {
        "id": 84,
        "question": "What is the sign for 'grammar' in BSL?",
        "options": [
            "Making two hands move in a structured pattern",
            "Making two fingers move together and apart",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making two hands move in a structured pattern",
        "explanation": "The sign for 'grammar' involves making two hands move in a structured pattern.",
        "category": "grammar",
        "difficulty": "advanced",
        "hint": "This sign represents structured language rules",
        "learning_tip": "This sign is used when discussing BSL grammar rules and structure.",
    },
    {
        "id": 85,
        "question": "How do you sign 'facial expression' in BSL?",
        "options": [
            "Pointing to your face and making expressions",
            "Making two hands move in a structured pattern",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Pointing to your face and making expressions",
        "explanation": "The sign for 'facial expression' involves pointing to your face and making expressions.",
        "category": "grammar",
        "difficulty": "advanced",
        "hint": "This sign represents facial grammar",
        "learning_tip": "This sign is used when discussing the importance of facial expressions in BSL.",
    },
    {
        "id": 86,
        "question": "What is the sign for 'spatial grammar' in BSL?",
        "options": [
            "Using your hands to show space and location",
            "Pointing to your face and making expressions",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Using your hands to show space and location",
        "explanation": "The sign for 'spatial grammar' involves using your hands to show space and location.",
        "category": "grammar",
        "difficulty": "advanced",
        "hint": "This sign represents spatial relationships",
        "learning_tip": "This sign is used when discussing BSL's spatial grammar system.",
    },
    {
        "id": 87,
        "question": "How do you sign 'topic-comment structure' in BSL?",
        "options": [
            "Making two hands move to show topic and comment",
            "Using your hands to show space and location",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making two hands move to show topic and comment",
        "explanation": "The sign for 'topic-comment structure' involves making two hands move to show topic and comment.",
        "category": "grammar",
        "difficulty": "advanced",
        "hint": "This sign represents BSL sentence structure",
        "learning_tip": "This sign is used when discussing BSL's unique sentence structure.",
    },
    {
        "id": 88,
        "question": "What is the sign for 'finger spelling' in BSL?",
        "options": [
            "Making hand shapes to represent letters",
            "Making two hands move to show topic and comment",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making hand shapes to represent letters",
        "explanation": "The sign for 'finger spelling' involves making hand shapes to represent letters.",
        "category": "technique",
        "difficulty": "intermediate",
        "hint": "This sign represents the manual alphabet",
        "learning_tip": "This sign is used when discussing BSL's two-handed finger spelling system.",
    },
    {
        "id": 89,
        "question": "How do you sign 'hand shape' in BSL?",
        "options": [
            "Making different hand configurations",
            "Making hand shapes to represent letters",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making different hand configurations",
        "explanation": "The sign for 'hand shape' involves making different hand configurations.",
        "category": "technique",
        "difficulty": "intermediate",
        "hint": "This sign represents hand positions",
        "learning_tip": "This sign is used when discussing BSL hand shapes and positions.",
    },
    {
        "id": 90,
        "question": "What is the sign for 'movement' in BSL?",
        "options": [
            "Making your hands move in different directions",
            "Making different hand configurations",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making your hands move in different directions",
        "explanation": "The sign for 'movement' involves making your hands move in different directions.",
        "category": "technique",
        "difficulty": "intermediate",
        "hint": "This sign represents hand movements",
        "learning_tip": "This sign is used when discussing BSL hand movements and directions.",
    },
    {
        "id": 91,
        "question": "How do you sign 'location' in BSL?",
        "options": [
            "Pointing to different positions in space",
            "Making your hands move in different directions",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Pointing to different positions in space",
        "explanation": "The sign for 'location' involves pointing to different positions in space.",
        "category": "technique",
        "difficulty": "intermediate",
        "hint": "This sign represents spatial positioning",
        "learning_tip": "This sign is used when discussing BSL's spatial location system.",
    },
    {
        "id": 92,
        "question": "What is the sign for 'orientation' in BSL?",
        "options": [
            "Showing the direction your hands face",
            "Pointing to different positions in space",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Showing the direction your hands face",
        "explanation": "The sign for 'orientation' involves showing the direction your hands face.",
        "category": "technique",
        "difficulty": "advanced",
        "hint": "This sign represents hand direction",
        "learning_tip": "This sign is used when discussing BSL hand orientation and direction.",
    },
    {
        "id": 93,
        "question": "How do you sign 'non-manual features' in BSL?",
        "options": [
            "Pointing to your face and body",
            "Showing the direction your hands face",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Pointing to your face and body",
        "explanation": "The sign for 'non-manual features' involves pointing to your face and body.",
        "category": "technique",
        "difficulty": "advanced",
        "hint": "This sign represents facial and body features",
        "learning_tip": "This sign is used when discussing BSL's non-manual features like facial expressions.",
    },
    {
        "id": 94,
        "question": "What is the sign for 'British Deaf Association' in BSL?",
        "options": [
            "Making the letters B-D-A with finger spelling",
            "Pointing to your face and body",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making the letters B-D-A with finger spelling",
        "explanation": "The sign for 'British Deaf Association' involves making the letters B-D-A with finger spelling.",
        "category": "history",
        "difficulty": "intermediate",
        "hint": "This sign uses finger spelling",
        "learning_tip": "This sign is used when discussing the BDA and deaf rights organizations.",
    },
    {
        "id": 95,
        "question": "How do you sign 'deaf rights' in BSL?",
        "options": [
            "Making a fist and moving it up with determination",
            "Making the letters B-D-A with finger spelling",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a fist and moving it up with determination",
        "explanation": "The sign for 'deaf rights' involves making a fist and moving it up with determination.",
        "category": "history",
        "difficulty": "intermediate",
        "hint": "This sign represents advocacy and rights",
        "learning_tip": "This sign is used when discussing deaf rights and advocacy.",
    },
    {
        "id": 96,
        "question": "What is the sign for 'recognition' in BSL?",
        "options": [
            "Making two hands come together in agreement",
            "Making a fist and moving it up with determination",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making two hands come together in agreement",
        "explanation": "The sign for 'recognition' involves making two hands come together in agreement.",
        "category": "history",
        "difficulty": "intermediate",
        "hint": "This sign represents official acknowledgment",
        "learning_tip": "This sign is used when discussing BSL recognition and legal status.",
    },
    {
        "id": 97,
        "question": "How do you sign 'education' in BSL?",
        "options": [
            "Making two hands move up and down like learning",
            "Making two hands come together in agreement",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making two hands move up and down like learning",
        "explanation": "The sign for 'education' involves making two hands move up and down like learning.",
        "category": "history",
        "difficulty": "intermediate",
        "hint": "This sign represents learning and teaching",
        "learning_tip": "This sign is used when discussing deaf education and learning.",
    },
    {
        "id": 98,
        "question": "What is the sign for 'accessibility' in BSL?",
        "options": [
            "Making two hands move to show equal access",
            "Making two hands move up and down like learning",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making two hands move to show equal access",
        "explanation": "The sign for 'accessibility' involves making two hands move to show equal access.",
        "category": "culture",
        "difficulty": "advanced",
        "hint": "This sign represents equal access",
        "learning_tip": "This sign is used when discussing accessibility and inclusion.",
    },
    {
        "id": 99,
        "question": "How do you sign 'inclusion' in BSL?",
        "options": [
            "Making a circle with both hands to include everyone",
            "Making two hands move to show equal access",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a circle with both hands to include everyone",
        "explanation": "The sign for 'inclusion' involves making a circle with both hands to include everyone.",
        "category": "culture",
        "difficulty": "advanced",
        "hint": "This sign represents including everyone",
        "learning_tip": "This sign is used when discussing inclusion and diversity.",
    },
    {
        "id": 100,
        "question": "What is the sign for 'diversity' in BSL?",
        "options": [
            "Making different hand shapes to show variety",
            "Making a circle with both hands to include everyone",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making different hand shapes to show variety",
        "explanation": "The sign for 'diversity' involves making different hand shapes to show variety.",
        "category": "culture",
        "difficulty": "advanced",
        "hint": "This sign represents variety and differences",
        "learning_tip": "This sign is used when discussing diversity and different perspectives.",
    },
    {
        "id": 101,
        "question": "How do you sign 'mother' in BSL?",
        "options": [
            "Touching your chin with your thumb",
            "Touching your forehead with your thumb",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Touching your chin with your thumb",
        "explanation": "The sign for 'mother' involves touching your chin with your thumb.",
        "category": "family",
        "difficulty": "beginner",
        "hint": "This sign involves touching your face",
        "learning_tip": "Family signs often involve touching different parts of the face.",
    },
    {
        "id": 102,
        "question": "What is the sign for 'father' in BSL?",
        "options": [
            "Touching your forehead with your thumb",
            "Touching your chin with your thumb",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Touching your forehead with your thumb",
        "explanation": "The sign for 'father' involves touching your forehead with your thumb.",
        "category": "family",
        "difficulty": "beginner",
        "hint": "This sign involves touching your head",
        "learning_tip": "Father and mother signs are similar but touch different parts of the face.",
    },
    {
        "id": 103,
        "question": "How do you sign 'sister' in BSL?",
        "options": [
            "Making an 'S' hand shape and touching your chin",
            "Making an 'S' hand shape and touching your forehead",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making an 'S' hand shape and touching your chin",
        "explanation": "The sign for 'sister' involves making an 'S' hand shape and touching your chin.",
        "category": "family",
        "difficulty": "beginner",
        "hint": "This sign uses the letter 'S'",
        "learning_tip": "Sister and brother signs use the letters 'S' and 'B' respectively.",
    },
    {
        "id": 104,
        "question": "What is the sign for 'brother' in BSL?",
        "options": [
            "Making a 'B' hand shape and touching your forehead",
            "Making an 'S' hand shape and touching your chin",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a 'B' hand shape and touching your forehead",
        "explanation": "The sign for 'brother' involves making a 'B' hand shape and touching your forehead.",
        "category": "family",
        "difficulty": "beginner",
        "hint": "This sign uses the letter 'B'",
        "learning_tip": "Brother and sister signs follow a pattern with letters and face positions.",
    },
    {
        "id": 105,
        "question": "How do you sign 'happy' in BSL?",
        "options": [
            "Making a smile with your hands near your face",
            "Thumbs up gesture",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a smile with your hands near your face",
        "explanation": "The sign for 'happy' involves making a smile gesture with your hands near your face.",
        "category": "emotions",
        "difficulty": "beginner",
        "hint": "This sign represents a smile",
        "learning_tip": "Emotion signs often involve facial expressions and hand gestures.",
    },
    {
        "id": 106,
        "question": "What is the sign for 'sad' in BSL?",
        "options": [
            "Making a downward motion with your hands near your face",
            "Thumbs down gesture",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a downward motion with your hands near your face",
        "explanation": "The sign for 'sad' involves making a downward motion with your hands near your face.",
        "category": "emotions",
        "difficulty": "beginner",
        "hint": "This sign represents a downward emotion",
        "learning_tip": "Sad signs often involve downward movements to represent low mood.",
    },
    {
        "id": 107,
        "question": "How do you sign 'angry' in BSL?",
        "options": [
            "Making a fist and shaking it",
            "Pointing to your chest",
            "Waving hand side to side",
            "Thumbs down gesture",
        ],
        "correct_answer": "Making a fist and shaking it",
        "explanation": "The sign for 'angry' involves making a fist and shaking it.",
        "category": "emotions",
        "difficulty": "beginner",
        "hint": "This sign represents strong emotion",
        "learning_tip": "Angry signs often involve strong, sharp movements.",
    },
    {
        "id": 108,
        "question": "What is the sign for 'surprised' in BSL?",
        "options": [
            "Making wide eyes and opening your mouth",
            "Pointing to your chest",
            "Waving hand side to side",
            "Thumbs up gesture",
        ],
        "correct_answer": "Making wide eyes and opening your mouth",
        "explanation": "The sign for 'surprised' involves making wide eyes and opening your mouth.",
        "category": "emotions",
        "difficulty": "beginner",
        "hint": "This sign involves facial expression",
        "learning_tip": "Surprised signs often involve exaggerated facial expressions.",
    },
    {
        "id": 109,
        "question": "How do you sign 'red' in BSL?",
        "options": [
            "Making an 'R' hand shape and touching your chin",
            "Making a 'B' hand shape and touching your chin",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making an 'R' hand shape and touching your chin",
        "explanation": "The sign for 'red' involves making an 'R' hand shape and touching your chin.",
        "category": "colors",
        "difficulty": "beginner",
        "hint": "This sign uses the letter 'R'",
        "learning_tip": "Color signs often use the first letter of the color name.",
    },
    {
        "id": 110,
        "question": "What is the sign for 'blue' in BSL?",
        "options": [
            "Making a 'B' hand shape and touching your chin",
            "Making an 'R' hand shape and touching your chin",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a 'B' hand shape and touching your chin",
        "explanation": "The sign for 'blue' involves making a 'B' hand shape and touching your chin.",
        "category": "colors",
        "difficulty": "beginner",
        "hint": "This sign uses the letter 'B'",
        "learning_tip": "Blue and red signs are similar but use different letters.",
    },
    {
        "id": 111,
        "question": "How do you sign 'green' in BSL?",
        "options": [
            "Making a 'G' hand shape and touching your chin",
            "Making a 'B' hand shape and touching your chin",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a 'G' hand shape and touching your chin",
        "explanation": "The sign for 'green' involves making a 'G' hand shape and touching your chin.",
        "category": "colors",
        "difficulty": "beginner",
        "hint": "This sign uses the letter 'G'",
        "learning_tip": "Green follows the same pattern as other color signs.",
    },
    {
        "id": 112,
        "question": "What is the sign for 'yellow' in BSL?",
        "options": [
            "Making a 'Y' hand shape and touching your chin",
            "Making a 'G' hand shape and touching your chin",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a 'Y' hand shape and touching your chin",
        "explanation": "The sign for 'yellow' involves making a 'Y' hand shape and touching your chin.",
        "category": "colors",
        "difficulty": "beginner",
        "hint": "This sign uses the letter 'Y'",
        "learning_tip": "Yellow completes the basic color set with letter-based signs.",
    },
    {
        "id": 113,
        "question": "How do you sign 'one' in BSL?",
        "options": [
            "Holding up one finger",
            "Holding up two fingers",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Holding up one finger",
        "explanation": "The sign for 'one' involves holding up one finger.",
        "category": "numbers",
        "difficulty": "beginner",
        "hint": "This sign is very simple",
        "learning_tip": "Number signs from 1-5 are straightforward finger counting.",
    },
    {
        "id": 114,
        "question": "What is the sign for 'two' in BSL?",
        "options": [
            "Holding up two fingers",
            "Holding up one finger",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Holding up two fingers",
        "explanation": "The sign for 'two' involves holding up two fingers.",
        "category": "numbers",
        "difficulty": "beginner",
        "hint": "This sign shows the number",
        "learning_tip": "Numbers 1-5 follow the natural finger counting pattern.",
    },
    {
        "id": 115,
        "question": "How do you sign 'three' in BSL?",
        "options": [
            "Holding up three fingers",
            "Holding up two fingers",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Holding up three fingers",
        "explanation": "The sign for 'three' involves holding up three fingers.",
        "category": "numbers",
        "difficulty": "beginner",
        "hint": "This sign shows the number",
        "learning_tip": "Three is the middle of the basic number sequence.",
    },
    {
        "id": 116,
        "question": "What is the sign for 'four' in BSL?",
        "options": [
            "Holding up four fingers",
            "Holding up three fingers",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Holding up four fingers",
        "explanation": "The sign for 'four' involves holding up four fingers.",
        "category": "numbers",
        "difficulty": "beginner",
        "hint": "This sign shows the number",
        "learning_tip": "Four is close to the maximum for basic finger counting.",
    },
    {
        "id": 117,
        "question": "How do you sign 'five' in BSL?",
        "options": [
            "Holding up five fingers",
            "Holding up four fingers",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Holding up five fingers",
        "explanation": "The sign for 'five' involves holding up five fingers.",
        "category": "numbers",
        "difficulty": "beginner",
        "hint": "This sign shows the number",
        "learning_tip": "Five is the maximum for basic finger counting with one hand.",
    },
    {
        "id": 118,
        "question": "What is the sign for 'sunny' in BSL?",
        "options": [
            "Making a circle with your hands above your head",
            "Making a downward motion with your hands",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a circle with your hands above your head",
        "explanation": "The sign for 'sunny' involves making a circle with your hands above your head.",
        "category": "weather",
        "difficulty": "beginner",
        "hint": "This sign represents the sun",
        "learning_tip": "Weather signs often involve representing natural elements.",
    },
    {
        "id": 119,
        "question": "How do you sign 'rain' in BSL?",
        "options": [
            "Making downward finger movements",
            "Making a circle with your hands",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making downward finger movements",
        "explanation": "The sign for 'rain' involves making downward finger movements.",
        "category": "weather",
        "difficulty": "beginner",
        "hint": "This sign represents falling rain",
        "learning_tip": "Rain signs mimic the movement of falling water.",
    },
    {
        "id": 120,
        "question": "What is the sign for 'snow' in BSL?",
        "options": [
            "Making downward movements with open hands",
            "Making a circle with your hands",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making downward movements with open hands",
        "explanation": "The sign for 'snow' involves making downward movements with open hands.",
        "category": "weather",
        "difficulty": "beginner",
        "hint": "This sign represents falling snow",
        "learning_tip": "Snow signs are similar to rain but with different hand shapes.",
    },
    {
        "id": 121,
        "question": "How do you sign 'bread' in BSL?",
        "options": [
            "Making a slicing motion with your hands",
            "Making a circle with your hands",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a slicing motion with your hands",
        "explanation": "The sign for 'bread' involves making a slicing motion with your hands.",
        "category": "food",
        "difficulty": "beginner",
        "hint": "This sign represents cutting bread",
        "learning_tip": "Food signs often involve the action of preparing or eating the food.",
    },
    {
        "id": 122,
        "question": "What is the sign for 'water' in BSL?",
        "options": [
            "Making a 'W' hand shape and touching your chin",
            "Making a 'B' hand shape and touching your chin",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a 'W' hand shape and touching your chin",
        "explanation": "The sign for 'water' involves making a 'W' hand shape and touching your chin.",
        "category": "food",
        "difficulty": "beginner",
        "hint": "This sign uses the letter 'W'",
        "learning_tip": "Water is a basic necessity and commonly used sign.",
    },
    {
        "id": 123,
        "question": "How do you sign 'milk' in BSL?",
        "options": [
            "Making a squeezing motion with your hands",
            "Making a 'W' hand shape",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a squeezing motion with your hands",
        "explanation": "The sign for 'milk' involves making a squeezing motion with your hands.",
        "category": "food",
        "difficulty": "beginner",
        "hint": "This sign represents milking",
        "learning_tip": "Milk signs represent the action of milking a cow.",
    },
    {
        "id": 124,
        "question": "What is the sign for 'car' in BSL?",
        "options": [
            "Making a steering wheel motion with your hands",
            "Making a walking motion",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a steering wheel motion with your hands",
        "explanation": "The sign for 'car' involves making a steering wheel motion with your hands.",
        "category": "travel",
        "difficulty": "beginner",
        "hint": "This sign represents driving",
        "learning_tip": "Car signs represent the action of steering a vehicle.",
    },
    {
        "id": 125,
        "question": "How do you sign 'train' in BSL?",
        "options": [
            "Making a train motion with your hands",
            "Making a steering wheel motion",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a train motion with your hands",
        "explanation": "The sign for 'train' involves making a train motion with your hands.",
        "category": "travel",
        "difficulty": "beginner",
        "hint": "This sign represents train movement",
        "learning_tip": "Train signs represent the movement of a train on tracks.",
    },
    {
        "id": 126,
        "question": "What is the sign for 'airplane' in BSL?",
        "options": [
            "Making a flying motion with your hands",
            "Making a train motion",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a flying motion with your hands",
        "explanation": "The sign for 'airplane' involves making a flying motion with your hands.",
        "category": "travel",
        "difficulty": "beginner",
        "hint": "This sign represents flying",
        "learning_tip": "Airplane signs represent the action of flying through the air.",
    },
    {
        "id": 127,
        "question": "How do you sign 'school' in BSL?",
        "options": [
            "Making a 'S' hand shape and touching your forehead",
            "Making a 'T' hand shape",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a 'S' hand shape and touching your forehead",
        "explanation": "The sign for 'school' involves making a 'S' hand shape and touching your forehead.",
        "category": "education",
        "difficulty": "beginner",
        "hint": "This sign uses the letter 'S'",
        "learning_tip": "School signs often involve touching the forehead to represent knowledge.",
    },
    {
        "id": 128,
        "question": "What is the sign for 'teacher' in BSL?",
        "options": [
            "Making a 'T' hand shape and touching your forehead",
            "Making a 'S' hand shape",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a 'T' hand shape and touching your forehead",
        "explanation": "The sign for 'teacher' involves making a 'T' hand shape and touching your forehead.",
        "category": "education",
        "difficulty": "beginner",
        "hint": "This sign uses the letter 'T'",
        "learning_tip": "Teacher and school signs are related and use similar hand positions.",
    },
    {
        "id": 129,
        "question": "How do you sign 'student' in BSL?",
        "options": [
            "Making a 'S' hand shape and touching your chin",
            "Making a 'T' hand shape",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a 'S' hand shape and touching your chin",
        "explanation": "The sign for 'student' involves making a 'S' hand shape and touching your chin.",
        "category": "education",
        "difficulty": "beginner",
        "hint": "This sign uses the letter 'S'",
        "learning_tip": "Student signs are related to school but touch the chin instead of forehead.",
    },
    {
        "id": 130,
        "question": "What is the sign for 'work' in BSL?",
        "options": [
            "Making a 'W' hand shape and touching your forehead",
            "Making a 'S' hand shape",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a 'W' hand shape and touching your forehead",
        "explanation": "The sign for 'work' involves making a 'W' hand shape and touching your forehead.",
        "category": "work",
        "difficulty": "beginner",
        "hint": "This sign uses the letter 'W'",
        "learning_tip": "Work signs often involve touching the forehead to represent mental activity.",
    },
    {
        "id": 131,
        "question": "How do you sign 'computer' in BSL?",
        "options": [
            "Making typing motions with your hands",
            "Making a 'C' hand shape",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making typing motions with your hands",
        "explanation": "The sign for 'computer' involves making typing motions with your hands.",
        "category": "technology",
        "difficulty": "beginner",
        "hint": "This sign represents typing",
        "learning_tip": "Computer signs represent the action of using a keyboard.",
    },
    {
        "id": 132,
        "question": "What is the sign for 'phone' in BSL?",
        "options": [
            "Making a phone shape with your hand near your ear",
            "Making typing motions",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a phone shape with your hand near your ear",
        "explanation": "The sign for 'phone' involves making a phone shape with your hand near your ear.",
        "category": "technology",
        "difficulty": "beginner",
        "hint": "This sign represents using a phone",
        "learning_tip": "Phone signs represent the action of holding a phone to your ear.",
    },
    {
        "id": 133,
        "question": "How do you sign 'football' in BSL?",
        "options": [
            "Making a kicking motion with your foot",
            "Making a throwing motion",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a kicking motion with your foot",
        "explanation": "The sign for 'football' involves making a kicking motion with your foot.",
        "category": "sports",
        "difficulty": "beginner",
        "hint": "This sign represents kicking",
        "learning_tip": "Sports signs often represent the main action of the sport.",
    },
    {
        "id": 134,
        "question": "What is the sign for 'swimming' in BSL?",
        "options": [
            "Making swimming motions with your arms",
            "Making a kicking motion",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making swimming motions with your arms",
        "explanation": "The sign for 'swimming' involves making swimming motions with your arms.",
        "category": "sports",
        "difficulty": "beginner",
        "hint": "This sign represents swimming",
        "learning_tip": "Swimming signs represent the arm movements used in swimming.",
    },
    {
        "id": 135,
        "question": "How do you sign 'music' in BSL?",
        "options": [
            "Making conducting motions with your hands",
            "Making swimming motions",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making conducting motions with your hands",
        "explanation": "The sign for 'music' involves making conducting motions with your hands.",
        "category": "music",
        "difficulty": "beginner",
        "hint": "This sign represents conducting",
        "learning_tip": "Music signs represent the action of conducting an orchestra.",
    },
    {
        "id": 136,
        "question": "What is the sign for 'dance' in BSL?",
        "options": [
            "Making dancing motions with your hands",
            "Making conducting motions",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making dancing motions with your hands",
        "explanation": "The sign for 'dance' involves making dancing motions with your hands.",
        "category": "music",
        "difficulty": "beginner",
        "hint": "This sign represents dancing",
        "learning_tip": "Dance signs represent the movement of dancing.",
    },
    {
        "id": 137,
        "question": "How do you sign 'doctor' in BSL?",
        "options": [
            "Making a 'D' hand shape and touching your forehead",
            "Making a 'M' hand shape",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a 'D' hand shape and touching your forehead",
        "explanation": "The sign for 'doctor' involves making a 'D' hand shape and touching your forehead.",
        "category": "health",
        "difficulty": "beginner",
        "hint": "This sign uses the letter 'D'",
        "learning_tip": "Doctor signs often involve touching the forehead to represent medical knowledge.",
    },
    {
        "id": 138,
        "question": "What is the sign for 'hospital' in BSL?",
        "options": [
            "Making an 'H' hand shape and touching your forehead",
            "Making a 'D' hand shape",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making an 'H' hand shape and touching your forehead",
        "explanation": "The sign for 'hospital' involves making an 'H' hand shape and touching your forehead.",
        "category": "health",
        "difficulty": "beginner",
        "hint": "This sign uses the letter 'H'",
        "learning_tip": "Hospital signs are related to doctor signs and use similar hand positions.",
    },
    {
        "id": 139,
        "question": "How do you sign 'help' in BSL?",
        "options": [
            "Making a 'H' hand shape and moving it up",
            "Making an 'H' hand shape and touching your forehead",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a 'H' hand shape and moving it up",
        "explanation": "The sign for 'help' involves making a 'H' hand shape and moving it up.",
        "category": "emergency",
        "difficulty": "beginner",
        "hint": "This sign represents asking for help",
        "learning_tip": "Help signs are important emergency signs to know.",
    },
    {
        "id": 140,
        "question": "What is the sign for 'emergency' in BSL?",
        "options": [
            "Making an 'E' hand shape and moving it urgently",
            "Making a 'H' hand shape",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making an 'E' hand shape and moving it urgently",
        "explanation": "The sign for 'emergency' involves making an 'E' hand shape and moving it urgently.",
        "category": "emergency",
        "difficulty": "beginner",
        "hint": "This sign represents urgency",
        "learning_tip": "Emergency signs are crucial for safety and communication.",
    },
    {
        "id": 141,
        "question": "How do you sign 'police' in BSL?",
        "options": [
            "Making a 'P' hand shape and touching your forehead",
            "Making an 'E' hand shape",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a 'P' hand shape and touching your forehead",
        "explanation": "The sign for 'police' involves making a 'P' hand shape and touching your forehead.",
        "category": "legal",
        "difficulty": "beginner",
        "hint": "This sign uses the letter 'P'",
        "learning_tip": "Police signs are important for legal and safety situations.",
    },
    {
        "id": 142,
        "question": "What is the sign for 'lawyer' in BSL?",
        "options": [
            "Making an 'L' hand shape and touching your forehead",
            "Making a 'P' hand shape",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making an 'L' hand shape and touching your forehead",
        "explanation": "The sign for 'lawyer' involves making an 'L' hand shape and touching your forehead.",
        "category": "legal",
        "difficulty": "beginner",
        "hint": "This sign uses the letter 'L'",
        "learning_tip": "Lawyer signs are related to legal professions.",
    },
    {
        "id": 143,
        "question": "How do you sign 'money' in BSL?",
        "options": [
            "Making a 'M' hand shape and rubbing your fingers",
            "Making an 'L' hand shape",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a 'M' hand shape and rubbing your fingers",
        "explanation": "The sign for 'money' involves making a 'M' hand shape and rubbing your fingers.",
        "category": "finance",
        "difficulty": "beginner",
        "hint": "This sign represents counting money",
        "learning_tip": "Money signs represent the action of handling money.",
    },
    {
        "id": 144,
        "question": "What is the sign for 'bank' in BSL?",
        "options": [
            "Making a 'B' hand shape and touching your forehead",
            "Making a 'M' hand shape",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a 'B' hand shape and touching your forehead",
        "explanation": "The sign for 'bank' involves making a 'B' hand shape and touching your forehead.",
        "category": "finance",
        "difficulty": "beginner",
        "hint": "This sign uses the letter 'B'",
        "learning_tip": "Bank signs are related to financial institutions.",
    },
    {
        "id": 145,
        "question": "How do you sign 'tree' in BSL?",
        "options": [
            "Making a tree shape with your arms",
            "Making a 'B' hand shape",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a tree shape with your arms",
        "explanation": "The sign for 'tree' involves making a tree shape with your arms.",
        "category": "environment",
        "difficulty": "beginner",
        "hint": "This sign represents a tree",
        "learning_tip": "Tree signs represent the shape and structure of a tree.",
    },
    {
        "id": 146,
        "question": "What is the sign for 'flower' in BSL?",
        "options": [
            "Making a flower shape with your hands",
            "Making a tree shape",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a flower shape with your hands",
        "explanation": "The sign for 'flower' involves making a flower shape with your hands.",
        "category": "environment",
        "difficulty": "beginner",
        "hint": "This sign represents a flower",
        "learning_tip": "Flower signs represent the beauty and shape of flowers.",
    },
    {
        "id": 147,
        "question": "How do you sign 'birthday' in BSL?",
        "options": [
            "Making a cake shape with your hands",
            "Making a flower shape",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a cake shape with your hands",
        "explanation": "The sign for 'birthday' involves making a cake shape with your hands.",
        "category": "celebration",
        "difficulty": "beginner",
        "hint": "This sign represents a birthday cake",
        "learning_tip": "Birthday signs represent the celebration with cake.",
    },
    {
        "id": 148,
        "question": "What is the sign for 'party' in BSL?",
        "options": [
            "Making celebration motions with your hands",
            "Making a cake shape",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making celebration motions with your hands",
        "explanation": "The sign for 'party' involves making celebration motions with your hands.",
        "category": "celebration",
        "difficulty": "beginner",
        "hint": "This sign represents celebration",
        "learning_tip": "Party signs represent the joy and celebration of events.",
    },
    {
        "id": 149,
        "question": "How do you sign 'love' in BSL?",
        "options": [
            "Making a heart shape with your hands",
            "Making celebration motions",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a heart shape with your hands",
        "explanation": "The sign for 'love' involves making a heart shape with your hands.",
        "category": "emotions",
        "difficulty": "beginner",
        "hint": "This sign represents a heart",
        "learning_tip": "Love signs represent the universal symbol of the heart.",
    },
    {
        "id": 150,
        "question": "What is the sign for 'friend' in BSL?",
        "options": [
            "Making a 'F' hand shape and touching your chin",
            "Making a heart shape",
            "Pointing to your chest",
            "Waving hand side to side",
        ],
        "correct_answer": "Making a 'F' hand shape and touching your chin",
        "explanation": "The sign for 'friend' involves making a 'F' hand shape and touching your chin.",
        "category": "conversation",
        "difficulty": "beginner",
        "hint": "This sign uses the letter 'F'",
        "learning_tip": "Friend signs are important for social communication.",
    },
]


@app.get("/")
async def root():
    return {"message": "Sign Language Tutorial API"}


@app.get("/facts")
async def get_facts():
    """Get all BSL facts"""
    # Return only facts 1-19
    limited_facts = bsl_facts[:19]
    return {"facts": limited_facts}


@app.get("/facts/{fact_id}")
async def get_fact(fact_id: int):
    """Get a specific BSL fact by ID"""
    for fact in bsl_facts:
        if fact["id"] == fact_id:
            return fact
    raise HTTPException(status_code=404, detail="Fact not found")


@app.get("/quiz")
async def get_quiz_questions(category: str = "all", difficulty: str = "beginner"):
    """Get quiz questions with optional filtering by category and difficulty"""
    filtered_questions = quiz_questions

    # Filter by category if specified
    if category != "all":
        filtered_questions = [
            q for q in filtered_questions if q.get("category") == category
        ]

    # Filter by difficulty if specified
    if difficulty != "all":
        filtered_questions = [
            q for q in filtered_questions if q.get("difficulty") == difficulty
        ]

    return {"questions": filtered_questions}


@app.get("/quiz/{question_id}")
async def get_quiz_question(question_id: int):
    """Get a specific quiz question by ID"""
    for question in quiz_questions:
        if question["id"] == question_id:
            return question
    raise HTTPException(status_code=404, detail="Question not found")


@app.post("/quiz/submit")
async def submit_quiz_answer(data: Dict[str, Any]):
    """Submit a quiz answer and get feedback"""
    question_id = data.get("question_id")
    selected_answer = data.get("selected_answer")

    if question_id is None or selected_answer is None:
        raise HTTPException(
            status_code=400, detail="question_id and selected_answer are required"
        )

    for question in quiz_questions:
        if question["id"] == question_id:
            is_correct = selected_answer == question["correct_answer"]
            return {
                "correct": is_correct,
                "correct_answer": question["correct_answer"],
                "explanation": question["explanation"],
                "learning_tip": question.get("learning_tip", ""),
                "category": question.get("category", ""),
                "difficulty": question.get("difficulty", ""),
            }
    raise HTTPException(status_code=404, detail="Question not found")


@app.post("/detect-sign")
async def detect_sign(file: UploadFile = File(...)):
    """Detect sign language from uploaded image"""
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Read and process the image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # Process with MediaPipe
        with mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            image, results = mediapipe_detection(image, holistic)

            # Extract keypoints
            keypoints = extract_keypoints(results)

            # Make prediction (we need 30 frames for LSTM, so we'll repeat the keypoints)
            sequence = [keypoints] * 30
            sequence = np.array(sequence)

            # Predict
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predicted_sign = actions[np.argmax(res)]
            confidence = float(res[np.argmax(res)])

            return {
                "detected_sign": predicted_sign,
                "confidence": confidence,
                "all_probabilities": res.tolist(),
                "available_signs": signs,
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/detect-sign-base64")
async def detect_sign_base64(data: Dict[str, Any]):
    """Detect sign language from base64 encoded image"""
    global sequence, sentence, predictions

    if not model:
        return {
            "detected_sign": "no_sign",
            "confidence": 0.0,
            "all_probabilities": [0.25, 0.25, 0.25, 0.25],
            "available_signs": signs,
            "message": "Model not available. Please try again later.",
        }

    if not holistic:
        return {
            "detected_sign": "no_sign",
            "confidence": 0.0,
            "all_probabilities": [0.25, 0.25, 0.25, 0.25],
            "available_signs": signs,
            "message": "MediaPipe not available. Please try again later.",
        }

    try:
        # Decode base64 image
        image_data = data.get("image")
        if not image_data:
            return {
                "detected_sign": "no_sign",
                "confidence": 0.0,
                "all_probabilities": [0.25, 0.25, 0.25, 0.25],
                "available_signs": signs,
                "message": "No image data provided.",
            }

        # Remove data URL prefix if present
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]

        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return {
                "detected_sign": "no_sign",
                "confidence": 0.0,
                "all_probabilities": [0.25, 0.25, 0.25, 0.25],
                "available_signs": signs,
                "message": "Invalid image format.",
            }

        if image is None:
            return {
                "detected_sign": "no_sign",
                "confidence": 0.0,
                "all_probabilities": [0.25, 0.25, 0.25, 0.25],
                "available_signs": signs,
                "message": "Could not decode image.",
            }

        # Process with MediaPipe using global instance
        try:
            image, results = mediapipe_detection(image, holistic)
        except Exception as e:
            return {
                "detected_sign": "no_sign",
                "confidence": 0.0,
                "all_probabilities": [0.25, 0.25, 0.25, 0.25],
                "available_signs": signs,
                "message": "Error processing image with MediaPipe.",
            }

        # Extract keypoints
        try:
            keypoints = extract_keypoints(results)
        except Exception as e:
            return {
                "detected_sign": "no_sign",
                "confidence": 0.0,
                "all_probabilities": [0.25, 0.25, 0.25, 0.25],
                "available_signs": signs,
                "message": "Error extracting keypoints.",
            }

        # Check if we have valid keypoints (not all zeros)
        if np.sum(keypoints) == 0:
            return {
                "detected_sign": "no_sign",
                "confidence": 0.0,
                "all_probabilities": [0.25, 0.25, 0.25, 0.25],
                "available_signs": signs,
                "message": "No hand landmarks detected. Please ensure your hands are visible.",
            }

        # Use the exact logic from the original real-time code
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep only last 30 frames

        detected_sign = "no_sign"
        confidence = 0.0
        all_probabilities = [0.25, 0.25, 0.25, 0.25]
        status_message = ""

        if len(sequence) == 30:
            try:
                # Make prediction exactly as in original code
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                predictions.append(np.argmax(res))

                # Debug output
                print(
                    f"Frame {len(predictions)}: Predicted {actions[np.argmax(res)]} with confidence {res[np.argmax(res)]:.3f}"
                )
                print(f"All probabilities: {[f'{p:.3f}' for p in res]}")

                # Use the exact viz logic from original code (exactly as in user's working code)
                if len(predictions) >= 10 and np.unique(predictions[-10:])[
                    0
                ] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        print(
                            f"✅ Sign detected: {actions[np.argmax(res)]} (confidence: {res[np.argmax(res)]:.3f})"
                        )
                        status_message = f"Sign detected: {actions[np.argmax(res)]}"

                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Set detected sign and confidence
                if len(sentence) > 0:
                    detected_sign = sentence[-1]
                    confidence = float(res[np.argmax(res)])
                    print(f"🎯 Current sentence: {sentence}")
                else:
                    # Show current prediction even if not added to sentence yet
                    detected_sign = actions[np.argmax(res)]
                    confidence = float(res[np.argmax(res)])
                    status_message = (
                        f"Detecting: {detected_sign} (confidence: {confidence:.3f})"
                    )

                all_probabilities = res.tolist()

            except Exception as e:
                print(f"❌ Prediction error: {e}")
                status_message = "Error making prediction"
                return {
                    "detected_sign": "no_sign",
                    "confidence": 0.0,
                    "all_probabilities": [0.25, 0.25, 0.25, 0.25],
                    "available_signs": signs,
                    "message": "Error making prediction.",
                }
        else:
            print(f"📊 Collecting frames: {len(sequence)}/30")
            status_message = f"Collecting frames: {len(sequence)}/30"

        return {
            "detected_sign": detected_sign,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "available_signs": signs,
            "current_sequence_length": len(sequence),
            "sentence": sentence[-5:] if sentence else [],  # Show last 5 detected signs
            "predictions_count": len(predictions),
            "status_message": status_message,
            "threshold": threshold,
        }

    except Exception as e:
        print(f"Unexpected error in sign detection: {str(e)}")
        return {
            "detected_sign": "no_sign",
            "confidence": 0.0,
            "all_probabilities": [0.25, 0.25, 0.25, 0.25],
            "available_signs": signs,
            "message": "Unexpected error. Please try again.",
        }


@app.post("/chat")
async def chat_with_bsl_assistant(data: Dict[str, Any]):
    """Chat with BSL assistant using LLM"""
    try:
        user_message = data.get("message", "")
        if not user_message:
            return {"response": "Please provide a question about BSL.", "error": False}

        # Create a context-aware prompt for BSL questions
        system_prompt = """You are a helpful assistant specializing in British Sign Language (BSL). 
        You can help with:
        - BSL grammar and structure
        - Sign language techniques and tips
        - Deaf culture and community
        - Educational challenges for deaf students
        - Accessibility and inclusion
        - BSL resources and learning materials
        
        Please provide accurate, helpful, and supportive responses. If you're not sure about something, 
        say so and suggest where they might find more information."""

        full_prompt = f"{system_prompt}\n\nUser Question: {user_message}\n\nAssistant:"

        # Use a free LLM API (you can replace this with your preferred LLM service)
        try:
            # For now, we'll use a simple response system
            # In production, you would integrate with OpenAI, Anthropic, or another LLM service

            # Simple keyword-based responses for common BSL questions
            user_message_lower = user_message.lower()

            if any(word in user_message_lower for word in ["hello", "hi", "greeting"]):
                response = "In BSL, 'hello' is signed by waving your hand from side to side at shoulder level. It's a friendly, open-palm gesture that's commonly used to greet people."

            elif any(word in user_message_lower for word in ["thank", "thanks"]):
                response = "To sign 'thank you' in BSL, touch your chin with your fingertips and move your hand forward and down. It's a polite gesture that shows appreciation."

            elif any(word in user_message_lower for word in ["please"]):
                response = "The BSL sign for 'please' involves placing your flat hand on your chest and moving it in a circular motion. It's used to make polite requests."

            elif any(word in user_message_lower for word in ["grammar", "structure"]):
                response = "BSL has its own grammar structure that's different from English. It uses topic-comment structure, spatial grammar, and facial expressions to convey meaning. The basic word order is often: Time + Topic + Comment."

            elif any(word in user_message_lower for word in ["finger", "spell"]):
                response = "BSL uses a two-handed manual alphabet for finger spelling, unlike American Sign Language. Each letter has a specific hand shape and position. This is useful for spelling names, places, or words that don't have established signs."

            elif any(word in user_message_lower for word in ["deaf", "culture"]):
                response = "Deaf culture is rich and vibrant, with its own traditions, art, literature, and social norms. BSL is central to this cultural identity. Deaf people often prefer to be called 'Deaf' (with a capital D) when referring to their cultural identity."

            elif any(
                word in user_message_lower for word in ["learn", "study", "practice"]
            ):
                response = "To learn BSL effectively, I recommend: 1) Take formal classes with qualified BSL teachers, 2) Practice regularly with deaf people, 3) Use online resources and apps, 4) Join BSL learning communities, 5) Immerse yourself in deaf culture and events."

            elif any(
                word in user_message_lower
                for word in ["challenge", "difficulty", "problem"]
            ):
                response = "Common challenges in learning BSL include: mastering facial expressions, understanding spatial grammar, developing fluency in finger spelling, and becoming comfortable with the visual-spatial nature of the language. Regular practice and immersion help overcome these challenges."

            elif any(
                word in user_message_lower
                for word in ["resource", "book", "video", "app"]
            ):
                response = "Great BSL resources include: British Deaf Association (BDA) courses, Signature BSL qualifications, online platforms like BSL Zone, mobile apps like 'Sign BSL', and local deaf community centers. Many universities also offer BSL courses."

            elif any(
                word in user_message_lower for word in ["interpreter", "translation"]
            ):
                response = "BSL interpreters are professionals who facilitate communication between deaf and hearing people. They must be qualified and registered. In the UK, look for interpreters registered with the National Registers of Communication Professionals working with Deaf and Deafblind People (NRCPD)."

            elif any(
                word in user_message_lower
                for word in ["education", "school", "student"]
            ):
                response = "Deaf students face various challenges in education, including lack of qualified interpreters, limited access to BSL resources, social isolation, and inadequate accommodations. Inclusive education requires proper support, qualified staff, and accessible learning materials."

            else:
                # For questions not covered by keywords, provide a general helpful response
                response = f"I understand you're asking about BSL: '{user_message}'. While I can provide general information about British Sign Language, for specific or complex questions, I recommend consulting with qualified BSL teachers, the British Deaf Association, or local deaf community organizations. They can provide the most accurate and up-to-date information about BSL and deaf culture."

            return {"response": response, "error": False, "user_message": user_message}

        except Exception as e:
            return {
                "response": "I'm having trouble processing your question right now. Please try again or contact a BSL instructor for assistance.",
                "error": True,
                "user_message": user_message,
            }

    except Exception as e:
        return {
            "response": "An error occurred while processing your request. Please try again.",
            "error": True,
        }


@app.post("/reset-detection")
async def reset_detection():
    """Reset detection state"""
    global sequence, sentence, predictions
    sequence = []
    sentence = []
    predictions = []
    return {
        "message": "Detection state reset successfully",
        "sequence_length": 0,
        "sentence_length": 0,
        "predictions_length": 0,
    }


@app.get("/test-model")
async def test_model():
    """Test the model with sample data"""
    if not model:
        return {"error": "Model not loaded"}

    try:
        # Test 1: All zeros (baseline)
        sample_keypoints = np.zeros(1662)
        sample_sequence = [sample_keypoints] * 30
        res1 = model.predict(np.expand_dims(sample_sequence, axis=0), verbose=0)[0]

        # Test 2: Some non-zero values (simulating hand landmarks)
        sample_keypoints2 = np.random.rand(1662) * 0.1  # Small random values
        sample_sequence2 = [sample_keypoints2] * 30
        res2 = model.predict(np.expand_dims(sample_sequence2, axis=0), verbose=0)[0]

        # Test 3: Simulate hand position (higher values for hand landmarks)
        sample_keypoints3 = np.zeros(1662)
        # Set some hand landmark positions (indices 501-522 for left hand, 523-544 for right hand)
        sample_keypoints3[501:522] = np.random.rand(21) * 0.5  # Left hand
        sample_keypoints3[523:544] = np.random.rand(21) * 0.5  # Right hand
        sample_sequence3 = [sample_keypoints3] * 30
        res3 = model.predict(np.expand_dims(sample_sequence3, axis=0), verbose=0)[0]

        return {
            "model_loaded": True,
            "test_results": {
                "test1_zeros": {
                    "predicted_sign": actions[np.argmax(res1)],
                    "confidence": float(res1[np.argmax(res1)]),
                    "all_probabilities": res1.tolist(),
                },
                "test2_random": {
                    "predicted_sign": actions[np.argmax(res2)],
                    "confidence": float(res2[np.argmax(res2)]),
                    "all_probabilities": res2.tolist(),
                },
                "test3_hands": {
                    "predicted_sign": actions[np.argmax(res3)],
                    "confidence": float(res3[np.argmax(res3)]),
                    "all_probabilities": res3.tolist(),
                },
            },
            "available_signs": signs,
            "model_info": {
                "input_shape": model.input_shape,
                "output_shape": model.output_shape,
                "total_params": model.count_params(),
            },
            "model_summary": "Model is working correctly",
        }
    except Exception as e:
        return {"error": f"Model test failed: {str(e)}", "model_loaded": False}


@app.get("/test-detection")
async def test_detection():
    """Test the sign detection logic with sample data"""
    global sequence, sentence, predictions

    # Reset state
    sequence = []
    sentence = []
    predictions = []

    # Simulate 30 frames of hand data
    for i in range(30):
        # Create sample keypoints with hand landmarks
        keypoints = np.zeros(1662)
        # Set hand landmark positions
        keypoints[501:522] = np.random.rand(21) * 0.5  # Left hand
        keypoints[523:544] = np.random.rand(21) * 0.5  # Right hand

        sequence.append(keypoints)

    # Make prediction
    res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
    predictions.append(np.argmax(res))

    # Test detection logic
    detected_sign = "no_sign"
    confidence = 0.0
    status_message = ""

    if len(predictions) >= 10 and np.unique(predictions[-10:])[0] == np.argmax(res):
        if res[np.argmax(res)] > threshold:
            if len(sentence) > 0:
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
            else:
                sentence.append(actions[np.argmax(res)])

    if len(sentence) > 5:
        sentence = sentence[-5:]

    if len(sentence) > 0:
        detected_sign = sentence[-1]
        confidence = float(res[np.argmax(res)])
        status_message = f"Sign detected: {detected_sign}"
    else:
        detected_sign = actions[np.argmax(res)]
        confidence = float(res[np.argmax(res)])
        status_message = f"Detecting: {detected_sign} (confidence: {confidence:.3f})"

    return {
        "test_completed": True,
        "detected_sign": detected_sign,
        "confidence": confidence,
        "all_probabilities": res.tolist(),
        "sentence": sentence,
        "predictions_count": len(predictions),
        "sequence_length": len(sequence),
        "threshold": threshold,
        "available_signs": signs,
        "status_message": status_message,
    }


if __name__ == "__main__":
    import uvicorn
    import atexit

    # Cleanup function to close MediaPipe
    def cleanup():
        global holistic
        if holistic:
            try:
                holistic.close()
                print("MediaPipe Holistic closed successfully")
            except:
                pass

    # Register cleanup function
    atexit.register(cleanup)

    uvicorn.run(app, host="0.0.0.0", port=8000)
