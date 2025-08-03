from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

# BSL Facts Data (only facts 1-19)
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
]


@app.get("/")
async def root():
    return {"message": "Sign Language Tutorial API is running!"}


@app.get("/facts")
async def get_facts():
    """Get BSL facts (1-19)"""
    return {"facts": bsl_facts}


@app.get("/facts/{fact_id}")
async def get_fact(fact_id: int):
    """Get a specific fact by ID"""
    for fact in bsl_facts:
        if fact["id"] == fact_id:
            return fact
    return {"error": "Fact not found"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
