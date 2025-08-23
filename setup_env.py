#!/usr/bin/env python3
"""
Setup script for SignLango AI Chat Backend
This script helps you configure the Google Gemini API key for the chat functionality.
"""

import os
import sys

def create_env_file():
    """Create a .env file with the Google Gemini API key"""
    
    print("🤖 SignLango AI Chat Setup")
    print("=" * 40)
    print()
    print("To enable AI chat functionality, you need a Google Gemini API key.")
    print("Get your free API key from: https://makersuite.google.com/app/apikey")
    print()
    
    # Check if .env file already exists
    if os.path.exists('.env'):
        print("⚠️  .env file already exists!")
        overwrite = input("Do you want to overwrite it? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("Setup cancelled.")
            return
    
    # Get API key from user
    api_key = input("Enter your Google Gemini API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Setup cancelled.")
        return
    
    # Create .env file
    env_content = f"""# Google Gemini API Configuration
# Get your API key from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY={api_key}

# Optional: Database configuration (for production)
# DATABASE_URL=sqlite:///./chat_history.db

# Optional: Server configuration
# HOST=0.0.0.0
# PORT=8000
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ .env file created successfully!")
        print("🔑 Your API key has been saved.")
        print()
        print("Next steps:")
        print("1. Start the backend: python signlango/quiz_app/backend/chat_backend.py")
        print("2. Test the chat functionality at: http://localhost:8000/docs")
        print()
        print("Note: Keep your API key secure and never commit it to version control.")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

def test_api_key():
    """Test if the API key is working"""
    print("🧪 Testing API Key...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ No API key found in .env file")
            return False
        
        # Test the API key
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=100
        )
        
        response = llm.invoke("Hello! Can you confirm you're working?")
        print("✅ API key is working!")
        print(f"🤖 AI Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_api_key()
    else:
        create_env_file() 