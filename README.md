# SignLango - BSL Learning Application

A comprehensive British Sign Language (BSL) learning application with AI-powered sign detection, interactive quizzes, and educational content.

## Features

- **AI Detection**: Record and analyze sign language videos using AI
- **Interactive Quizzes**: Test your BSL knowledge with various difficulty levels
- **Educational Content**: Learn BSL facts and common questions
- **Chat Assistant**: Get help with BSL learning through an AI-powered chat
- **Modern UI**: Beautiful, responsive interface

## Prerequisites

Before running the application, make sure you have the following installed:

- **Python 3.8+** (for backend)
- **Node.js 14+** (for frontend)
- **npm** (comes with Node.js)
- **Git** (for cloning the repository)

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd signlango-msc-project
```

### 2. Backend Setup

Navigate to the backend directory and set up the Python environment:

```bash
cd signlango/quiz_app/backend
```

#### Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### Set Up Environment Variables

Create a `.env` file in the backend directory:

```bash
touch .env
```

Add your Google API key to the `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

**Note**: You can get a free Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

### 3. Frontend Setup

Navigate to the frontend directory:

```bash
cd ../../frontend
```

#### Install Node.js Dependencies

```bash
npm install
```

## Running the Application

### Step 1: Start the Backend Server

Open a terminal and navigate to the backend directory:

```bash
cd signlango/quiz_app/backend
python main.py
```

You should see output similar to:

```
✅ Gemini AI with LangChain initialized successfully
INFO:     Started server process [XXXXX]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

The backend server will be running on `http://localhost:8000`.

### Step 2: Start the Frontend Server

Open a new terminal window and navigate to the frontend directory:

```bash
cd signlango/quiz_app/frontend
npm start
```

You should see output similar to:

```
Compiled successfully!
You can now view sign-language-tutorial-frontend in the browser.
  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000
```

The frontend application will be running on `http://localhost:3000`.

### Step 3: Access the Application

Open your web browser and navigate to:

```
http://localhost:3000
```

## Using the Application

### AI Detection Feature

1. Click on "AI Detection" in the navigation menu
2. Select one of the four signs: "hi", "please", "excuse me", or "okay"
3. Click "Start Camera" to enable your webcam
4. Click "Start Recording" and perform the selected sign for 3-4 seconds
5. Click "Stop Recording" when done
6. Watch your recorded video
7. Click "Analyze with AI" to get feedback on your sign

### Quiz Feature

1. Click on "Quiz" in the navigation menu
2. Choose your preferred category and difficulty level
3. Answer the multiple-choice questions
4. Review your results and learn from explanations

### Chat Assistant

1. Click on "Chat" in the navigation menu
2. Type your BSL-related questions
3. Get instant responses about signs, grammar, and Deaf culture

### Educational Content

- **Facts**: Learn interesting facts about BSL and Deaf culture
- **Common Questions**: Find answers to frequently asked questions

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

If you see "address already in use" errors:

```bash
# Find processes using port 8000
lsof -i :8000

# Kill the process
kill <process_id>
```

#### 2. API Key Issues

If you see Gemini API quota errors:

- Check that your `GOOGLE_API_KEY` is correctly set in the `.env` file
- Verify your API key is valid and has sufficient quota
- The application will still work with fallback responses even without the API

#### 3. Camera Access Issues

If the camera doesn't work in AI Detection:

- Make sure your browser has permission to access the camera
- Try refreshing the page
- Check that no other applications are using the camera

#### 4. Dependencies Issues

If you encounter dependency errors:

**Backend:**

```bash
cd signlango/quiz_app/backend
pip install --upgrade pip
pip install -r requirements.txt
```

**Frontend:**

```bash
cd signlango/quiz_app/frontend
rm -rf node_modules package-lock.json
npm install
```

### File Structure

```
signlango-msc-project/
├── signlango/
│   └── quiz_app/
│       ├── backend/
│       │   ├── main.py
│       │   ├── requirements.txt
│       │   └── .env
│       └── frontend/
│           ├── src/
│           ├── public/
│           └── package.json
├── README.md
└── AI_DETECTION_SETUP.md
```

## Development

### Backend Development

The backend is built with:

- **FastAPI**: Modern Python web framework
- **LangChain**: For conversational AI
- **MediaPipe**: For video processing

### Frontend Development

The frontend is built with:

- **React**: JavaScript library for building user interfaces
- **React Router**: For navigation
- **Axios**: For HTTP requests
- **MediaRecorder API**: For video recording

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the logs in your terminal
3. Create an issue in the repository

## Acknowledgments

- The Deaf community for inspiration and guidance
- All contributors to this project
