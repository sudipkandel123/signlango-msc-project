# Sign Language Tutorial Web App

A comprehensive web application for learning British Sign Language (BSL) with interactive features including facts, quizzes, and real-time sign detection using AI.

## Features

- **📚 BSL Facts**: Learn interesting facts about British Sign Language
- **🧠 Interactive Quiz**: Test your knowledge with multiple-choice questions
- **📹 Real-time Sign Detection**: Use your webcam to practice signs and get instant feedback
- **🎨 Modern UI**: Beautiful, responsive design with glassmorphism effects
- **🤖 AI-Powered**: Uses TensorFlow and MediaPipe for accurate sign recognition

## Supported Signs

The app currently supports detection of these BSL signs:

- 👋 Hi (waving hand)
- 🙏 Please (circular palm motion)
- 🤚 Excuse Me (shoulder tap)
- 👍 Okay (thumbs up)

## Project Structure

```
quiz_app/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main API server
│   ├── run.py              # Run script
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend
│   ├── public/             # Static files
│   ├── src/                # React source code
│   │   ├── components/     # React components
│   │   ├── App.js          # Main app component
│   │   └── index.js        # Entry point
│   └── package.json        # Node.js dependencies
└── README.md               # This file
```

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- Webcam (for sign detection feature)

## Installation & Setup

### Backend Setup

1. **Navigate to backend directory:**

   ```bash
   cd backend
   ```
2. **Create virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
4. **Add your trained model files:**

   - Place your `model.h5` and `model.weights.h5` files in the `backend/` directory
   - These should be the trained LSTM model from your sign detection script
5. **Run the backend server:**

   ```bash
   python run.py
   ```

   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory:**

   ```bash
   cd frontend
   ```
2. **Install dependencies:**

   ```bash
   npm install
   ```
3. **Start the development server:**

   ```bash
   npm start
   ```

   The app will open at `http://localhost:3000`

## Usage

### Learning BSL Facts

- Navigate to the "BSL Facts" section
- Read through interesting facts about British Sign Language
- Learn about its history, recognition, and importance

### Taking the Quiz

- Go to the "Quiz" section
- Answer multiple-choice questions about BSL signs
- Get instant feedback and explanations
- Track your score and progress

### Real-time Sign Detection

- Visit the "Sign Detection" section
- Click "Start Camera" to enable your webcam
- Position yourself so your hands and face are visible
- Perform one of the supported signs
- Wait for AI detection and feedback
- Click "Stop Camera" when finished

## API Endpoints

### Facts

- `GET /facts` - Get all BSL facts
- `GET /facts/{fact_id}` - Get specific fact by ID

### Quiz

- `GET /quiz` - Get all quiz questions
- `GET /quiz/{question_id}` - Get specific question by ID
- `POST /quiz/submit` - Submit quiz answer

### Sign Detection

- `POST /detect-sign` - Detect sign from uploaded image
- `POST /detect-sign-base64` - Detect sign from base64 encoded image

## Technical Details

### Backend Technologies

- **FastAPI**: Modern Python web framework
- **TensorFlow**: Machine learning framework for sign detection
- **MediaPipe**: Hand and pose detection
- **OpenCV**: Computer vision processing
- **Uvicorn**: ASGI server

### Frontend Technologies

- **React**: JavaScript library for building user interfaces
- **React Router**: Client-side routing
- **Axios**: HTTP client for API communication
- **CSS3**: Modern styling with glassmorphism effects

### AI Model

The sign detection uses:

- LSTM neural network for sequence classification
- MediaPipe Holistic for pose and hand landmark detection
- Real-time video processing with confidence scoring

## Troubleshooting

### Camera Issues

- Ensure camera permissions are granted in your browser
- Try refreshing the page if camera doesn't start
- Check that no other applications are using the camera

### Model Loading Issues

- Verify that `model.h5` and `model.weights.h5` files are in the backend directory
- Check that TensorFlow and MediaPipe are properly installed
- Ensure sufficient system resources for model loading

### API Connection Issues

- Verify the backend server is running on port 8000
- Check that CORS is properly configured
- Ensure the frontend proxy is set to `http://localhost:8000`

## Development

### Adding New Signs

1. Retrain the model with new sign data
2. Update the `signs` array in `backend/main.py`
3. Add corresponding quiz questions
4. Update the frontend sign descriptions

### Customizing the UI

- Modify CSS in `frontend/src/index.css`
- Update component styles in individual component files
- Add new routes in `frontend/src/App.js`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- British Sign Language community
- TensorFlow and MediaPipe teams
- React and FastAPI communities

---

**Note**: This application requires your trained LSTM model files (`model.h5` and `model.weights.h5`) to function properly. Make sure to place these files in the backend directory before running the application.
