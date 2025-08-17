import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './Quiz.css';

function Quiz() {
  const navigate = useNavigate();
  const [questions, setQuestions] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [score, setScore] = useState(0);
  const [showResult, setShowResult] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeLeft, setTimeLeft] = useState(30);
  const [timerActive, setTimerActive] = useState(false);
  const [quizStarted, setQuizStarted] = useState(false);
  const [quizCompleted, setQuizCompleted] = useState(false);
  const [showHint, setShowHint] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [difficulty, setDifficulty] = useState('beginner');
  const [streak, setStreak] = useState(0);
  const [maxStreak, setMaxStreak] = useState(0);
  const [correctAnswers, setCorrectAnswers] = useState([]);
  const [incorrectAnswers, setIncorrectAnswers] = useState([]);
  const [showStats, setShowStats] = useState(false);

  const categories = [
    { id: 'all', name: 'All Categories', icon: '🎯' },
    { id: 'basics', name: 'Basic Signs', icon: '👋' },
    { id: 'family', name: 'Family & Relationships', icon: '👨‍👩‍👧‍👦' },
    { id: 'emotions', name: 'Emotions & Feelings', icon: '😊' },
    { id: 'colors', name: 'Colors', icon: '🎨' },
    { id: 'numbers', name: 'Numbers', icon: '🔢' },
    { id: 'weather', name: 'Weather', icon: '🌤️' },
    { id: 'food', name: 'Food & Drinks', icon: '🍽️' },
    { id: 'travel', name: 'Travel & Transport', icon: '✈️' },
    { id: 'education', name: 'Education', icon: '🎓' },
    { id: 'work', name: 'Work & Jobs', icon: '💼' },
    { id: 'technology', name: 'Technology', icon: '💻' },
    { id: 'sports', name: 'Sports & Activities', icon: '⚽' },
    { id: 'music', name: 'Music & Arts', icon: '🎵' },
    { id: 'health', name: 'Health & Medical', icon: '🏥' },
    { id: 'emergency', name: 'Emergency', icon: '🚨' },
    { id: 'legal', name: 'Legal & Law', icon: '⚖️' },
    { id: 'finance', name: 'Finance & Money', icon: '💰' },
    { id: 'environment', name: 'Environment', icon: '🌳' },
    { id: 'celebration', name: 'Celebration', icon: '🎉' },
    { id: 'conversation', name: 'Conversation', icon: '💬' }
  ];

  const difficulties = [
    { id: 'beginner', name: 'Beginner', color: '#28a745' },
    { id: 'intermediate', name: 'Intermediate', color: '#ffc107' },
    { id: 'advanced', name: 'Advanced', color: '#dc3545' }
  ];

  useEffect(() => {
    fetchQuestions();
  }, [selectedCategory, difficulty]);

  useEffect(() => {
    let interval = null;
    if (timerActive && timeLeft > 0 && quizStarted && !showResult) {
      interval = setInterval(() => {
        setTimeLeft(timeLeft => {
          if (timeLeft <= 1) {
            handleTimeUp();
            return 0;
          }
          return timeLeft - 1;
        });
      }, 1000);
    } else if (timeLeft === 0) {
      handleTimeUp();
    }
    return () => clearInterval(interval);
  }, [timerActive, timeLeft, quizStarted, showResult]);

  const fetchQuestions = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`http://localhost:8000/quiz?category=${selectedCategory}&difficulty=${difficulty}`);
      const quizData = response.data.questions || response.data;
      setQuestions(quizData);
      setError(null);
    } catch (err) {
      console.error('Error fetching questions:', err);
      setError('Failed to load quiz questions. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const startQuiz = () => {
    setQuizStarted(true);
    setTimerActive(true);
    setTimeLeft(30);
    setCurrentQuestionIndex(0);
    setScore(0);
    setStreak(0);
    setMaxStreak(0);
    setCorrectAnswers([]);
    setIncorrectAnswers([]);
    setShowResult(false);
    setQuizCompleted(false);
    setShowStats(false);
  };

  const handleAnswerSelect = (answer) => {
    if (!showResult && timerActive) {
      setSelectedAnswer(answer);
    }
  };

  const handleSubmitAnswer = async () => {
    if (selectedAnswer === null || showResult) return;

    try {
      const response = await axios.post('http://localhost:8000/quiz/submit', {
        question_id: questions[currentQuestionIndex].id,
        selected_answer: selectedAnswer
      });

      const isCorrect = response.data.correct;
      
      if (isCorrect) {
        setScore(score + 1);
        setStreak(streak + 1);
        setMaxStreak(Math.max(maxStreak, streak + 1));
        setCorrectAnswers([...correctAnswers, questions[currentQuestionIndex]]);
      } else {
        setStreak(0);
        setIncorrectAnswers([...incorrectAnswers, questions[currentQuestionIndex]]);
      }

      setShowResult(true);
      setTimerActive(false);
    } catch (err) {
      console.error('Error submitting answer:', err);
      setShowResult(true);
    }
  };

  const handleTimeUp = () => {
    setTimerActive(false);
    setShowResult(true);
    setStreak(0);
    setIncorrectAnswers([...incorrectAnswers, questions[currentQuestionIndex]]);
  };

  const handleNextQuestion = () => {
    setSelectedAnswer(null);
    setShowResult(false);
    setShowHint(false);
    setCurrentQuestionIndex(currentQuestionIndex + 1);
    setTimeLeft(30);
    setTimerActive(true);
  };

  const handleRestartQuiz = () => {
    setQuizStarted(false);
    setQuizCompleted(false);
    setShowStats(false);
    setCurrentQuestionIndex(0);
    setSelectedAnswer(null);
    setScore(0);
    setStreak(0);
    setMaxStreak(0);
    setCorrectAnswers([]);
    setIncorrectAnswers([]);
    setShowResult(false);
    setTimeLeft(30);
    setTimerActive(false);
  };

  const toggleHint = () => {
    setShowHint(!showHint);
  };

  const getTimeColor = () => {
    if (timeLeft > 20) return '#28a745';
    if (timeLeft > 10) return '#ffc107';
    return '#dc3545';
  };

  const getPerformanceMessage = () => {
    const percentage = Math.round((score / questions.length) * 100);
    if (percentage >= 90) return { message: "Excellent! You're a BSL expert!", emoji: "🏆" };
    if (percentage >= 80) return { message: "Great job! You're doing really well!", emoji: "🎉" };
    if (percentage >= 70) return { message: "Good work! Keep practicing!", emoji: "👍" };
    if (percentage >= 60) return { message: "Not bad! You're on the right track!", emoji: "📈" };
    return { message: "Keep practicing! You'll get better!", emoji: "💪" };
  };

  const handleBackToHome = () => {
    navigate('/');
  };

  if (loading) {
    return (
      <div className="quiz-container">
        <div className="quiz-card">
          <div className="loading-spinner">
            <div className="spinner"></div>
            <h2>Loading Quiz Questions...</h2>
            <p>Preparing your BSL learning experience</p>
            <button onClick={handleBackToHome} className="btn btn-secondary">
              ← Back to Home
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="quiz-container">
        <div className="quiz-card">
          <div className="error-message">
            <h2>⚠️ Error Loading Quiz</h2>
            <p>{error}</p>
            <div className="error-actions">
              <button onClick={fetchQuestions} className="btn btn-primary">
                Try Again
              </button>
              <button onClick={handleBackToHome} className="btn btn-secondary">
                ← Back to Home
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (questions.length === 0) {
    return (
      <div className="quiz-container">
        <div className="quiz-card">
          <div className="no-questions">
            <h2>📝 No Questions Available</h2>
            <p>No quiz questions are available for the selected category and difficulty.</p>
            <div className="no-questions-actions">
              <button onClick={() => setSelectedCategory('all')} className="btn btn-primary">
                Try All Categories
              </button>
              <button onClick={handleBackToHome} className="btn btn-secondary">
                ← Back to Home
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!quizStarted) {
    return (
      <div className="quiz-container">
        <div className="quiz-card">
          <div className="quiz-setup">
            <h1>🎯 BSL Quiz Challenge</h1>
            <p>Test your knowledge of British Sign Language</p>
            
            <div className="setup-sections">
              <div className="setup-section">
                <h3>📂 Select Category</h3>
                <div className="category-grid">
                  {categories.map(category => (
                    <button
                      key={category.id}
                      className={`category-btn ${selectedCategory === category.id ? 'active' : ''}`}
                      onClick={() => setSelectedCategory(category.id)}
                    >
                      <span className="category-icon">{category.icon}</span>
                      <span className="category-name">{category.name}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="setup-section">
                <h3>🎚️ Select Difficulty</h3>
                <div className="difficulty-grid">
                  {difficulties.map(diff => (
                    <button
                      key={diff.id}
                      className={`difficulty-btn ${difficulty === diff.id ? 'active' : ''}`}
                      onClick={() => setDifficulty(diff.id)}
                      style={{ '--difficulty-color': diff.color }}
                    >
                      {diff.name}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="quiz-info">
              <div className="info-item">
                <span className="info-icon">⏱️</span>
                <span>30 seconds per question</span>
              </div>
              <div className="info-item">
                <span className="info-icon">🎯</span>
                <span>{questions.length} questions</span>
              </div>
              <div className="info-item">
                <span className="info-icon">🏆</span>
                <span>Track your progress</span>
              </div>
              <div className="info-item">
                <span className="info-icon">💡</span>
                <span>Try different difficulties for more questions</span>
              </div>
            </div>

            <div className="quiz-actions">
              <button onClick={startQuiz} className="btn btn-primary btn-large">
                Start Quiz
              </button>
              <button onClick={handleBackToHome} className="btn btn-secondary">
                ← Back to Home
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (quizCompleted) {
    const percentage = Math.round((score / questions.length) * 100);
    const performance = getPerformanceMessage();
    
    return (
      <div className="quiz-container">
        <div className="quiz-card">
          <div className="quiz-complete">
            <div className="completion-header">
              <h1>{performance.emoji} Quiz Complete!</h1>
              <p className="performance-message">{performance.message}</p>
            </div>

            <div className="score-display">
              <div className="score-circle">
                <div className="score-percentage">{percentage}%</div>
                <div className="score-fraction">{score}/{questions.length}</div>
              </div>
            </div>

            <div className="stats-grid">
              <div className="stat-item">
                <div className="stat-icon">🔥</div>
                <div className="stat-value">{maxStreak}</div>
                <div className="stat-label">Best Streak</div>
              </div>
              <div className="stat-item">
                <div className="stat-icon">✅</div>
                <div className="stat-value">{correctAnswers.length}</div>
                <div className="stat-label">Correct</div>
              </div>
              <div className="stat-item">
                <div className="stat-icon">❌</div>
                <div className="stat-value">{incorrectAnswers.length}</div>
                <div className="stat-label">Incorrect</div>
              </div>
            </div>

            <div className="action-buttons">
              <button onClick={() => setShowStats(!showStats)} className="btn btn-secondary">
                {showStats ? 'Hide' : 'Show'} Detailed Results
              </button>
              <button onClick={handleRestartQuiz} className="btn btn-primary">
                Try Again
              </button>
              <button onClick={() => {
                setQuizCompleted(false);
                setQuizStarted(false);
                setSelectedCategory('all');
                setDifficulty('beginner');
              }} className="btn btn-success">
                🎯 Take Another Quiz
              </button>
              <button onClick={handleBackToHome} className="btn btn-secondary">
                ← Back to Home
              </button>
            </div>

            {showStats && (
              <div className="detailed-results">
                <h3>📊 Detailed Results</h3>
                <div className="results-section">
                  <h4>✅ Correct Answers ({correctAnswers.length})</h4>
                  {correctAnswers.map((question, index) => (
                    <div key={index} className="result-item correct">
                      <span className="question-text">{question.question}</span>
                      <span className="result-status">✓ Correct</span>
                    </div>
                  ))}
                </div>
                <div className="results-section">
                  <h4>❌ Incorrect Answers ({incorrectAnswers.length})</h4>
                  {incorrectAnswers.map((question, index) => (
                    <div key={index} className="result-item incorrect">
                      <span className="question-text">{question.question}</span>
                      <span className="result-status">✗ Incorrect</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  const isLastQuestion = currentQuestionIndex === questions.length - 1;
  const currentQuestion = questions[currentQuestionIndex];

  if (!currentQuestion) {
    return (
      <div className="quiz-container">
        <div className="quiz-card">
          <div className="error-message">
            <h2>❌ Question Not Found</h2>
            <p>There was an error loading the current question.</p>
            <button onClick={handleRestartQuiz} className="btn btn-primary">
              Restart Quiz
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="quiz-container">
      <div className="quiz-card">
        <div className="quiz-header">
          <div className="quiz-navigation">
            <button onClick={handleBackToHome} className="btn btn-secondary btn-small">
              ← Back to Home
            </button>
          </div>
          
          <div className="quiz-progress">
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${((currentQuestionIndex + 1) / questions.length) * 100}%` }}
              ></div>
            </div>
            <div className="progress-text">
              Question {currentQuestionIndex + 1} of {questions.length}
            </div>
          </div>

          <div className="quiz-stats">
            <div className="stat">
              <span className="stat-label">Score</span>
              <span className="stat-value">{score}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Streak</span>
              <span className="stat-value streak">{streak}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Time</span>
              <span className="stat-value timer" style={{ color: getTimeColor() }}>
                {timeLeft}s
              </span>
            </div>
          </div>
        </div>

        {!showResult ? (
          <div className="question-section">
            <div className="question-header">
              <h2 className="question-text">{currentQuestion.question}</h2>
              <button 
                onClick={toggleHint} 
                className="hint-btn"
                disabled={showHint}
              >
                💡 {showHint ? 'Hint Used' : 'Get Hint'}
              </button>
            </div>

            {showHint && currentQuestion.hint && (
              <div className="hint-box">
                <span className="hint-icon">💡</span>
                <span className="hint-text">{currentQuestion.hint}</span>
              </div>
            )}
            
            <div className="answers-grid">
              {currentQuestion.options && currentQuestion.options.map((option, index) => (
                <button
                  key={index}
                  className={`answer-btn ${selectedAnswer === option ? 'selected' : ''}`}
                  onClick={() => handleAnswerSelect(option)}
                  disabled={showResult}
                >
                  <span className="answer-letter">{String.fromCharCode(65 + index)}</span>
                  <span className="answer-text">{option}</span>
                </button>
              ))}
            </div>

            <div className="question-actions">
              <button
                className="btn btn-primary"
                onClick={handleSubmitAnswer}
                disabled={selectedAnswer === null || showResult}
              >
                Submit Answer
              </button>
            </div>
          </div>
        ) : (
          <div className="result-section">
            <div className="result-header">
              <h2>Answer Result</h2>
              <div className={`result-indicator ${selectedAnswer === currentQuestion.correct_answer ? 'correct' : 'incorrect'}`}>
                {selectedAnswer === currentQuestion.correct_answer ? '✅ Correct!' : '❌ Incorrect'}
              </div>
            </div>

            <div className="result-details">
              <div className="result-item">
                <span className="result-label">Your Answer:</span>
                <span className="result-value">{selectedAnswer}</span>
              </div>
              <div className="result-item">
                <span className="result-label">Correct Answer:</span>
                <span className="result-value correct">{currentQuestion.correct_answer}</span>
              </div>
            </div>

            {currentQuestion.explanation && (
              <div className="explanation-box">
                <h4>💡 Explanation</h4>
                <p>{currentQuestion.explanation}</p>
              </div>
            )}

            {currentQuestion.learning_tip && (
              <div className="learning-tip">
                <h4>🎓 Learning Tip</h4>
                <p>{currentQuestion.learning_tip}</p>
              </div>
            )}

            <div className="result-actions">
              {isLastQuestion ? (
                <button 
                  className="btn btn-primary btn-large" 
                  onClick={() => setQuizCompleted(true)}
                >
                  View Final Results
                </button>
              ) : (
                <button className="btn btn-primary btn-large" onClick={handleNextQuestion}>
                  Next Question
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Quiz; 