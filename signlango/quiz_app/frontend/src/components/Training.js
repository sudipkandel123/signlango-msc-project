import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import './Training.css';

function Training() {
  // State management
  const [availableSigns, setAvailableSigns] = useState([]);
  const [selectedSign, setSelectedSign] = useState(null);
  const [currentSession, setCurrentSession] = useState(null);
  const [sessionMode, setSessionMode] = useState('learn'); // learn, practice, test
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const [detectionResult, setDetectionResult] = useState(null);
  const [sessionResults, setSessionResults] = useState(null);
  const [userProgress, setUserProgress] = useState(null);
  const [cameraError, setCameraError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState('select'); // select, learn, practice, test, results
  
  // Video and camera refs
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const detectionIntervalRef = useRef(null);
  const countdownIntervalRef = useRef(null);
  const canvasRef = useRef(null);

  // Session tracking
  const [frameCount, setFrameCount] = useState(0);
  const [correctDetections, setCorrectDetections] = useState(0);
  const [sessionStartTime, setSessionStartTime] = useState(null);

  // Load available signs
  useEffect(() => {
    loadAvailableSigns();
    loadUserProgress();
  }, []);

  const loadAvailableSigns = async () => {
    try {
      const response = await axios.get('/training/signs');
      setAvailableSigns(response.data.signs);
    } catch (error) {
      console.error('Error loading signs:', error);
    }
  };

  const loadUserProgress = async () => {
    try {
      const response = await axios.get('/training/progress/user123');
      setUserProgress(response.data);
    } catch (error) {
      console.error('Error loading progress:', error);
    }
  };

  // Start camera function
  const startCamera = useCallback(async () => {
    try {
      setIsLoading(true);
      setCameraError('');
      
      // Stop any existing stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }

      // Request camera
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        }, 
        audio: false 
      });
      
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          setIsCameraActive(true);
          setIsLoading(false);
        };
      }
    } catch (error) {
      console.error('Camera error:', error);
      setCameraError('Failed to access camera. Please check permissions.');
      setIsLoading(false);
    }
  }, []);

  // Stop camera function
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsCameraActive(false);
    setIsRecording(false);
    setCountdown(0);
    
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
      countdownIntervalRef.current = null;
    }
  }, []);

  // Start training session
  const startSession = async (sign, mode) => {
    try {
      setIsLoading(true);
      
      const response = await axios.post('/training/start-session', {
        sign_id: sign.id,
        session_type: mode
      });
      
      setCurrentSession(response.data);
      setSelectedSign(sign);
      setSessionMode(mode);
      setCurrentStep('learn');
      setFrameCount(0);
      setCorrectDetections(0);
      setSessionStartTime(Date.now());
      
      // Start camera
      await startCamera();
      
    } catch (error) {
      console.error('Error starting session:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Start recording for practice/test
  const startRecording = () => {
    if (!isCameraActive) return;
    
    setIsRecording(true);
    setCountdown(3);
    
    countdownIntervalRef.current = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          clearInterval(countdownIntervalRef.current);
          startDetection();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  // Start detection loop
  const startDetection = () => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
    }
    
    detectionIntervalRef.current = setInterval(async () => {
      if (videoRef.current && canvasRef.current) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        
        // Draw video frame to canvas
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        ctx.drawImage(videoRef.current, 0, 0);
        
        // Get frame data
        const frameData = canvas.toDataURL('image/jpeg', 0.8);
        
        try {
          const response = await axios.post('/training/process-frame', {
            frame_data: frameData,
            session_id: currentSession.session_id,
            sign_id: selectedSign.id,
            mode: sessionMode === 'practice' ? 'training' : 'detection'
          });
          
          setFrameCount(prev => prev + 1);
          
          if (response.data.status === 'detection_complete') {
            if (response.data.is_correct) {
              setCorrectDetections(prev => prev + 1);
            }
            setDetectionResult(response.data);
          }
          
        } catch (error) {
          console.error('Error processing frame:', error);
        }
      }
    }, 1000); // Process frame every second
  };

  // Stop recording and end session
  const stopRecording = async () => {
    setIsRecording(false);
    
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    
    try {
      const response = await axios.post('/training/end-session', {
        session_id: currentSession.session_id,
        sign_id: selectedSign.id,
        total_frames: frameCount,
        correct_detections: correctDetections
      });
      
      setSessionResults(response.data);
      setCurrentStep('results');
      
    } catch (error) {
      console.error('Error ending session:', error);
    }
  };

  // Reset session
  const resetSession = () => {
    setCurrentSession(null);
    setSelectedSign(null);
    setSessionMode('learn');
    setCurrentStep('select');
    setDetectionResult(null);
    setSessionResults(null);
    setFrameCount(0);
    setCorrectDetections(0);
    stopCamera();
  };

  // Continue to next step
  const continueToNext = () => {
    if (currentStep === 'learn') {
      setCurrentStep('practice');
    } else if (currentStep === 'practice') {
      setCurrentStep('test');
    } else if (currentStep === 'test') {
      startRecording();
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return (
    <div className="training-container">
      <div className="training-header">
        <h1>BSL Training Center</h1>
        <p>Learn, practice, and master British Sign Language with AI-powered feedback</p>
      </div>

      {currentStep === 'select' && (
        <div className="sign-selection">
          <h2>Choose a Sign to Learn</h2>
          <div className="signs-grid">
            {availableSigns.map((sign) => (
              <div key={sign.id} className="sign-card">
                <h3>{sign.name}</h3>
                <p>{sign.description}</p>
                <div className="sign-difficulty">
                  <span className={`difficulty-badge ${sign.difficulty}`}>
                    {sign.difficulty}
                  </span>
                </div>
                {userProgress && userProgress.signs_progress[sign.id] && (
                  <div className="sign-progress">
                    <div className="progress-bar">
                      <div 
                        className="progress-fill" 
                        style={{width: `${userProgress.signs_progress[sign.id].accuracy}%`}}
                      ></div>
                    </div>
                    <span>{userProgress.signs_progress[sign.id].accuracy.toFixed(1)}% accuracy</span>
                  </div>
                )}
                <button 
                  className="start-training-btn"
                  onClick={() => startSession(sign, 'learn')}
                  disabled={isLoading}
                >
                  {isLoading ? 'Starting...' : 'Start Training'}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {currentStep === 'learn' && selectedSign && (
        <div className="learning-step">
          <h2>Learn: {selectedSign.name}</h2>
          <div className="learning-content">
            <div className="video-section">
              <h3>Watch the Sign</h3>
              <div className="video-placeholder">
                <p>Video demonstration will appear here</p>
                <a href={selectedSign.videoUrl} target="_blank" rel="noopener noreferrer">
                  Watch on SignBSL
                </a>
              </div>
            </div>
            
            <div className="instructions-section">
              <h3>Step-by-Step Instructions</h3>
              <ol className="instructions-list">
                {selectedSign.instructions.map((instruction, index) => (
                  <li key={index}>{instruction}</li>
                ))}
              </ol>
              
              <h3>Tips for Success</h3>
              <ul className="tips-list">
                {selectedSign.tips.map((tip, index) => (
                  <li key={index}>{tip}</li>
                ))}
              </ul>
            </div>
          </div>
          
          <div className="step-navigation">
            <button className="btn-secondary" onClick={resetSession}>
              Back to Selection
            </button>
            <button className="btn-primary" onClick={continueToNext}>
              Continue to Practice
            </button>
          </div>
        </div>
      )}

      {currentStep === 'practice' && selectedSign && (
        <div className="practice-step">
          <h2>Practice: {selectedSign.name}</h2>
          <p>Practice the sign in front of your camera. The AI will help you improve!</p>
          
          <div className="camera-section">
            {!isCameraActive ? (
              <div className="camera-placeholder">
                <p>Camera not active</p>
                <button className="btn-primary" onClick={startCamera}>
                  Start Camera
                </button>
              </div>
            ) : (
              <div className="video-container">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="training-video"
                />
                <canvas ref={canvasRef} style={{ display: 'none' }} />
                
                {detectionResult && (
                  <div className={`detection-feedback ${detectionResult.is_correct ? 'correct' : 'incorrect'}`}>
                    <h3>{detectionResult.is_correct ? '✅ Correct!' : '❌ Try Again'}</h3>
                    <p>Confidence: {(detectionResult.confidence * 100).toFixed(1)}%</p>
                    <p>{detectionResult.feedback}</p>
                  </div>
                )}
              </div>
            )}
          </div>
          
          <div className="step-navigation">
            <button className="btn-secondary" onClick={() => setCurrentStep('learn')}>
              Back to Learning
            </button>
            <button className="btn-primary" onClick={continueToNext}>
              Ready for Test
            </button>
          </div>
        </div>
      )}

      {currentStep === 'test' && selectedSign && (
        <div className="test-step">
          <h2>Test: {selectedSign.name}</h2>
          <p>Show the sign to test your mastery. You'll get a score based on accuracy!</p>
          
          <div className="camera-section">
            {!isCameraActive ? (
              <div className="camera-placeholder">
                <p>Camera not active</p>
                <button className="btn-primary" onClick={startCamera}>
                  Start Camera
                </button>
              </div>
            ) : (
              <div className="video-container">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="training-video"
                />
                <canvas ref={canvasRef} style={{ display: 'none' }} />
                
                {countdown > 0 && (
                  <div className="countdown-overlay">
                    <div className="countdown-number">{countdown}</div>
                    <p>Get ready...</p>
                  </div>
                )}
                
                {isRecording && countdown === 0 && (
                  <div className="recording-overlay">
                    <div className="recording-indicator">🔴 Recording</div>
                    <p>Show the sign now!</p>
                    <button className="btn-stop" onClick={stopRecording}>
                      Stop Recording
                    </button>
                  </div>
                )}
                
                {detectionResult && (
                  <div className={`detection-feedback ${detectionResult.is_correct ? 'correct' : 'incorrect'}`}>
                    <h3>{detectionResult.is_correct ? '✅ Correct!' : '❌ Try Again'}</h3>
                    <p>Confidence: {(detectionResult.confidence * 100).toFixed(1)}%</p>
                    <p>{detectionResult.feedback}</p>
                  </div>
                )}
              </div>
            )}
          </div>
          
          <div className="test-stats">
            <div className="stat">
              <span className="stat-label">Frames Processed:</span>
              <span className="stat-value">{frameCount}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Correct Detections:</span>
              <span className="stat-value">{correctDetections}</span>
            </div>
          </div>
        </div>
      )}

      {currentStep === 'results' && sessionResults && (
        <div className="results-step">
          <h2>Training Results</h2>
          
          <div className="results-card">
            <div className="result-header">
              <h3>{selectedSign.name}</h3>
              <div className={`performance-badge ${sessionResults.performance}`}>
                {sessionResults.performance}
              </div>
            </div>
            
            <div className="results-stats">
              <div className="result-stat">
                <span className="stat-label">Accuracy:</span>
                <span className="stat-value">{sessionResults.accuracy.toFixed(1)}%</span>
              </div>
              <div className="result-stat">
                <span className="stat-label">Total Frames:</span>
                <span className="stat-value">{sessionResults.total_frames}</span>
              </div>
              <div className="result-stat">
                <span className="stat-label">Correct Detections:</span>
                <span className="stat-value">{sessionResults.correct_detections}</span>
              </div>
            </div>
            
            <div className="result-message">
              <p>{sessionResults.message}</p>
            </div>
          </div>
          
          <div className="step-navigation">
            <button className="btn-secondary" onClick={resetSession}>
              Back to Selection
            </button>
            <button className="btn-primary" onClick={() => startSession(selectedSign, 'practice')}>
              Practice Again
            </button>
          </div>
        </div>
      )}

      {cameraError && (
        <div className="error-message">
          <p>{cameraError}</p>
        </div>
      )}
    </div>
  );
}

export default Training; 