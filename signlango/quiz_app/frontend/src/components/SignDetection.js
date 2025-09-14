import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import config from '../config';
import './SignDetection.css';

function SignDetection() {
  // State for sign selection and learning flow
  const [selectedSign, setSelectedSign] = useState(null);
  const [learningStep, setLearningStep] = useState('select'); // 'select', 'watch', 'practice', 'result'
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const [detectionResult, setDetectionResult] = useState(null);
  const [cameraError, setCameraError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // Video and camera refs
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const detectionIntervalRef = useRef(null);
  const countdownIntervalRef = useRef(null);

  // Available signs with video references
  const availableSigns = [
    {
      id: 'hi',
      name: 'Hello/Hi',
      description: 'Wave hand side to side',
      vidref: 'cnyia0upyj',
      videoUrl: 'https://www.signbsl.com/sign/hello'
    },
    {
      id: 'please',
      name: 'Please',
      description: 'Rub palm in circular motion on chest',
      vidref: 'f7a1xiefkh',
      videoUrl: 'https://www.signbsl.com/sign/please'
    },
    {
      id: 'excuse_me',
      name: 'Excuse Me',
      description: 'Tap shoulder to get attention',
      vidref: '26ojajoxrq',
      videoUrl: 'https://www.signbsl.com/sign/excuse-me'
    },
    {
      id: 'okay',
      name: 'Okay',
      description: 'Give a thumbs up gesture',
      vidref: 'cj1jijzqra',
      videoUrl: 'https://www.signbsl.com/sign/okay'
    }
  ];

  // Load SignBSL widget script
  useEffect(() => {
    if (!window.signbsl) {
      const script = document.createElement('script');
      script.src = 'https://embed.signbsl.com/widgets.js';
      script.async = true;
      script.charset = 'utf-8';
      
      script.onload = () => {
        if (window.signbsl) {
          window.signbsl.init();
        }
      };
      
      document.head.appendChild(script);
    } else {
      window.signbsl.init();
    }
  }, []);

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
          videoRef.current.play().then(() => {
            setIsCameraActive(true);
            console.log('Camera started successfully');
          }).catch((playError) => {
            console.error('Video play error:', playError);
            videoRef.current.muted = true;
            videoRef.current.play().then(() => {
              setIsCameraActive(true);
            });
          });
        };
      }
    } catch (error) {
      console.error('Camera error:', error);
      setIsCameraActive(false);
      
      if (error.name === 'NotAllowedError') {
        setCameraError('Camera permission denied. Please allow camera access and try again.');
      } else if (error.name === 'NotFoundError') {
        setCameraError('No camera found. Please connect a camera and try again.');
      } else {
        setCameraError(`Camera error: ${error.message}`);
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Stop camera function
  const stopCamera = useCallback(() => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
      countdownIntervalRef.current = null;
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    setIsCameraActive(false);
    setIsRecording(false);
    setCountdown(0);
  }, []);

  // Start practice session
  const startPractice = async () => {
    if (!isCameraActive) {
      await startCamera();
    }
    
    setLearningStep('practice');
    setIsRecording(true);
    setCountdown(3);
    
    // Start countdown
    countdownIntervalRef.current = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          clearInterval(countdownIntervalRef.current);
          countdownIntervalRef.current = null;
          startDetection();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  // Start detection
  const startDetection = () => {
    detectionIntervalRef.current = setInterval(captureAndDetect, 100);
  };

  // Capture and detect function
  const captureAndDetect = useCallback(async () => {
    if (!isCameraActive || !videoRef.current || !videoRef.current.videoWidth) {
      return;
    }

    try {
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      context.drawImage(videoRef.current, 0, 0);
      
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      
      const response = await axios.post(`${config.API_BASE_URL}${config.API_ENDPOINTS.DETECT}`, {
        image: imageData
      });

      const { detected_sign, confidence, all_probabilities } = response.data;
      
      // Check if the detected sign matches the selected sign
      if (detected_sign === selectedSign.id && confidence > 0.7) {
        // Success! Stop detection and show result
        if (detectionIntervalRef.current) {
          clearInterval(detectionIntervalRef.current);
          detectionIntervalRef.current = null;
        }
        
        setIsRecording(false);
        setDetectionResult({
          success: true,
          detectedSign: detected_sign,
          confidence: confidence,
          message: 'Excellent! You signed correctly!'
        });
        
        setLearningStep('result');
      }
    } catch (error) {
      console.error('Detection error:', error);
    }
  }, [isCameraActive, selectedSign]);

  // Reset and go back to selection
  const resetLearning = () => {
    stopCamera();
    setSelectedSign(null);
    setLearningStep('select');
    setDetectionResult(null);
  };

  // Try again
  const tryAgain = () => {
    setDetectionResult(null);
    setLearningStep('watch');
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
      }
      if (countdownIntervalRef.current) {
        clearInterval(countdownIntervalRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Render sign selection screen
  const renderSignSelection = () => (
    <div className="sign-selection">
      <div className="selection-header">
        <h1>Learn BSL Signs</h1>
        <p>Choose a sign to learn and practice with real-time feedback</p>
      </div>
      
      <div className="signs-grid">
        {availableSigns.map((sign) => (
          <div 
            key={sign.id} 
            className="sign-card"
            onClick={() => {
              setSelectedSign(sign);
              setLearningStep('watch');
            }}
          >
            <div className="sign-icon">
              {sign.id === 'hi' && '👋'}
              {sign.id === 'please' && '🙏'}
              {sign.id === 'excuse_me' && '🤚'}
              {sign.id === 'okay' && '👍'}
            </div>
            <h3>{sign.name}</h3>
            <p>{sign.description}</p>
            <div className="sign-action">Click to Learn →</div>
          </div>
        ))}
      </div>
    </div>
  );

  // Render video watching screen
  const renderVideoWatch = () => (
    <div className="video-watch">
      <div className="video-header">
        <button className="back-btn" onClick={() => setLearningStep('select')}>
          ← Back to Signs
        </button>
        <h2>Learn: {selectedSign.name}</h2>
        <p>{selectedSign.description}</p>
      </div>
      
      <div className="video-container">
        <div className="video-embed-wrapper">
          <blockquote 
            className="signbsldata-embed" 
            data-vidref={selectedSign.vidref}
            style={{
              borderRadius: '10px',
              border: 'none',
              boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)',
              margin: '0',
              padding: '0'
            }}
          >
            <a href={selectedSign.videoUrl}>
              Watch how to sign '{selectedSign.name.toLowerCase()}' in British Sign Language
            </a>
          </blockquote>
        </div>
      </div>
      
      <div className="practice-tips">
        <h3>Practice Tips</h3>
        <ul>
          <li>Watch the video carefully and note the hand movements</li>
          <li>Pay attention to the position and shape of the hands</li>
          <li>Practice the movement a few times before recording</li>
          <li>Make sure you have good lighting and clear camera view</li>
        </ul>
      </div>
      
      <div className="practice-actions">
        <button 
          className="btn btn-primary"
          onClick={startPractice}
          disabled={isLoading}
        >
          {isLoading ? 'Starting Camera...' : 'Start Practice'}
        </button>
      </div>
    </div>
  );

  // Render practice screen
  const renderPractice = () => (
    <div className="practice-screen">
      <div className="practice-header">
        <h2>Practice: {selectedSign.name}</h2>
        <p>Perform the sign when the countdown finishes</p>
      </div>
      
      <div className="camera-section">
        <div className="camera-container">
          {isCameraActive ? (
            <div className="video-wrapper">
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                style={{ 
                  width: '100%', 
                  maxWidth: '640px', 
                  borderRadius: '10px',
                  border: '2px solid #28a745'
                }}
              />
              {countdown > 0 && (
                <div className="countdown-overlay">
                  <div className="countdown-number">{countdown}</div>
                  <p>Get ready...</p>
                </div>
              )}
              {isRecording && countdown === 0 && (
                <div className="recording-overlay">
                  <div className="recording-indicator">● Recording</div>
                  <p>Perform the sign now!</p>
                </div>
              )}
            </div>
          ) : (
            <div className="camera-placeholder">
              <p>Camera not active</p>
            </div>
          )}
        </div>
      </div>
      
      <div className="practice-controls">
        <button className="btn btn-warning" onClick={stopCamera}>
          Stop Camera
        </button>
        <button className="btn btn-secondary" onClick={() => setLearningStep('watch')}>
          Back to Video
        </button>
      </div>
      
      {cameraError && (
        <div className="camera-error">
          <p>{cameraError}</p>
          <button className="btn btn-primary" onClick={startCamera}>
            Retry Camera
          </button>
        </div>
      )}
    </div>
  );

  // Render result screen
  const renderResult = () => (
    <div className="result-screen">
      <div className="result-content">
        {detectionResult.success ? (
          <div className="success-result">
            <div className="success-icon">🎉</div>
            <h2>Excellent!</h2>
            <p>You signed "{selectedSign.name}" correctly!</p>
            <div className="result-details">
              <p><strong>Confidence:</strong> {(detectionResult.confidence * 100).toFixed(1)}%</p>
            </div>
          </div>
        ) : (
          <div className="failure-result">
            <div className="failure-icon">😔</div>
            <h2>Keep Practicing!</h2>
            <p>Your sign wasn't quite right. Try watching the video again and practice more.</p>
          </div>
        )}
        
        <div className="result-actions">
          <button className="btn btn-primary" onClick={tryAgain}>
            Try Again
          </button>
          <button className="btn btn-secondary" onClick={resetLearning}>
            Learn Another Sign
          </button>
        </div>
      </div>
    </div>
  );

  // Main render
  return (
    <div className="sign-detection">
      <div className="card">
        {learningStep === 'select' && renderSignSelection()}
        {learningStep === 'watch' && renderVideoWatch()}
        {learningStep === 'practice' && renderPractice()}
        {learningStep === 'result' && renderResult()}
      </div>
    </div>
  );
}

export default SignDetection; 