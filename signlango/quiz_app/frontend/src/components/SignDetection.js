import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';

function SignDetection() {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [detectedSign, setDetectedSign] = useState('No sign detected');
  const [confidence, setConfidence] = useState(0);
  const [sequenceLength, setSequenceLength] = useState(0);
  const [predictionsCount, setPredictionsCount] = useState(0);
  const [sentence, setSentence] = useState([]);
  const [statusMessage, setStatusMessage] = useState('Click "Start Camera" to begin');
  const [allProbabilities, setAllProbabilities] = useState([]);
  const [cameraError, setCameraError] = useState('');
  const [showCameraGuide, setShowCameraGuide] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const detectionIntervalRef = useRef(null);

  const startCamera = useCallback(async () => {
    try {
      setIsLoading(true);
      setCameraError('');
      setStatusMessage('Requesting camera permission...');
      
      // Stop any existing stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }

      // Request camera with basic constraints
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
        
        // Wait for video to be ready
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play().then(() => {
            setIsCameraActive(true);
            setStatusMessage('Camera active - Start signing!');
            console.log('Camera started successfully');
          }).catch((playError) => {
            console.error('Video play error:', playError);
            // Try to play muted
            videoRef.current.muted = true;
            videoRef.current.play().then(() => {
              setIsCameraActive(true);
              setStatusMessage('Camera active (muted) - Start signing!');
            });
          });
        };
        
        videoRef.current.onerror = (error) => {
          console.error('Video error:', error);
          setCameraError('Video playback error');
        };
      }
    } catch (error) {
      console.error('Camera error:', error);
      setIsCameraActive(false);
      
      if (error.name === 'NotAllowedError') {
        setCameraError('Camera permission denied. Please allow camera access and try again.');
        setStatusMessage('Camera access denied');
      } else if (error.name === 'NotFoundError') {
        setCameraError('No camera found. Please connect a camera and try again.');
        setStatusMessage('No camera found');
      } else if (error.name === 'NotSupportedError') {
        setCameraError('Camera not supported. Please use Chrome, Firefox, or Safari.');
        setStatusMessage('Camera not supported');
      } else if (error.name === 'NotReadableError') {
        setCameraError('Camera is in use by another application. Please close other camera apps and try again.');
        setStatusMessage('Camera in use by another app');
      } else {
        setCameraError(`Camera error: ${error.message}`);
        setStatusMessage('Camera error');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    setIsCameraActive(false);
    setStatusMessage('Camera stopped - Click "Start Camera" to begin');
  }, []);

  const resetDetection = async () => {
    try {
      await axios.post('http://localhost:8000/reset-detection');
      setSentence([]);
      setPredictionsCount(0);
      setSequenceLength(0);
      setDetectedSign('No sign detected');
      setConfidence(0);
      setStatusMessage('Detection reset - Start signing!');
    } catch (error) {
      console.error('Error resetting detection:', error);
      setStatusMessage('Reset failed - check connection');
    }
  };

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
      
      const response = await axios.post('http://localhost:8000/detect-sign-base64', {
        image: imageData
      });

      const { detected_sign, confidence: conf, sequence_length, predictions_count, sentence: sent, status_message, all_probabilities } = response.data;
      
      setDetectedSign(detected_sign);
      setConfidence(conf);
      setSequenceLength(sequence_length);
      setPredictionsCount(predictions_count);
      setSentence(sent);
      setStatusMessage(status_message);
      setAllProbabilities(all_probabilities);
    } catch (error) {
      console.error('Detection error:', error);
      setStatusMessage('Detection error - check connection');
    }
  }, [isCameraActive]);

  useEffect(() => {
    if (isCameraActive) {
      detectionIntervalRef.current = setInterval(captureAndDetect, 100);
      return () => {
        if (detectionIntervalRef.current) {
          clearInterval(detectionIntervalRef.current);
          detectionIntervalRef.current = null;
        }
      };
    }
  }, [isCameraActive, captureAndDetect]);

  useEffect(() => {
    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="sign-detection">
      <div className="card">
        <h1>BSL Sign Detection</h1>
        <p>Use your webcam to practice BSL signs and get real-time feedback!</p>
        
        {/* Camera Error Display */}
        {cameraError && (
          <div className="camera-error">
            <h3>Camera Issue</h3>
            <p>{cameraError}</p>
            <div className="error-actions">
              <button onClick={startCamera} className="btn btn-primary" disabled={isLoading}>
                {isLoading ? 'Loading...' : 'Retry Camera'}
              </button>
              <button onClick={() => setShowCameraGuide(!showCameraGuide)} className="btn btn-secondary">
                Show Camera Guide
              </button>
            </div>
          </div>
        )}

        {/* Camera Guide */}
        {showCameraGuide && (
          <div className="camera-guide">
            <h3>Camera Setup Guide</h3>
            <div className="guide-content">
              <h4>Browser Compatibility:</h4>
              <ul>
                <li>Chrome (recommended)</li>
                <li>Firefox</li>
                <li>Safari</li>
                <li>Not supported: Internet Explorer</li>
              </ul>
              
              <h4>Troubleshooting:</h4>
              <ul>
                <li>Make sure your camera is connected and working</li>
                <li>Close other applications using the camera</li>
                <li>Check browser settings for camera permissions</li>
                <li>Try refreshing the page after allowing permissions</li>
                <li>Ensure you're using HTTPS or localhost</li>
              </ul>
            </div>
          </div>
        )}

        <div className="camera-section">
          <div className="camera-container">
            {isCameraActive ? (
              <div style={{ position: 'relative' }}>
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  style={{ 
                    width: '100%', 
                    maxWidth: '640px', 
                    borderRadius: '10px',
                    border: '2px solid #28a745',
                    display: 'block'
                  }}
                />
                <div style={{
                  position: 'absolute',
                  top: '10px',
                  left: '10px',
                  background: 'rgba(0,0,0,0.7)',
                  color: 'white',
                  padding: '5px 10px',
                  borderRadius: '5px',
                  fontSize: '12px'
                }}>
                  Camera Active
                </div>
              </div>
            ) : (
              <div className="camera-placeholder">
                <div className="placeholder-content">
                  <div className="camera-icon">Camera</div>
                  <h3>Camera Not Active</h3>
                  <p>Click "Start Camera" to begin sign detection</p>
                  <button 
                    onClick={startCamera} 
                    className="btn btn-primary" 
                    disabled={isLoading}
                  >
                    {isLoading ? 'Starting...' : 'Start Camera'}
                  </button>
                </div>
              </div>
            )}
          </div>

          <div className="camera-controls">
            <div className="camera-status">
              <span className="status-indicator">
                Camera {isCameraActive ? 'Active' : 'Inactive'}
              </span>
            </div>
            
            {isCameraActive && (
              <button onClick={stopCamera} className="btn btn-warning">
                Stop Camera
              </button>
            )}
            
            <button onClick={resetDetection} className="btn btn-secondary">
              Reset Detection
            </button>
          </div>
        </div>

        <div className="detection-info">
          <div className="info-grid">
            <div className="info-item">
              <h4>Status</h4>
              <p>{statusMessage}</p>
            </div>
            <div className="info-item">
              <h4>Detected Sign</h4>
              <p className="detected-sign">{detectedSign}</p>
            </div>
            <div className="info-item">
              <h4>Confidence</h4>
              <p className="confidence">{(confidence * 100).toFixed(1)}%</p>
            </div>
            <div className="info-item">
              <h4>Sequence Length</h4>
              <p>{sequenceLength}/30 frames</p>
            </div>
            <div className="info-item">
              <h4>Predictions</h4>
              <p>{predictionsCount}</p>
            </div>
          </div>

          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${(sequenceLength / 30) * 100}%` }}
            ></div>
          </div>

          {sentence.length > 0 && (
            <div className="sentence-display">
              <h4>Detected Signs:</h4>
              <div className="signs-list">
                {sentence.map((sign, index) => (
                  <span key={index} className="sign-tag">{sign}</span>
                ))}
              </div>
            </div>
          )}

          {allProbabilities.length > 0 && (
            <div className="probability-bars">
              <h4>Sign Probabilities:</h4>
              {allProbabilities.map((prob, index) => {
                const signs = ['hi', 'please', 'excuse_me', 'okay'];
                return (
                  <div key={index} className="probability-bar-item">
                    <span className="probability-label">{signs[index]}:</span>
                    <div className="probability-bar">
                      <div 
                        className="probability-fill" 
                        style={{ width: `${prob * 100}%` }}
                      ></div>
                    </div>
                    <span className="probability-value">{(prob * 100).toFixed(1)}%</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="detection-instructions">
          <h3>How to Use Sign Detection</h3>
          <div className="sign-examples">
            <h4>Supported Signs:</h4>
            <div className="sign-examples-grid">
              <div className="sign-example">
                <strong>Hello/Hi:</strong> Wave hand side to side
              </div>
              <div className="sign-example">
                <strong>Please:</strong> Rub palm in circular motion on chest
              </div>
              <div className="sign-example">
                <strong>Excuse Me:</strong> Tap shoulder to get attention
              </div>
              <div className="sign-example">
                <strong>Okay:</strong> Give a thumbs up gesture
              </div>
            </div>
          </div>
          
          <div className="usage-tips">
            <h4>Tips for Best Results:</h4>
            <ul>
              <li>Ensure good lighting on your hands and face</li>
              <li>Keep your hands clearly visible to the camera</li>
              <li>Hold each sign for a few seconds</li>
              <li>Make sure your full upper body is in frame</li>
              <li>Practice the signs from the Videos page first</li>
              <li>Wait for 30 frames to be collected before detection starts</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SignDetection; 