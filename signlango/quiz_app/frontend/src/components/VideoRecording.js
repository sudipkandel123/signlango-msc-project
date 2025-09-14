import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import config from '../config';
import './VideoRecording.css';

const VideoRecording = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordedVideo, setRecordedVideo] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [selectedSign, setSelectedSign] = useState('');
  const [recordingTime, setRecordingTime] = useState(0);
  const [showPreview, setShowPreview] = useState(false);
  
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  const targetSigns = ['hi', 'please', 'excuse me', 'okay'];

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 640, 
          height: 480,
          facingMode: 'user'
        }, 
        audio: false 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Error accessing camera. Please make sure you have granted camera permissions.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  const startRecording = async () => {
    if (!selectedSign) {
      alert('Please select a sign to record first!');
      return;
    }

    try {
      await startCamera();
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 640, 
          height: 480,
          facingMode: 'user'
        }, 
        audio: false 
      });

      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'video/webm;codecs=vp9'
      });

      chunksRef.current = [];
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'video/webm' });
        const videoUrl = URL.createObjectURL(blob);
        setRecordedVideo(videoUrl);
        setShowPreview(true);
        stopCamera();
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setRecordingTime(0);
      
      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => {
          if (prev >= 4) {
            stopRecording();
            return prev;
          }
          return prev + 1;
        });
      }, 1000);

      // Auto-stop after 4 seconds
      setTimeout(() => {
        if (isRecording) {
          stopRecording();
        }
      }, 4000);

    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Error starting recording. Please try again.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    }
  };

  const analyzeVideo = async () => {
    if (!recordedVideo) {
      alert('No video to analyze!');
      return;
    }

    if (!selectedSign) {
      alert('Please select a sign first!');
      return;
    }

    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      // Convert video URL to blob
      const response = await fetch(recordedVideo);
      const videoBlob = await response.blob();

      console.log('Video blob size:', videoBlob.size, 'bytes');

      // Create FormData
      const formData = new FormData();
      formData.append('video', videoBlob, 'recorded_sign.webm');
      formData.append('target_sign', selectedSign);

      console.log('Sending request to analyze video for sign:', selectedSign);

      const result = await axios.post(`${config.API_BASE_URL}${config.API_ENDPOINTS.ANALYZE_VIDEO}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 second timeout
      });

      console.log('Analysis result received:', result.data);
      setAnalysisResult(result.data);
    } catch (error) {
      console.error('Error analyzing video:', error);
      
      let errorMessage = 'Error analyzing video. Please try again.';
      
      if (error.response) {
        // Server responded with error status
        errorMessage = `Server error: ${error.response.data?.detail || error.response.statusText}`;
      } else if (error.request) {
        // Request was made but no response received
        errorMessage = 'No response from server. Please check your connection.';
      } else {
        // Something else happened
        errorMessage = `Error: ${error.message}`;
      }
      
      alert(errorMessage);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetRecording = () => {
    setRecordedVideo(null);
    setAnalysisResult(null);
    setShowPreview(false);
    setRecordingTime(0);
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }
  };

  const formatTime = (seconds) => {
    return `${seconds}s`;
  };

  return (
    <div className="video-recording-container">
      <div className="video-recording-header">
        <h1>AI Sign Language Detection</h1>
        <p>Record a 3-4 second video of yourself performing one of these signs:</p>
        <div className="sign-options">
          {targetSigns.map((sign) => (
            <button
              key={sign}
              className={`sign-option ${selectedSign === sign ? 'selected' : ''}`}
              onClick={() => setSelectedSign(sign)}
            >
              {sign}
            </button>
          ))}
        </div>
      </div>

      <div className="video-recording-content">
        <div className="video-section">
          <div className="camera-container">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="camera-feed"
            />
            {isRecording && (
              <div className="recording-overlay">
                <div className="recording-indicator">
                  <div className="recording-dot"></div>
                  Recording... {formatTime(recordingTime)}
                </div>
              </div>
            )}
          </div>

          <div className="controls">
            {!isRecording && !recordedVideo && (
              <button
                className="record-button"
                onClick={startRecording}
                disabled={!selectedSign}
              >
                Start Recording
              </button>
            )}
            
            {isRecording && (
              <button className="stop-button" onClick={stopRecording}>
                Stop Recording
              </button>
            )}
          </div>
        </div>

        {showPreview && recordedVideo && (
          <div className="preview-section">
            <h3>Recorded Video Preview</h3>
            <video
              src={recordedVideo}
              controls
              className="preview-video"
            />
            <div className="preview-controls">
              <button
                className="analyze-button"
                onClick={analyzeVideo}
                disabled={isAnalyzing}
              >
                {isAnalyzing ? 'Analyzing...' : 'Analyze with AI'}
              </button>
              <button className="reset-button" onClick={resetRecording}>
                Record Again
              </button>
            </div>
          </div>
        )}

        {analysisResult && (
          <div className="analysis-result">
            <h3>Analysis Result</h3>
            <div className="result-card">
              <div className="result-item">
                <strong>Target Sign:</strong> {analysisResult.target_sign}
              </div>
              <div className="result-item">
                <strong>Detected Sign:</strong> {analysisResult.detected_sign}
              </div>
              <div className="result-item">
                <strong>Confidence:</strong> {analysisResult.confidence}%
              </div>
              <div className="result-item">
                <strong>Feedback:</strong> {analysisResult.feedback}
              </div>
              <div className={`accuracy-indicator ${analysisResult.is_correct ? 'correct' : 'incorrect'}`}>
                {analysisResult.is_correct ? '✓ Correct!' : '✗ Incorrect'}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="instructions">
        <h3>Instructions:</h3>
        <ol>
          <li>Select one of the four signs you want to record</li>
          <li>Position yourself in front of the camera</li>
          <li>Click "Start Recording" and perform the selected sign</li>
          <li>The recording will automatically stop after 4 seconds</li>
          <li>Review your video and click "Analyze with AI"</li>
          <li>Get instant feedback on your sign language performance</li>
        </ol>
      </div>
    </div>
  );
};

export default VideoRecording; 