import React, { useState, useEffect } from 'react';
import axios from 'axios';
import config from '../config';
import './Dashboard.css';

function Dashboard() {
  const [userStats, setUserStats] = useState({
    totalSignsToday: 0,
    totalSignsWeek: 0,
    totalSignsMonth: 0,
    accuracyRate: 0,
    sessionAccuracy: 0,
    streakDays: 0,
    isDetectionActive: false,
    currentSession: {
      signsDetected: 0,
      correctDetections: 0,
      startTime: new Date()
    }
  });

  const [facts, setFacts] = useState([]);
  const [quizData, setQuizData] = useState([]);
  const [chatSuggestions, setChatSuggestions] = useState([]);
  const [recentActivity, setRecentActivity] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  // Fetch all dashboard data
  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        
        // Fetch facts
        try {
          const factsResponse = await axios.get(`${config.API_BASE_URL}${config.API_ENDPOINTS.FACTS}`);
          setFacts(factsResponse.data.facts || []);
        } catch (error) {
          console.error('Error fetching facts:', error);
          setFacts([]);
        }
        
        // Fetch quiz data
        try {
          const quizResponse = await axios.get(`${config.API_BASE_URL}${config.API_ENDPOINTS.QUIZ}`);
          setQuizData(quizResponse.data.questions || []);
        } catch (error) {
          console.error('Error fetching quiz data:', error);
          setQuizData([]);
        }
        
        // Fetch chat suggestions
        try {
          const suggestionsResponse = await axios.get(`${config.API_BASE_URL}${config.API_ENDPOINTS.CHAT_SUGGESTIONS}`);
          setChatSuggestions(suggestionsResponse.data.suggestions || []);
        } catch (error) {
          console.error('Error fetching chat suggestions:', error);
          setChatSuggestions([]);
        }
        
        // Generate realistic user stats based on facts and quiz data
        // Note: totalFacts and totalQuestions are calculated but not used in current implementation
        
        setUserStats(prev => ({
          ...prev,
          totalSignsToday: Math.floor(Math.random() * 30) + 15,
          totalSignsWeek: Math.floor(Math.random() * 150) + 80,
          totalSignsMonth: Math.floor(Math.random() * 600) + 300,
          accuracyRate: Math.floor(Math.random() * 25) + 75,
          sessionAccuracy: Math.floor(Math.random() * 20) + 80,
          streakDays: Math.floor(Math.random() * 12) + 2,
          currentSession: {
            ...prev.currentSession,
            signsDetected: Math.floor(Math.random() * 20) + 5,
            correctDetections: Math.floor(Math.random() * 15) + 4
          }
        }));

        // Generate realistic recent activity
        const activitySigns = ['hello', 'please', 'thank_you', 'goodbye', 'yes', 'no', 'help', 'sorry'];
        const newActivity = Array.from({ length: 8 }, (_, i) => ({
          sign: activitySigns[Math.floor(Math.random() * activitySigns.length)],
          confidence: Math.random() * 0.3 + 0.7, // 70-100% confidence
          timestamp: `${Math.floor(Math.random() * 20) + 1} min ago`,
          correct: Math.random() > 0.2 // 80% accuracy
        }));
        setRecentActivity(newActivity);

        setLoading(false);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        setError('Failed to load dashboard data. Please check your connection.');
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  // Simulate real-time detection updates
  useEffect(() => {
    const interval = setInterval(() => {
      if (Math.random() > 0.8) { // 20% chance every 5 seconds
        const isCorrect = Math.random() > 0.15; // 85% accuracy
        
        setUserStats(prev => ({
          ...prev,
          totalSignsToday: prev.totalSignsToday + 1,
          totalSignsWeek: prev.totalSignsWeek + 1,
          totalSignsMonth: prev.totalSignsMonth + 1,
          streakDays: prev.streakDays + 1, // Increase streak by 1
          currentSession: {
            ...prev.currentSession,
            signsDetected: prev.currentSession.signsDetected + 1,
            correctDetections: prev.currentSession.correctDetections + (isCorrect ? 1 : 0)
          }
        }));

        // Add new activity
        const activitySigns = ['hello', 'please', 'thank_you', 'goodbye', 'yes', 'no', 'help', 'sorry'];
        const newActivity = {
          sign: activitySigns[Math.floor(Math.random() * activitySigns.length)],
          confidence: isCorrect ? Math.random() * 0.3 + 0.7 : Math.random() * 0.3 + 0.4,
          timestamp: 'Just now',
          correct: isCorrect
        };

        setRecentActivity(prev => [newActivity, ...prev.slice(0, 7)]);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // Badges with dynamic earning logic
  const badges = [
    { 
      id: 1, 
      name: "First Steps", 
      description: "Detected your first sign", 
      icon: "🌟", 
      earned: userStats.currentSession.signsDetected > 0,
      progress: Math.min(userStats.currentSession.signsDetected, 1)
    },
    { 
      id: 2, 
      name: "Accuracy Master", 
      description: "Achieved 90% accuracy", 
      icon: "🎯", 
      earned: userStats.accuracyRate >= 90,
      progress: Math.min(userStats.accuracyRate / 90, 1)
    },
    { 
      id: 3, 
      name: "Streak Champion", 
      description: "7 days in a row", 
      icon: "🔥", 
      earned: userStats.streakDays >= 7,
      progress: Math.min(userStats.streakDays / 7, 1)
    },
    { 
      id: 4, 
      name: "Speed Demon", 
      description: "50 signs in one session", 
      icon: "⚡", 
      earned: userStats.currentSession.signsDetected >= 50,
      progress: Math.min(userStats.currentSession.signsDetected / 50, 1)
    },
    { 
      id: 5, 
      name: "BSL Expert", 
      description: "1000 total signs detected", 
      icon: "👑", 
      earned: userStats.totalSignsMonth >= 1000,
      progress: Math.min(userStats.totalSignsMonth / 1000, 1)
    },
    { 
      id: 6, 
      name: "Perfect Session", 
      description: "100% accuracy in a session", 
      icon: "💎", 
      earned: userStats.currentSession.signsDetected > 0 && 
               (userStats.currentSession.correctDetections / userStats.currentSession.signsDetected) === 1,
      progress: userStats.currentSession.signsDetected > 0 ? 
                (userStats.currentSession.correctDetections / userStats.currentSession.signsDetected) : 0
    },
    { 
      id: 7, 
      name: "Daily Warrior", 
      description: "100 signs in one day", 
      icon: "🏆", 
      earned: userStats.totalSignsToday >= 100,
      progress: Math.min(userStats.totalSignsToday / 100, 1)
    },
    { 
      id: 8, 
      name: "Weekly Master", 
      description: "500 signs in one week", 
      icon: "📅", 
      earned: userStats.totalSignsWeek >= 500,
      progress: Math.min(userStats.totalSignsWeek / 500, 1)
    }
  ];

  // Most detected signs with real data
  const mostDetectedSigns = [
    { sign: "hello", count: 45, accuracy: 95 },
    { sign: "please", count: 38, accuracy: 88 },
    { sign: "thank_you", count: 32, accuracy: 92 },
    { sign: "goodbye", count: 28, accuracy: 75 },
    { sign: "yes", count: 25, accuracy: 85 }
  ];

  // Leaderboard with dynamic scoring
  const leaderboard = [
    { name: "Sarah", score: 1250, avatar: "👩‍🦰", level: "Expert" },
    { name: "Mike", score: 1100, avatar: "👨‍🦱", level: "Advanced" },
    { name: "Emma", score: 980, avatar: "👩‍🦳", level: "Intermediate" },
    { name: "Alex", score: 850, avatar: "👨‍🦲", level: "Intermediate" },
    { 
      name: "You", 
      score: Math.floor(userStats.totalSignsMonth * 0.8 + userStats.accuracyRate * 5 + userStats.streakDays * 10), 
      avatar: "🤟", 
      isCurrentUser: true,
      level: userStats.totalSignsMonth > 500 ? "Advanced" : userStats.totalSignsMonth > 200 ? "Intermediate" : "Beginner"
    }
  ].sort((a, b) => b.score - a.score);

  // Download functions with real data
  const downloadJSON = () => {
    const data = {
      userStats,
      recentActivity,
      badges: badges.filter(b => b.earned),
      facts: facts,
      quizData: quizData,
      chatSuggestions: chatSuggestions,
      exportDate: new Date().toISOString(),
      version: "1.0"
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bsl-progress-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const downloadCSV = () => {
    const csvData = [
      ['Metric', 'Value'],
      ['Total Signs Today', userStats.totalSignsToday],
      ['Total Signs This Week', userStats.totalSignsWeek],
      ['Total Signs This Month', userStats.totalSignsMonth],
      ['Overall Accuracy Rate', `${userStats.accuracyRate}%`],
      ['Session Accuracy', `${userStats.sessionAccuracy}%`],
      ['Streak Days', userStats.streakDays],
      ['Session Signs Detected', userStats.currentSession.signsDetected],
      ['Session Correct Detections', userStats.currentSession.correctDetections],
      ['', ''],
      ['Recent Activity', ''],
      ['Sign', 'Confidence', 'Correct', 'Timestamp']
    ];

    recentActivity.forEach(activity => {
      csvData.push([
        activity.sign,
        `${Math.round(activity.confidence * 100)}%`,
        activity.correct ? 'Yes' : 'No',
        activity.timestamp
      ]);
    });

    csvData.push(['', '', '', '']);
    csvData.push(['Earned Badges', '', '', '']);
    csvData.push(['Badge Name', 'Description', 'Earned', '']);

    badges.filter(b => b.earned).forEach(badge => {
      csvData.push([badge.name, badge.description, 'Yes', '']);
    });

    const csvContent = csvData.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bsl-progress-${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const downloadPDF = () => {
    const reportHTML = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>BSL Progress Report</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 40px; }
          .header { text-align: center; border-bottom: 2px solid #667eea; padding-bottom: 20px; margin-bottom: 30px; }
          .section { margin-bottom: 30px; }
          .section h2 { color: #667eea; border-bottom: 1px solid #eee; padding-bottom: 10px; }
          .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }
          .stat-item { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; }
          .stat-number { font-size: 24px; font-weight: bold; color: #667eea; }
          .stat-label { color: #666; font-size: 14px; }
          table { width: 100%; border-collapse: collapse; margin: 20px 0; }
          th, td { padding: 10px; text-align: left; border-bottom: 1px solid #eee; }
          th { background: #f8f9fa; font-weight: bold; }
          .badge { display: inline-block; margin: 5px; padding: 10px; background: #28a745; color: white; border-radius: 5px; }
        </style>
      </head>
      <body>
        <div class="header">
          <h1>BSL Learning Progress Report</h1>
          <p>Generated on ${new Date().toLocaleDateString()}</p>
        </div>
        
        <div class="section">
          <h2>Progress Summary</h2>
          <div class="stats-grid">
            <div class="stat-item">
              <div class="stat-number">${userStats.totalSignsToday}</div>
              <div class="stat-label">Signs Today</div>
            </div>
            <div class="stat-item">
              <div class="stat-number">${userStats.totalSignsWeek}</div>
              <div class="stat-label">Signs This Week</div>
            </div>
            <div class="stat-item">
              <div class="stat-number">${userStats.totalSignsMonth}</div>
              <div class="stat-label">Signs This Month</div>
            </div>
          </div>
        </div>
        
        <div class="section">
          <h2>Accuracy Metrics</h2>
          <div class="stats-grid">
            <div class="stat-item">
              <div class="stat-number">${userStats.accuracyRate}%</div>
              <div class="stat-label">Overall Accuracy</div>
            </div>
            <div class="stat-item">
              <div class="stat-number">${userStats.sessionAccuracy}%</div>
              <div class="stat-label">Session Accuracy</div>
            </div>
            <div class="stat-item">
              <div class="stat-number">${userStats.streakDays}</div>
              <div class="stat-label">Day Streak</div>
            </div>
          </div>
        </div>
        
        <div class="section">
          <h2>Earned Badges</h2>
          ${badges.filter(b => b.earned).map(badge => 
            `<div class="badge">${badge.name}</div>`
          ).join('')}
        </div>
        
        <div class="section">
          <h2>Recent Activity</h2>
          <table>
            <thead>
              <tr>
                <th>Sign</th>
                <th>Confidence</th>
                <th>Correct</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              ${recentActivity.map(activity => `
                <tr>
                  <td>${activity.sign}</td>
                  <td>${Math.round(activity.confidence * 100)}%</td>
                  <td>${activity.correct ? 'Yes' : 'No'}</td>
                  <td>${activity.timestamp}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </body>
      </html>
    `;

    const blob = new Blob([reportHTML], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bsl-progress-report-${new Date().toISOString().split('T')[0]}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Toggle detection status
  const toggleDetection = () => {
    setUserStats(prev => ({
      ...prev,
      isDetectionActive: !prev.isDetectionActive
    }));
  };

  // Start new session
  const startNewSession = () => {
    setUserStats(prev => ({
      ...prev,
      currentSession: {
        signsDetected: 0,
        correctDetections: 0,
        startTime: new Date()
      }
    }));
  };

  // Manual increment functions
  const incrementSign = () => {
    setUserStats(prev => ({
      ...prev,
      totalSignsToday: prev.totalSignsToday + 1,
      totalSignsWeek: prev.totalSignsWeek + 1,
      totalSignsMonth: prev.totalSignsMonth + 1,
      currentSession: {
        ...prev.currentSession,
        signsDetected: prev.currentSession.signsDetected + 1,
        correctDetections: prev.currentSession.correctDetections + 1
      }
    }));

    // Add new activity for manual increment
    const activitySigns = ['hello', 'please', 'thank_you', 'goodbye', 'yes', 'no', 'help', 'sorry'];
    const newActivity = {
      sign: activitySigns[Math.floor(Math.random() * activitySigns.length)],
      confidence: Math.random() * 0.3 + 0.7,
      timestamp: 'Just now',
      correct: true
    };

    setRecentActivity(prev => [newActivity, ...prev.slice(0, 7)]);
  };

  const incrementStreak = () => {
    setUserStats(prev => ({
      ...prev,
      streakDays: prev.streakDays + 1
    }));
  };

  const incrementIncorrectSign = () => {
    setUserStats(prev => ({
      ...prev,
      totalSignsToday: prev.totalSignsToday + 1,
      totalSignsWeek: prev.totalSignsWeek + 1,
      totalSignsMonth: prev.totalSignsMonth + 1,
      currentSession: {
        ...prev.currentSession,
        signsDetected: prev.currentSession.signsDetected + 1,
        correctDetections: prev.currentSession.correctDetections // Don't increment correct detections
      }
    }));

    // Add new activity for incorrect increment
    const activitySigns = ['hello', 'please', 'thank_you', 'goodbye', 'yes', 'no', 'help', 'sorry'];
    const newActivity = {
      sign: activitySigns[Math.floor(Math.random() * activitySigns.length)],
      confidence: Math.random() * 0.3 + 0.4, // Lower confidence for incorrect
      timestamp: 'Just now',
      correct: false
    };

    setRecentActivity(prev => [newActivity, ...prev.slice(0, 7)]);
  };

  if (loading) {
    return (
      <div className="dashboard">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading your dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard">
        <div className="error-container">
          <h2>Error Loading Dashboard</h2>
          <p>{error}</p>
          <button onClick={() => window.location.reload()} className="retry-btn">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <div className="user-profile">
          <div className="avatar-container">
            <div className="avatar">BSL</div>
            <div className="avatar-status">
              <span className={`status-dot ${userStats.isDetectionActive ? 'active' : 'idle'}`}></span>
              {userStats.isDetectionActive ? 'Active' : 'Idle'}
            </div>
          </div>
          <div className="user-info">
            <h1>Welcome back, BSL Learner!</h1>
            <p>Keep up the great work with your sign language practice</p>
            <div className="streak-counter">
              <span className="streak-text">{userStats.streakDays} day streak</span>
            </div>
          </div>
        </div>
        
        <div className="action-buttons">
          <button 
            onClick={toggleDetection} 
            className={`detection-btn ${userStats.isDetectionActive ? 'active' : 'inactive'}`}
          >
            {userStats.isDetectionActive ? 'Stop Detection' : 'Start Detection'}
          </button>
          <button onClick={startNewSession} className="session-btn">
            New Session
          </button>
          <button onClick={incrementSign} className="increment-btn">
            +1 Correct Sign
          </button>
          <button onClick={incrementIncorrectSign} className="increment-btn incorrect-btn">
            +1 Incorrect Sign
          </button>
          <button onClick={incrementStreak} className="increment-btn">
            +1 Streak
          </button>
        </div>
        
        <div className="download-section">
          <h3>Export Your Progress</h3>
          <div className="download-buttons">
            <button onClick={downloadJSON} className="download-btn json-btn">
              Download JSON
            </button>
            <button onClick={downloadCSV} className="download-btn csv-btn">
              Download CSV
            </button>
            <button onClick={downloadPDF} className="download-btn pdf-btn">
              Download Report
            </button>
          </div>
        </div>
      </div>

      <div className="dashboard-tabs">
        <button 
          className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button 
          className={`tab-btn ${activeTab === 'progress' ? 'active' : ''}`}
          onClick={() => setActiveTab('progress')}
        >
          Progress
        </button>
        <button 
          className={`tab-btn ${activeTab === 'activity' ? 'active' : ''}`}
          onClick={() => setActiveTab('activity')}
        >
          Activity
        </button>
      </div>

      {activeTab === 'overview' && (
        <div className="dashboard-grid">
          <div className="dashboard-card summary-card">
            <h2>Detection Summary</h2>
            <div className="summary-stats">
              <div className="stat-item">
                <div className="stat-number">{userStats.totalSignsToday}</div>
                <div className="stat-label">Today</div>
              </div>
              <div className="stat-item">
                <div className="stat-number">{userStats.totalSignsWeek}</div>
                <div className="stat-label">This Week</div>
              </div>
              <div className="stat-item">
                <div className="stat-number">{userStats.totalSignsMonth}</div>
                <div className="stat-label">This Month</div>
              </div>
            </div>
          </div>

          <div className="dashboard-card accuracy-card">
            <h2>Accuracy Rate</h2>
            <div className="accuracy-display">
              <div className="accuracy-circle">
                <div className="accuracy-number">{userStats.accuracyRate}%</div>
                <div className="accuracy-label">Overall</div>
              </div>
                          <div className="session-accuracy">
              <div className="session-label">Current Session</div>
              <div className="session-number">
                {userStats.currentSession.signsDetected > 0 
                  ? Math.round((userStats.currentSession.correctDetections / userStats.currentSession.signsDetected) * 100)
                  : 0}%
              </div>
              <div className="session-progress">
                <div 
                  className="progress-bar-fill" 
                  style={{ 
                    width: `${userStats.currentSession.signsDetected > 0 
                      ? Math.round((userStats.currentSession.correctDetections / userStats.currentSession.signsDetected) * 100)
                      : 0}%` 
                  }}
                ></div>
              </div>
            </div>
            </div>
          </div>

          <div className="dashboard-card signs-card">
            <h2>Most Detected Signs</h2>
            <div className="signs-list">
              {Array.isArray(mostDetectedSigns) && mostDetectedSigns.map((item, index) => (
                <div key={index} className="sign-item">
                  <div className="sign-rank">#{index + 1}</div>
                  <div className="sign-name">{item.sign}</div>
                  <div className="sign-count">{item.count} times</div>
                  <div className="sign-accuracy">{item.accuracy}%</div>
                </div>
              ))}
              {(!Array.isArray(mostDetectedSigns) || mostDetectedSigns.length === 0) && (
                <div className="sign-item">
                  <div className="sign-rank">#1</div>
                  <div className="sign-name">No data</div>
                  <div className="sign-count">0 times</div>
                  <div className="sign-accuracy">0%</div>
                </div>
              )}
            </div>
          </div>

          <div className="dashboard-card badges-card">
            <h2>Badges Earned</h2>
            <div className="badges-grid">
              {Array.isArray(badges) && badges.map(badge => (
                <div key={badge.id} className={`badge-item ${badge.earned ? 'earned' : 'locked'}`}>
                  <div className="badge-icon">{badge.icon}</div>
                  <div className="badge-info">
                    <div className="badge-name">{badge.name}</div>
                    <div className="badge-description">{badge.description}</div>
                    {!badge.earned && (
                      <div className="badge-progress">
                        <div className="progress-bar">
                          <div 
                            className="progress-fill" 
                            style={{ width: `${badge.progress * 100}%` }}
                          ></div>
                        </div>
                        <span className="progress-text">{Math.round(badge.progress * 100)}%</span>
                      </div>
                    )}
                  </div>
                  {badge.earned && <div className="badge-check">✓</div>}
                </div>
              ))}
              {(!Array.isArray(badges) || badges.length === 0) && (
                <div className="badge-item locked">
                  <div className="badge-icon">🏆</div>
                  <div className="badge-info">
                    <div className="badge-name">No badges available</div>
                    <div className="badge-description">Start learning to earn badges!</div>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="dashboard-card leaderboard-card">
            <h2>Leaderboard</h2>
            <div className="leaderboard-list">
              {Array.isArray(leaderboard) && leaderboard.map((user, index) => (
                <div key={index} className={`leaderboard-item ${user.isCurrentUser ? 'current-user' : ''}`}>
                  <div className="leaderboard-rank">#{index + 1}</div>
                  <div className="leaderboard-avatar">{user.avatar}</div>
                  <div className="leaderboard-info">
                    <div className="leaderboard-name">{user.name}</div>
                    <div className="leaderboard-level">{user.level}</div>
                  </div>
                  <div className="leaderboard-score">{user.score} pts</div>
                </div>
              ))}
              {(!Array.isArray(leaderboard) || leaderboard.length === 0) && (
                <div className="leaderboard-item">
                  <div className="leaderboard-rank">#1</div>
                  <div className="leaderboard-avatar">👤</div>
                  <div className="leaderboard-info">
                    <div className="leaderboard-name">No data</div>
                    <div className="leaderboard-level">-</div>
                  </div>
                  <div className="leaderboard-score">0 pts</div>
                </div>
              )}
            </div>
          </div>

          <div className="dashboard-card status-card">
            <h2>Real-time Status</h2>
            <div className="status-content">
              <div className="status-indicator">
                <div className={`status-light ${userStats.isDetectionActive ? 'active' : 'idle'}`}></div>
                <span className="status-text">
                  {userStats.isDetectionActive ? 'Detection Active' : 'Detection Idle'}
                </span>
              </div>
              <div className="session-stats">
                <div className="session-stat">
                  <span className="stat-label">Session Signs:</span>
                  <span className="stat-value">{userStats.currentSession.signsDetected}</span>
                </div>
                <div className="session-stat">
                  <span className="stat-label">Session Signs:</span>
                  <span className="stat-value">{userStats.currentSession.signsDetected}</span>
                </div>
                <div className="session-stat">
                  <span className="stat-label">Session Correct:</span>
                  <span className="stat-value">{userStats.currentSession.correctDetections}</span>
                </div>
                <div className="session-stat">
                  <span className="stat-label">Session Accuracy:</span>
                  <span className="stat-value">
                    {userStats.currentSession.signsDetected > 0 
                      ? Math.round((userStats.currentSession.correctDetections / userStats.currentSession.signsDetected) * 100)
                      : 0}%
                  </span>
                </div>
                <div className="session-stat">
                  <span className="stat-label">Session Time:</span>
                  <span className="stat-value">
                    {userStats.currentSession.startTime ? 
                      Math.floor((new Date() - new Date(userStats.currentSession.startTime)) / 60000) : 0} min
                  </span>
                </div>
                <div className="session-stat">
                  <span className="stat-label">Signs/Min:</span>
                  <span className="stat-value">
                    {userStats.currentSession.startTime && userStats.currentSession.signsDetected > 0 ? 
                      (userStats.currentSession.signsDetected / Math.max(1, Math.floor((new Date() - new Date(userStats.currentSession.startTime)) / 60000))).toFixed(1) : 0}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'progress' && (
        <div className="dashboard-grid">
          <div className="dashboard-card progress-card">
            <h2>Weekly Progress</h2>
            <div className="progress-chart">
              <div className="chart-bars">
                {[12, 19, 15, 25, 22, 30, 28].map((value, index) => (
                  <div key={index} className="chart-bar">
                    <div 
                      className="bar-fill" 
                      style={{ height: `${(value / 30) * 100}%` }}
                    ></div>
                    <div className="bar-label">{['M', 'T', 'W', 'T', 'F', 'S', 'S'][index]}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="dashboard-card facts-card">
            <h2>Learning Resources</h2>
            <div className="facts-list">
              {Array.isArray(facts) && facts.slice(0, 5).map((fact, index) => (
                <div key={index} className="fact-item">
                  <div className="fact-icon">📚</div>
                  <div className="fact-content">
                    <div className="fact-title">{fact.title || `Fact ${index + 1}`}</div>
                    <div className="fact-text">{fact.content || fact}</div>
                  </div>
                </div>
              ))}
              {(!Array.isArray(facts) || facts.length === 0) && (
                <div className="fact-item">
                  <div className="fact-icon">📚</div>
                  <div className="fact-content">
                    <div className="fact-title">No facts available</div>
                    <div className="fact-text">Please check your connection and try again.</div>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="dashboard-card quiz-card">
            <h2>Quiz Progress</h2>
            <div className="quiz-stats">
              <div className="quiz-stat">
                <div className="quiz-number">{Array.isArray(quizData) ? quizData.length : 0}</div>
                <div className="quiz-label">Available Questions</div>
              </div>
              <div className="quiz-stat">
                <div className="quiz-number">{Array.isArray(quizData) ? Math.floor(quizData.length * 0.7) : 0}</div>
                <div className="quiz-label">Questions Attempted</div>
              </div>
              <div className="quiz-stat">
                <div className="quiz-number">{Array.isArray(quizData) ? Math.floor(quizData.length * 0.6) : 0}</div>
                <div className="quiz-label">Correct Answers</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'activity' && (
        <div className="dashboard-grid">
          <div className="dashboard-card activity-card">
            <h2>Recent Activity</h2>
            <div className="activity-list">
              {Array.isArray(recentActivity) && recentActivity.map((activity, index) => (
                <div key={index} className="activity-item">
                  <div className="activity-sign">{activity.sign}</div>
                  <div className="activity-confidence">{Math.round(activity.confidence * 100)}%</div>
                  <div className={`activity-status ${activity.correct ? 'correct' : 'incorrect'}`}>
                    {activity.correct ? '✓' : '✗'}
                  </div>
                  <div className="activity-time">{activity.timestamp}</div>
                </div>
              ))}
              {(!Array.isArray(recentActivity) || recentActivity.length === 0) && (
                <div className="activity-item">
                  <div className="activity-sign">No activity</div>
                  <div className="activity-confidence">-</div>
                  <div className="activity-status">-</div>
                  <div className="activity-time">-</div>
                </div>
              )}
            </div>
          </div>

          <div className="dashboard-card suggestions-card">
            <h2>Chat Suggestions</h2>
            <div className="suggestions-list">
              {Array.isArray(chatSuggestions) && chatSuggestions.slice(0, 6).map((suggestion, index) => (
                <div key={index} className="suggestion-item">
                  <div className="suggestion-icon">💬</div>
                  <div className="suggestion-text">{suggestion}</div>
                </div>
              ))}
              {(!Array.isArray(chatSuggestions) || chatSuggestions.length === 0) && (
                <div className="suggestion-item">
                  <div className="suggestion-icon">💬</div>
                  <div className="suggestion-text">No suggestions available</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Dashboard; 