import React, { useState, useEffect } from 'react';
import axios from 'axios';

function Dashboard() {
  const [userStats, setUserStats] = useState({
    totalSignsToday: 0,
    totalSignsWeek: 0,
    totalSignsMonth: 0,
    accuracyRate: 0,
    sessionAccuracy: 0,
    mostDetectedSigns: [],
    misclassifiedSigns: [],
    badges: [],
    streakDays: 0,
    isDetectionActive: false,
    currentSession: {
      signsDetected: 0,
      correctDetections: 0,
      startTime: null
    }
  });

  const [leaderboard, setLeaderboard] = useState([
    { name: "Sarah", score: 1250, avatar: "👩‍🦰" },
    { name: "Mike", score: 1100, avatar: "👨‍🦱" },
    { name: "Emma", score: 980, avatar: "👩‍🦳" },
    { name: "Alex", score: 850, avatar: "👨‍🦲" },
    { name: "You", score: 720, avatar: "🤟", isCurrentUser: true }
  ]);

  const [recentActivity, setRecentActivity] = useState([
    { sign: "hello", confidence: 0.95, timestamp: "2 min ago", correct: true },
    { sign: "please", confidence: 0.87, timestamp: "5 min ago", correct: true },
    { sign: "okay", confidence: 0.92, timestamp: "8 min ago", correct: true },
    { sign: "excuse_me", confidence: 0.78, timestamp: "12 min ago", correct: false },
    { sign: "hello", confidence: 0.89, timestamp: "15 min ago", correct: true }
  ]);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate detection activity
      if (Math.random() > 0.7) {
        setUserStats(prev => ({
          ...prev,
          currentSession: {
            ...prev.currentSession,
            signsDetected: prev.currentSession.signsDetected + 1,
            correctDetections: prev.currentSession.correctDetections + (Math.random() > 0.2 ? 1 : 0)
          }
        }));
      }
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const badges = [
    { id: 1, name: "First Steps", description: "Detected your first sign", icon: "🌟", earned: true },
    { id: 2, name: "Accuracy Master", description: "Achieved 90% accuracy", icon: "🎯", earned: true },
    { id: 3, name: "Streak Champion", description: "7 days in a row", icon: "🔥", earned: true },
    { id: 4, name: "Speed Demon", description: "50 signs in one session", icon: "⚡", earned: false },
    { id: 5, name: "BSL Expert", description: "1000 total signs detected", icon: "👑", earned: false },
    { id: 6, name: "Perfect Session", description: "100% accuracy in a session", icon: "💎", earned: false }
  ];

  const progressData = {
    weekly: [12, 19, 15, 25, 22, 30, 28],
    monthly: [120, 145, 180, 220, 195, 250, 280, 320, 290, 350, 380, 420]
  };

  // Download functions
  const downloadJSON = () => {
    const data = {
      userStats,
      recentActivity,
      badges: badges.filter(b => b.earned),
      progressData,
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
    // Create CSV data
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

    // Add recent activity
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

    // Add earned badges
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
    // Create a simple HTML report that can be printed as PDF
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
        
        {/* Download Section */}
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

      <div className="dashboard-grid">
        {/* Detection Summary */}
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

        {/* Accuracy Rate */}
        <div className="dashboard-card accuracy-card">
          <h2>Accuracy Rate</h2>
          <div className="accuracy-display">
            <div className="accuracy-circle">
              <div className="accuracy-number">{userStats.accuracyRate}%</div>
              <div className="accuracy-label">Overall</div>
            </div>
            <div className="session-accuracy">
              <div className="session-label">Current Session</div>
              <div className="session-number">{userStats.sessionAccuracy}%</div>
              <div className="session-progress">
                <div 
                  className="progress-bar-fill" 
                  style={{ width: `${userStats.sessionAccuracy}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>

        {/* Most Detected Signs */}
        <div className="dashboard-card signs-card">
          <h2>Most Detected Signs</h2>
          <div className="signs-list">
            {[
              { sign: "hello", count: 45, accuracy: 95 },
              { sign: "please", count: 38, accuracy: 88 },
              { sign: "okay", count: 32, accuracy: 92 },
              { sign: "excuse_me", count: 28, accuracy: 75 },
              { sign: "thank_you", count: 25, accuracy: 85 }
            ].map((item, index) => (
              <div key={index} className="sign-item">
                <div className="sign-rank">#{index + 1}</div>
                <div className="sign-name">{item.sign}</div>
                <div className="sign-count">{item.count} times</div>
                <div className="sign-accuracy">{item.accuracy}%</div>
              </div>
            ))}
          </div>
        </div>

        {/* Badges */}
        <div className="dashboard-card badges-card">
          <h2>Badges Earned</h2>
          <div className="badges-grid">
            {badges.map(badge => (
              <div key={badge.id} className={`badge-item ${badge.earned ? 'earned' : 'locked'}`}>
                <div className="badge-icon">{badge.icon}</div>
                <div className="badge-info">
                  <div className="badge-name">{badge.name}</div>
                  <div className="badge-description">{badge.description}</div>
                </div>
                {badge.earned && <div className="badge-check">✓</div>}
              </div>
            ))}
          </div>
        </div>

        {/* Recent Activity */}
        <div className="dashboard-card activity-card">
          <h2>Recent Activity</h2>
          <div className="activity-list">
            {recentActivity.map((activity, index) => (
              <div key={index} className="activity-item">
                <div className="activity-sign">{activity.sign}</div>
                <div className="activity-confidence">{Math.round(activity.confidence * 100)}%</div>
                <div className={`activity-status ${activity.correct ? 'correct' : 'incorrect'}`}>
                  {activity.correct ? '✓' : '✗'}
                </div>
                <div className="activity-time">{activity.timestamp}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Leaderboard */}
        <div className="dashboard-card leaderboard-card">
          <h2>Leaderboard</h2>
          <div className="leaderboard-list">
            {leaderboard.map((user, index) => (
              <div key={index} className={`leaderboard-item ${user.isCurrentUser ? 'current-user' : ''}`}>
                <div className="leaderboard-rank">#{index + 1}</div>
                <div className="leaderboard-avatar">{user.avatar}</div>
                <div className="leaderboard-name">{user.name}</div>
                <div className="leaderboard-score">{user.score} pts</div>
              </div>
            ))}
          </div>
        </div>

        {/* Real-time Status */}
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
                <span className="stat-label">Session Accuracy:</span>
                <span className="stat-value">
                  {userStats.currentSession.signsDetected > 0 
                    ? Math.round((userStats.currentSession.correctDetections / userStats.currentSession.signsDetected) * 100)
                    : 0}%
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Progress Chart */}
        <div className="dashboard-card progress-card">
          <h2>Weekly Progress</h2>
          <div className="progress-chart">
            <div className="chart-bars">
              {progressData.weekly.map((value, index) => (
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
      </div>
    </div>
  );
}

export default Dashboard; 