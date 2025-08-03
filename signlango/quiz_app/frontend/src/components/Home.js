import React from 'react';
import { Link } from 'react-router-dom';

function Home() {
  return (
    <div className="home">
      <div className="hero-section">
        <h1>Welcome to BSL Tutorial</h1>
        <p>Learn British Sign Language through interactive lessons, real-time sign detection, and comprehensive resources.</p>
      </div>

      <div className="features-grid">
        <div className="feature-card">
          <h3>Progress Dashboard</h3>
          <p>Track your BSL learning progress, earn badges, and view detailed statistics.</p>
          <Link to="/dashboard" className="btn">View Dashboard</Link>
        </div>
        
        <div className="feature-card">
          <h3>BSL Facts</h3>
          <p>Learn interesting facts about British Sign Language and deaf culture.</p>
          <Link to="/facts" className="btn">Explore Facts</Link>
        </div>
        
        <div className="feature-card">
          <h3>Video Tutorials</h3>
          <p>Watch professional BSL videos to learn proper signing techniques.</p>
          <Link to="/videos" className="btn">Watch Videos</Link>
        </div>
        
        <div className="feature-card">
          <h3>Interactive Quiz</h3>
          <p>Test your BSL knowledge with our interactive quiz questions.</p>
          <Link to="/quiz" className="btn">Take Quiz</Link>
        </div>
        
        <div className="feature-card">
          <h3>Sign Detection</h3>
          <p>Use your webcam to practice signs and get real-time feedback.</p>
          <Link to="/sign-detection" className="btn">Start Detection</Link>
        </div>
        
        <div className="feature-card">
          <h3>BSL Assistant</h3>
          <p>Chat with our AI assistant to learn more about BSL and get help.</p>
          <Link to="/chat" className="btn">Start Chat</Link>
        </div>
      </div>

      <div className="info-section">
        <h2>About This Application</h2>
        <p>This comprehensive BSL learning platform combines traditional educational content with cutting-edge technology to provide an immersive learning experience. Whether you're a beginner or looking to improve your signing skills, our tools and resources are designed to support your journey in mastering British Sign Language.</p>
      </div>
    </div>
  );
}

export default Home; 