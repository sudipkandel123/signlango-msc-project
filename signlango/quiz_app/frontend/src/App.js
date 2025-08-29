import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from './components/Home';
import Facts from './components/Facts';
import Quiz from './components/Quiz';
import Chat from './components/Chat';
import Videos from './components/Videos';
import Dashboard from './components/Dashboard';
import VideoRecording from './components/VideoRecording';
import Footer from './components/Footer';
import './index.css';

function App() {
  return (
    <Router>
      <div className="App">
        <nav className="navbar">
          <div className="nav-container">
            <Link to="/" className="nav-logo">
              BSL Tutorial
            </Link>
            <div className="nav-links">
              <Link to="/" className="nav-link">Home</Link>
              <Link to="/dashboard" className="nav-link">Dashboard</Link>
              <Link to="/facts" className="nav-link">Facts</Link>
              <Link to="/videos" className="nav-link">Videos</Link>
              <Link to="/quiz" className="nav-link">Quiz</Link>
              <Link to="/ai-detection" className="nav-link">AI Detection</Link>
              <Link to="/chat" className="nav-link">Chat</Link>
            </div>
          </div>
        </nav>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/facts" element={<Facts />} />
            <Route path="/videos" element={<Videos />} />
            <Route path="/quiz" element={<Quiz />} />
            <Route path="/ai-detection" element={<VideoRecording />} />
            <Route path="/chat" element={<Chat />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App; 