import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chatMode] = useState('general');
  const [showCommonQuestions, setShowCommonQuestions] = useState(false);
  const [commonQuestions, setCommonQuestions] = useState({});

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Load common questions on component mount
    loadCommonQuestions();
  }, []);



  const loadCommonQuestions = async () => {
    try {
      const response = await axios.get('http://localhost:8000/common-questions');
      setCommonQuestions(response.data.questions);
    } catch (error) {
      console.error('Error loading common questions:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = {
      text: input,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/chat', {
        message: input,
        type: chatMode
      });

      const assistantMessage = {
        text: response.data.response,
        sender: 'assistant',
        timestamp: new Date().toLocaleTimeString()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'assistant',
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };



  const handleCommonQuestionClick = async (question) => {
    const userMessage = {
      text: question,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/chat', {
        message: question,
        type: chatMode
      });

      const assistantMessage = {
        text: response.data.response,
        sender: 'assistant',
        timestamp: new Date().toLocaleTimeString()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'assistant',
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const formatMessage = (text) => {
    // Convert markdown-style formatting to HTML
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/\n/g, '<br>')
      .replace(/•/g, '• ');
  };





  return (
    <div className="chat">
      <div className="card">
        <div className="chat-header">
          <h1>BSL Assistant</h1>
          <p>Your guide to British Sign Language</p>
        </div>



        {/* Quick Actions */}
        <div className="quick-actions">
          <button 
            className="action-btn"
            onClick={() => setShowCommonQuestions(!showCommonQuestions)}
          >
            Common Questions
          </button>
          <button 
            className="action-btn"
            onClick={clearChat}
          >
            Clear Chat
          </button>
        </div>



        {/* Common Questions Panel */}
        {showCommonQuestions && (
          <div className="common-questions-panel">
            <h3>Commonly Asked Questions</h3>
            <div className="questions-grid">
              <div className="question-category">
                <h4>General BSL Questions</h4>
                <div className="question-list">
                  {Array.isArray(commonQuestions) && commonQuestions.map((q, index) => (
                    <button
                      key={index}
                      className="question-btn"
                      onClick={() => handleCommonQuestionClick(q.question)}
                    >
                      <span className="question-text">{q.question}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Chat Container */}
        <div className="chat-container">
          <div className="chat-messages">
            {messages.length === 0 && (
              <div className="message assistant-message">
                <div className="message-content">
                  <div className="message-text">
                    <strong>Hello! I'm your BSL assistant.</strong><br /><br />
                    I can help you with:
                    <ul>
                      <li>Learning BSL signs and grammar</li>
                      <li>Practice tips and advice</li>
                      <li>Deaf culture and community</li>
                      <li>Accessibility and inclusion</li>
                      <li>Resources and learning materials</li>
                    </ul>
                    <br />
                    Try the common questions or ask me anything about BSL!
                  </div>
                  <div className="message-time">Just now</div>
                </div>
              </div>
            )}

            {messages.map((message, index) => (
              <div key={index} className={`message ${message.sender}-message`}>
                <div className="message-content">
                  <div 
                    className="message-text"
                    dangerouslySetInnerHTML={{ __html: formatMessage(message.text) }}
                  />
                  <div className="message-time">{message.timestamp}</div>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="message assistant-message">
                <div className="message-content">
                  <div className="typing-indicator">
                    <span>Assistant is typing</span>
                    <span className="dots">...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSubmit} className="chat-input-container">
            <div className="chat-input-wrapper">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask me about BSL..."
                className="chat-input"
                disabled={isLoading}
              />
              <button 
                type="submit" 
                className="send-button"
                disabled={isLoading || !input.trim()}
              >
                Send
              </button>
            </div>
          </form>
        </div>

        {/* Enhanced Tips */}
        <div className="chat-tips">
          <h3>Tips for Better Conversations</h3>
          <div className="tips-grid">
            <div className="tip-item">
              <strong>Basic Signs:</strong> Ask about hello, thank you, please, sorry, goodbye
            </div>
            <div className="tip-item">
              <strong>Numbers & Colors:</strong> Learn to count and describe colors in BSL
            </div>
            <div className="tip-item">
              <strong>Family:</strong> Master family member signs
            </div>
            <div className="tip-item">
              <strong>Grammar:</strong> Understand BSL sentence structure and word order
            </div>
            <div className="tip-item">
              <strong>Finger Spelling:</strong> Learn the BSL alphabet and when to use it
            </div>
            <div className="tip-item">
              <strong>Culture:</strong> Explore Deaf culture and communication etiquette
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Chat; 