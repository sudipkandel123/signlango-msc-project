import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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
        message: input
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

  return (
    <div className="chat">
      <div className="card">
        <h1>BSL Assistant</h1>
        <p>Ask questions about British Sign Language and get helpful answers</p>

        <div className="chat-container">
          <div className="chat-messages">
            {messages.length === 0 && (
              <div className="message assistant-message">
                <div className="message-content">
                  <div className="message-text">
                    Hello! I'm your BSL assistant. Ask me anything about British Sign Language, 
                    deaf culture, or how to improve your signing skills.
                  </div>
                  <div className="message-time">Just now</div>
                </div>
              </div>
            )}

            {messages.map((message, index) => (
              <div key={index} className={`message ${message.sender}-message`}>
                <div className="message-content">
                  <div className="message-text">{message.text}</div>
                  <div className="message-time">{message.timestamp}</div>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="message assistant-message">
                <div className="message-content">
                  <div className="typing-indicator">Assistant is typing...</div>
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
                placeholder="Type your question here..."
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

        <div className="chat-tips">
          <h3>Tips for Better Conversations</h3>
          <ul>
            <li>Ask specific questions about BSL signs and their meanings</li>
            <li>Inquire about deaf culture and communication etiquette</li>
            <li>Get help with sign language grammar and structure</li>
            <li>Learn about accessibility and inclusion practices</li>
            <li>Ask for practice exercises and learning resources</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Chat; 