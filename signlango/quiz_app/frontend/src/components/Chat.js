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

  // Only scroll when messages are added, not when other state changes
  useEffect(() => {
    if (messages.length > 0) {
      scrollToBottom();
    }
  }, [messages.length]);

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
    e.stopPropagation();
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
      // Use the enhanced chat endpoint that can handle image generation
      const response = await axios.post('http://localhost:8000/chat-with-image', {
        message: input,
        type: chatMode
      });

      const assistantMessage = {
        text: response.data.response,
        sender: 'assistant',
        timestamp: new Date().toLocaleTimeString(),
        imageData: response.data.image_data,
        imageFormat: response.data.image_format,
        prompt: response.data.prompt
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
      const response = await axios.post('http://localhost:8000/chat-with-image', {
        message: question,
        type: chatMode
      });

      const assistantMessage = {
        text: response.data.response,
        sender: 'assistant',
        timestamp: new Date().toLocaleTimeString(),
        imageData: response.data.image_data,
        imageFormat: response.data.image_format,
        prompt: response.data.prompt
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



  const downloadImage = (imageData, prompt) => {
    try {
      // Convert base64 to blob
      const byteCharacters = atob(imageData);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: 'image/png' });
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `generated-image-${Date.now()}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading image:', error);
      alert('Failed to download image. Please try again.');
    }
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
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      setShowCommonQuestions(!showCommonQuestions);
                    }}
                  >
                    Common Questions
                  </button>
                  <button 
                    className="action-btn"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      clearChat();
                    }}
                  >
                    Clear Chat
                  </button>
        </div>



        {/* Common Questions Panel */}
        {showCommonQuestions && (
          <div className="common-questions-panel" style={{ scrollBehavior: 'auto' }}>
            <h3>Commonly Asked Questions</h3>
            <div className="questions-grid">
              <div className="question-category">
                <h4>General BSL Questions</h4>
                <div className="question-list">
                  {Array.isArray(commonQuestions) && commonQuestions.map((q, index) => (
                    <button
                      key={index}
                      className="question-btn"
                      onClick={(e) => {
                        e.preventDefault();
                        handleCommonQuestionClick(q.question);
                      }}
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
        <div className="chat-container" style={{ scrollBehavior: 'auto' }}>
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
                      <li>Generate images related to BSL and sign language</li>
                    </ul>
                    <br />
                    Try the common questions or ask me anything about BSL! You can also ask me to generate images by saying "generate image of..." or "create image of..."
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
                  
                  {/* Display generated image if available */}
                  {message.imageData && (
                    <div className="generated-image-container">
                      <img 
                        src={`data:${message.imageFormat};base64,${message.imageData}`}
                        alt={message.prompt || "Generated image"}
                        className="generated-image"
                      />
                      <button 
                        className="download-image-btn"
                        onClick={() => downloadImage(message.imageData, message.prompt)}
                      >
                        📥 Download Image
                      </button>
                    </div>
                  )}
                  
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
                placeholder="Ask me about BSL or generate images..."
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


      </div>
    </div>
  );
}

export default Chat; 