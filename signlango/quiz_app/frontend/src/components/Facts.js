import React, { useState, useEffect } from 'react';
import axios from 'axios';
import config from '../config';
import './Facts.css';

function Facts() {
  const [facts, setFacts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchFacts = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${config.API_BASE_URL}${config.API_ENDPOINTS.FACTS}`);
        // Handle the response format where data is wrapped in 'facts' property
        const factsData = response.data.facts || response.data;
        setFacts(factsData);
        setError(null);
      } catch (err) {
        console.error('Error fetching facts:', err);
        setError('Failed to load facts. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    fetchFacts();
  }, []);

  if (loading) {
    return (
      <div className="facts">
        <div className="card">
          <h1>BSL Facts</h1>
          <div className="loading-spinner">
            <div className="spinner"></div>
            <p>Loading facts...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="facts">
        <div className="card">
          <h1>BSL Facts</h1>
          <div className="error-message">
            <p>Error: {error}</p>
            <button onClick={() => window.location.reload()} className="retry-button">
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="facts">
      <div className="card">
        <h1>BSL Facts</h1>
        <p className="facts-intro">Learn interesting facts about British Sign Language and deaf culture</p>
        
        <div className="facts-grid">
          {facts.map((fact) => (
            <div key={fact.id} className="fact-card">
              <div className="fact-image-container">
                <img 
                  src={fact.image || '/images/placeholder.jpg'} 
                  alt={fact.title}
                  className="fact-image"
                  onError={(e) => {
                    e.target.style.display = 'none';
                    e.target.nextSibling.style.display = 'block';
                  }}
                />
                <div className="fact-image-placeholder" style={{ display: 'none' }}>
                  <span className="placeholder-icon">{fact.icon}</span>
                </div>
              </div>
              <div className="fact-content">
                <h3>{fact.title}</h3>
                <p>{fact.content}</p>
                <div className="fact-category">
                  <span className="category-tag">{fact.category}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default Facts; 