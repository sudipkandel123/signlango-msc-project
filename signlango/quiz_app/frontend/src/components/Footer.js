import React from 'react';
import './Footer.css';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-content">
        <div className="footer-section">
          <p className="copyright">
            © 2025 All rights reserved SignLango. Made by Sudip Kandel
          </p>
        </div>
        <div className="footer-section">
          <div className="social-links">
            <a 
              href="https://linkedin.com/in/sudipkandel123" 
              target="_blank" 
              rel="noopener noreferrer"
              className="social-link"
            >
              LinkedIn
            </a>
            <a 
              href="https://fb.com/sudipf" 
              target="_blank" 
              rel="noopener noreferrer"
              className="social-link"
            >
              Facebook
            </a>
            <a 
              href="https://github.com/sudipkandel123" 
              target="_blank" 
              rel="noopener noreferrer"
              className="social-link"
            >
              GitHub
            </a>
            <a 
              href="https://instagram.com/name_sudip" 
              target="_blank" 
              rel="noopener noreferrer"
              className="social-link"
            >
              Instagram
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer; 