import React, { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';

function Videos() {
  const scriptLoaded = useRef(false);

  useEffect(() => {
    // Load SignBSL widget script
    if (!scriptLoaded.current && !window.signbsl) {
      const script = document.createElement('script');
      script.src = 'https://embed.signbsl.com/widgets.js';
      script.async = true;
      script.charset = 'utf-8';
      
      script.onload = () => {
        if (window.signbsl) {
          window.signbsl.init();
        }
      };
      
      script.onerror = () => {
        console.log('SignBSL script failed to load');
      };
      
      document.head.appendChild(script);
      scriptLoaded.current = true;
    } else if (window.signbsl) {
      window.signbsl.init();
    }
  }, []);





  const signVideos = [
    {
      id: 'hello',
      title: 'Hello',
      description: 'Learn how to sign "Hello" in BSL',
      vidref: 'cnyia0upyj',
      link: 'https://www.signbsl.com/sign/hello'
    },
    {
      id: 'please',
      title: 'Please',
      description: 'Learn how to sign "Please" in BSL',
      vidref: 'f7a1xiefkh',
      link: 'https://www.signbsl.com/sign/please'
    },
    {
      id: 'excuse_me',
      title: 'Excuse Me',
      description: 'Learn how to sign "Excuse Me" in BSL',
      vidref: '26ojajoxrq',
      link: 'https://www.signbsl.com/sign/excuse-me'
    },
    {
      id: 'okay',
      title: 'Okay',
      description: 'Learn how to sign "Okay" in BSL',
      vidref: 'cj1jijzqra',
      link: 'https://www.signbsl.com/sign/okay'
    },
    {
      id: 'thank_you',
      title: 'Thank You',
      description: 'Learn how to sign "Thank You" in BSL',
      vidref: 'b7heyqequm',
      link: 'https://www.signbsl.com/sign/thank-you'
    },
    {
      id: 'sorry',
      title: 'Sorry',
      description: 'Learn how to sign "Sorry" in BSL',
      vidref: 'ddt71uvidh',
      link: 'https://www.signbsl.com/sign/sorry'
    },
    {
      id: 'good',
      title: 'Good',
      description: 'Learn how to sign "Good" in BSL',
      vidref: 'z4cxsgwomu',
      link: 'https://www.signbsl.com/sign/good'
    },
    {
      id: 'bad',
      title: 'Bad',
      description: 'Learn how to sign "Bad" in BSL',
      vidref: 'dwzsrkslrq',
      link: 'https://www.signbsl.com/sign/bad'
    }
  ];

  return (
    <div className="videos">
      <div className="card">
        <h1>BSL Video Tutorials</h1>
        <p>Watch professional BSL videos to learn authentic British Sign Language gestures and improve your signing skills</p>

        <div className="learning-tips">
          <h3>BSL Learning Tips</h3>
          <ul>
            <li><strong>Hand Shape:</strong> Pay close attention to the exact finger positions and hand shape for each sign</li>
            <li><strong>Movement:</strong> Practice the smooth, fluid motions that are characteristic of BSL</li>
            <li><strong>Location:</strong> Learn where each sign is performed relative to your body (chin, chest, shoulder, etc.)</li>
            <li><strong>Facial Expression:</strong> BSL uses facial expressions to convey meaning - practice matching expressions to signs</li>
            <li><strong>Mirror Practice:</strong> Use a mirror to check your form and ensure you're signing correctly</li>
            <li><strong>Speed:</strong> Start slowly and gradually increase speed while maintaining accuracy</li>
            <li><strong>Context:</strong> Practice signs in context to understand when and how to use them</li>
          </ul>
        </div>

        <div className="videos-grid">
          {signVideos.map((video) => (
            <div key={video.id} className="video-card">
              <div className="video-header">
                <h3>{video.title}</h3>
                <p>{video.description}</p>
              </div>
              
              <div className="video-container">
                <div className="video-embed-wrapper">
                  <blockquote 
                    className="signbsldata-embed" 
                    data-vidref={video.vidref}
                    style={{
                      borderRadius: '10px',
                      border: 'none',
                      boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)',
                      margin: '0',
                      padding: '0'
                    }}
                  >
                    <a href={video.link}>
                      Watch how to sign '{video.title.toLowerCase()}' in British Sign Language
                    </a>
                  </blockquote>
                </div>
              </div>
              
                              <div className="video-footer">
                  <div className="sign-tips">
                    <h4>Practice Tips</h4>
                    <p>
                      {video.title === 'Hello' && 'Make a gentle wave motion with your hand, palm facing forward. Start from your chin and move outward in a smooth arc. Keep your fingers together and relaxed.'}
                      {video.title === 'Please' && 'Place your flat hand on your chest and make a small circular motion. The movement should be gentle and polite, showing respect and courtesy.'}
                      {video.title === 'Excuse Me' && 'Gently tap your shoulder with your fingertips. Keep your hand open and make a light, respectful tapping motion to get attention politely.'}
                      {video.title === 'Okay' && 'Form a circle with your thumb and index finger, keeping other fingers extended. Hold your hand up with the circle facing forward, showing approval or agreement.'}
                      {video.title === 'Thank You' && 'Touch your chin with your fingertips, then move your hand forward and down. The gesture should convey genuine gratitude and appreciation.'}
                      {video.title === 'Sorry' && 'Make a fist and place it on your chest, then make a small circular motion. The movement should express sincere regret and apology.'}
                      {video.title === 'Good' && 'Start with your hand at your chin, then move it forward and down in a smooth motion. Keep your palm flat and fingers together to show positive approval.'}
                      {video.title === 'Bad' && 'Start with your hand at your chin, then move it downward and away from your body. The motion should clearly indicate disapproval or something negative.'}
                      {!['Hello', 'Please', 'Excuse Me', 'Okay', 'Thank You', 'Sorry', 'Good', 'Bad'].includes(video.title) && 'Practice this sign slowly at first, focusing on the correct hand shape and movement. Gradually increase your speed while maintaining accuracy.'}
                    </p>
                  </div>
                </div>
            </div>
          ))}
        </div>

        <div className="practice-links">
          <h3>Ready to Practice BSL?</h3>
          <p>Now that you've learned the basic signs, test your skills with our interactive features:</p>
          <Link to="/sign-detection" className="btn btn-primary">Try Sign Detection</Link>
          <Link to="/quiz" className="btn btn-secondary">Take BSL Quiz</Link>
          <div className="bsl-note">
            <p><strong>Remember:</strong> BSL is a complete language with its own grammar and structure. These basic signs are just the beginning of your BSL journey!</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Videos; 