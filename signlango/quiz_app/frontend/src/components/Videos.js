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



  const handleVideoClick = (link) => {
    window.open(link, '_blank');
  };

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
        <p>Watch professional BSL videos to learn proper signing techniques and improve your skills</p>

        <div className="learning-tips">
          <h3>Learning Tips</h3>
          <ul>
            <li>Watch each video multiple times to understand the hand movements</li>
            <li>Practice in front of a mirror to check your form</li>
            <li>Pay attention to facial expressions and body language</li>
            <li>Practice regularly to build muscle memory</li>
            <li>Use the Sign Detection feature to test your skills</li>
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
                    {video.title === 'Hello' && 'Practice the wave motion smoothly. Keep your hand relaxed and make the movement natural.'}
                    {video.title === 'Please' && 'Focus on the circular motion on your chest. Make sure the movement is gentle and polite.'}
                    {video.title === 'Excuse Me' && 'Practice the shoulder tap motion. Keep your hand open and make a gentle tapping motion.'}
                    {video.title === 'Okay' && 'Make sure your thumb points upward clearly. Keep your other fingers relaxed.'}
                    {video.title === 'Thank You' && 'Practice the chin touch and forward motion. Make the gesture sincere and appreciative.'}
                    {video.title === 'Sorry' && 'Focus on the circular motion on your chest. Make the gesture show genuine apology.'}
                    {video.title === 'Good' && 'Practice the forward motion from your chin. Keep your hand flat and movement smooth.'}
                    {video.title === 'Bad' && 'Practice the downward motion from your chin. Make the movement clear and deliberate.'}
                    {!['Hello', 'Please', 'Excuse Me', 'Okay', 'Thank You', 'Sorry', 'Good', 'Bad'].includes(video.title) && 'Practice this sign slowly at first, then gradually increase your speed. Focus on accuracy before speed.'}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="practice-links">
          <h3>Ready to Practice?</h3>
          <p>Now that you've watched the videos, test your skills with our interactive features:</p>
          <Link to="/sign-detection" className="btn btn-primary">Try Sign Detection</Link>
          <Link to="/quiz" className="btn btn-secondary">Take BSL Quiz</Link>
        </div>
      </div>
    </div>
  );
}

export default Videos; 