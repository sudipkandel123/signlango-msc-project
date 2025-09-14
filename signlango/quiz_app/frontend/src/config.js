// API Configuration
const config = {
  // Use environment variable if available, otherwise fallback to current domain for production
  API_BASE_URL: process.env.REACT_APP_API_URL || '',
  // API endpoints with correct paths (no /api prefix needed)
  API_ENDPOINTS: {
    FACTS: '/facts',
    QUIZ: '/quiz',
    DETECT: '/detect-sign',
    SIGNS: '/signs',
    HEALTH: '/health',
    CHAT: '/chat',
    CHAT_SUGGESTIONS: '/chat-suggestions',
    COMMON_QUESTIONS: '/common-questions',
    UPLOAD_VIDEO: '/upload-video',
    UPLOAD_BSL_VIDEO: '/upload-bsl-video',
    ANALYZE_VIDEO: '/api/analyze-video'
  }
};

export default config;
