

# Multi-stage build for SignLango full-stack application
FROM node:18-alpine AS frontend-build

# Set working directory for frontend
WORKDIR /app/frontend

# Copy frontend package files
COPY signlango/quiz_app/frontend/package*.json ./

# Install frontend dependencies
RUN npm ci --only=production

# Copy frontend source code
COPY signlango/quiz_app/frontend/ ./

# Build the frontend with correct API URL (empty for relative paths)
ENV REACT_APP_API_URL=""
RUN npm run build

# Backend stage
FROM python:3.11-slim AS backend

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy backend requirements and install Python dependencies
COPY signlango/quiz_app/backend/requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend application code
COPY signlango/quiz_app/backend/ ./

# Copy built frontend from frontend-build stage
COPY --from=frontend-build /app/frontend/build ./static/

# Create nginx configuration for serving frontend
RUN echo 'server { \
    listen 80; \
    server_name localhost; \
    \
    # Increase client body size for video uploads \
    client_max_body_size 50M; \
    \
    # Serve frontend static files \
    location / { \
        root /app/static; \
        index index.html; \
        try_files $uri $uri/ /index.html; \
    } \
    \
    # Serve static assets from nested static directory \
    location /static/ { \
        alias /app/static/static/; \
        expires 1y; \
        add_header Cache-Control "public, immutable"; \
    } \
    \
    # Proxy API requests to backend \
    location /api/ { \
        proxy_pass http://localhost:8000/; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
        # Increase timeout for large file uploads \
        proxy_read_timeout 300s; \
        proxy_connect_timeout 75s; \
    } \
    \
    # Proxy all backend endpoints \
    location ~ ^/(facts|quiz|detect-sign|chat|health|common-questions|chat-suggestions|upload-video|upload-bsl-video|api/analyze-video) { \
        proxy_pass http://localhost:8000; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
        # Increase timeout for large file uploads \
        proxy_read_timeout 300s; \
        proxy_connect_timeout 75s; \
    } \
}' > /etc/nginx/sites-available/default

# Create startup script
RUN echo '#!/bin/bash' > /start.sh && \
    echo '# Start backend in background as appuser with proper file limits' >> /start.sh && \
    echo 'su - appuser -c "cd /app && python start_server.py" &' >> /start.sh && \
    echo '# Start nginx in foreground as root' >> /start.sh && \
    echo 'nginx -g "daemon off;"' >> /start.sh && \
    chmod +x /start.sh

# Create nginx directories and set permissions
RUN mkdir -p /var/lib/nginx/body /var/lib/nginx/proxy /var/lib/nginx/fastcgi /var/lib/nginx/uwsgi /var/lib/nginx/scgi /var/log/nginx /var/cache/nginx && \
    chown -R www-data:www-data /var/lib/nginx /var/log/nginx /var/cache/nginx && \
    chmod -R 755 /var/lib/nginx /var/log/nginx /var/cache/nginx
    

# Create a non-root user for the app
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Expose port 80
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Start both services
CMD ["/start.sh"]
