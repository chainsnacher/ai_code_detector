# AI Code Detection System Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies + curl for healthcheck
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# ✅ FIXED: Install dependencies using the same extra-index-url as your GitHub Actions
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/train data/test data/validation \
    models results logs web_app/static web_app/templates

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose port
EXPOSE 8501

# Health check (Ensure curl is installed above)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "web_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
