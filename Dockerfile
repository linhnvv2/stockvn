
# Dockerfile for VN Stock Watchlist & Alerts (v3.3)
FROM python:3.11-slim

WORKDIR /app

# Install system deps (optional: for timezone/locale)
RUN apt-get update && apt-get install -y --no-install-recommends         tzdata ca-certificates &&         rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip &&         pip install --no-cache-dir -r requirements.txt

COPY app.py ./

EXPOSE 8501

# Streamlit configuration
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0         STREAMLIT_SERVER_PORT=8501         PYTHONUNBUFFERED=1

# Use env file via docker-compose for SMTP secrets
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
