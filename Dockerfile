FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set Python path for module imports
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 5000

# Default: run the FastAPI dashboard server
CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "5000"]
