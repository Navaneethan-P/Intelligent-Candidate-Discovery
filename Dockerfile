FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 5000

# Set Python Path
ENV PYTHONPATH=/app/src

# Run the FastAPI server
CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "5000"]
