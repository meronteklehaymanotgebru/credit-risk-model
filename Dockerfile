FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project (including mlruns/ if you want to use local tracking)
COPY . .

# Expose the port uvicorn will listen on
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]