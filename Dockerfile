FROM python:3.10-slim

# Install required system packages
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
COPY torch-requirements.txt .
RUN pip install --no-cache-dir -r torch-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app files
COPY . /app
WORKDIR /app

# Run your app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
