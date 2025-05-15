FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt torch-requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r torch-requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8667

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8667"]