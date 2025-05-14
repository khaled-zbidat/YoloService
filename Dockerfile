# Dockerfile for YOLO service
FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8667

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8667"]
