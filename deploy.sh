#!/bin/bash

# Use fixed project path since you gave it:
project_dir="/home/ubuntu/YoloService"

echo "Deploying project in $project_dir..."

cd "$project_dir" || { echo "Directory $project_dir not found!"; exit 1; }

# Build Docker images
echo "Building Docker images..."
docker-compose build || { echo "Docker build failed!"; exit 1; }

# Stop and remove existing containers
echo "Stopping existing containers (if any)..."
docker-compose down

# Start containers in detached mode
echo "Starting containers..."
docker-compose up -d || { echo "Docker Compose up failed!"; exit 1; }

# Give some time for container to start
sleep 5

# Check if the FastAPI app container is running
container_name=$(docker-compose ps -q fastapi-app)

if [ -z "$container_name" ]; then
  echo "❌ Docker container for fastapi-app is not running."
  docker-compose ps
  exit 1
fi

echo "✅ fastapi-app Docker container is running."
echo "Last 10 logs from container:"
docker logs --tail 10 "$container_name"
