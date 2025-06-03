#!/bin/bash

# Use fixed project path since you gave it:
project_dir="/home/ubuntu/YoloService"

echo "🚀 Deploying YOLO Service in $project_dir..."

cd "$project_dir" || { echo "❌ Directory $project_dir not found!"; exit 1; }

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running or user doesn't have permission."
    echo "🔄 Trying with sudo or checking if Docker needs to be started..."
    
    # Try to start Docker service if it's not running
    if ! sudo systemctl is-active --quiet docker; then
        sudo systemctl start docker
        sleep 3
    fi
    
    # If still no permission, use sudo for Docker commands
    if ! docker info >/dev/null 2>&1; then
        echo "⚠️  Using sudo for Docker commands due to permission issues"
        DOCKER_CMD="sudo docker"
        COMPOSE_CMD="sudo docker compose"
    else
        DOCKER_CMD="docker"
        COMPOSE_CMD="docker compose"
    fi
else
    DOCKER_CMD="docker"
    COMPOSE_CMD="docker compose"
fi

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found in $project_dir"
    exit 1
fi

# Stop and remove existing containers
echo "🛑 Stopping existing containers (if any)..."
$COMPOSE_CMD down --remove-orphans

# Clean up old images (optional - keeps only last 2 versions)
echo "🧹 Cleaning up old Docker images..."
$DOCKER_CMD image prune -f

# Build Docker images
echo "🔨 Building Docker images..."
$COMPOSE_CMD build --no-cache || { echo "❌ Docker build failed!"; exit 1; }

# Start containers in detached mode
echo "▶️  Starting containers..."
$COMPOSE_CMD up -d || { echo "❌ Docker Compose up failed!"; exit 1; }

# Give some time for container to start
echo "⏳ Waiting for containers to start..."
sleep 10

# Check if the container is running
container_name=$($COMPOSE_CMD ps -q yolo 2>/dev/null)

if [ -z "$container_name" ]; then
  echo "❌ Docker container for yolo service is not running."
  echo "📋 Current container status:"
  $COMPOSE_CMD ps
  echo "📋 Last 20 logs:"
  $COMPOSE_CMD logs --tail 20 yolo
  exit 1
fi

echo "✅ YOLO service Docker container is running."

# Health check
echo "🔍 Performing health check..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -sf http://localhost:8667/health >/dev/null 2>&1; then
        echo "✅ Health check passed! Service is ready."
        break
    fi
    
    if [ $attempt -eq $max_attempts ]; then
        echo "❌ Health check failed after $max_attempts attempts."
        echo "📋 Container logs:"
        docker logs --tail 20 "$container_name"
        exit 1
    fi
    
    echo "⏳ Health check attempt $attempt/$max_attempts failed, retrying in 2 seconds..."
    sleep 2
    ((attempt++))
done

# Check OpenTelemetry Collector status
echo "📊 Checking OpenTelemetry Collector status..."
if systemctl is-active --quiet otelcol; then
    echo "✅ OpenTelemetry Collector is running"
    
    # Verify metrics endpoint is accessible
    if curl -sf http://localhost:8889/metrics >/dev/null 2>&1; then
        echo "✅ OpenTelemetry Collector metrics endpoint is accessible at http://localhost:8889/metrics"
    else
        echo "⚠️  OpenTelemetry Collector metrics endpoint is not accessible"
        echo "📋 OTC service status:"
        sudo systemctl status otelcol --no-pager || true
    fi
else
    echo "⚠️  OpenTelemetry Collector is not running"
    echo "🔄 Attempting to start OpenTelemetry Collector..."
    sudo systemctl start otelcol || echo "❌ Failed to start OpenTelemetry Collector"
fi

echo "📋 Container status:"
$COMPOSE_CMD ps

echo "📋 Last 10 logs from container:"
$DOCKER_CMD logs --tail 10 "$container_name"

echo ""
echo "🎉 Deployment completed successfully!"
echo "🌐 Service is available at: http://localhost:8667"
echo "❤️  Health check: http://localhost:8667/health"
echo "📊 OpenTelemetry metrics: http://localhost:8889/metrics"