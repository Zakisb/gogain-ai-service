#!/bin/bash

# Script to test Qdrant connectivity and display content within Docker environment

echo "🐳 Testing Qdrant within Docker environment..."
echo "============================================="

# Check if docker-compose is running
if ! docker-compose ps | grep -q "Up"; then
    echo "⚠️ Starting Docker services..."
    docker-compose up -d
    echo "⏳ Waiting for services to start..."
    sleep 10
fi

echo ""
echo "🔍 Running Qdrant connectivity test..."
echo "======================================"

# Run the test script within the Docker environment
# This ensures we're testing from the same network as the services
docker-compose exec app python test_qdrant_full.py

# Alternative: Run from host with Qdrant service name
if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️ First attempt failed, trying alternative approach..."
    docker-compose run --rm app python test_qdrant_full.py
fi

echo ""
echo "✅ Qdrant test completed!"