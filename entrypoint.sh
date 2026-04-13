#!/bin/bash
# Wait for MongoDB, seed DB, then exec the container command (e.g. Gunicorn + Uvicorn for FastAPI).

set -e

echo "=============================================="
echo "Starting Application Initialization"
echo "=============================================="

# Wait for MongoDB to be ready
echo "Waiting for MongoDB to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if python3 -c "from pymongo import MongoClient; MongoClient('${MONGODB_URI}', serverSelectionTimeoutMS=2000).admin.command('ping')" 2>/dev/null; then
        echo "MongoDB is ready."
        break
    fi

    attempt=$((attempt + 1))
    echo "   Attempt $attempt/$max_attempts - MongoDB not ready yet..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "MongoDB failed to become ready after $max_attempts attempts"
    echo "Continuing anyway - application will retry connections"
fi

echo ""
echo "=============================================="
echo "Running database seeding"
echo "=============================================="

# Run database seeding
python3 seed_database.py

echo ""
echo "Starting ASGI server (Gunicorn + Uvicorn)"
exec "$@"
