#!/bin/bash

set -e

echo "## .....Waiting for MongoDB to be ready.... ##"
max_attempts=30
attempt=0

# Loop until Mongo responds to a ping
while [ $attempt -lt $max_attempts ]; do
    if python3 -c "from pymongo import MongoClient; import os; MongoClient(os.getenv('MONGODB_URI'), serverSelectionTimeoutMS=2000).admin.command('ping')" 2>/dev/null; then
        echo "#####..... MongoDB is ready! .....####"
        break
    fi
    attempt=$((attempt + 1))
    echo "   Attempt $attempt/$max_attempts - MongoDB not ready..."
    sleep 2
done

echo "#####..... Running Database Seeding..... ####"

python3 scripts/seed_database.py

echo "...Starting Application..."

exec "$@"